"""Check native PCC HTTP generation and continuation after a service restart."""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import struct
from urllib.parse import urlencode
import uuid
import sys


def verify_export(port, key, receipt):
    query = urlencode({k: receipt[k] for k in ('model_contract_id', 'execution_profile_id')})
    url = f"http://127.0.0.1:{port}/cow/cache/checkpoint/{receipt['snapshot']}?{query}"
    request = Request(url, headers={'Authorization': 'Bearer ' + key})
    digest = hashlib.sha256()
    size = 0
    with urlopen(request, timeout=60) as response:
        assert response.headers.get_content_type() == 'application/x-cow-checkpoint'
        length, = struct.unpack('!I', response.read(4))
        envelope = json.loads(response.read(length))
        manifest = envelope['manifest']
        while chunk := response.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    assert manifest['logits_count'] == 200058
    assert size == manifest['state_bytes'] + 4 * manifest['logits_count']
    assert digest.hexdigest() == manifest['payload_sha256'] == receipt['payload_sha256']
    print(f"Verified {size} exported checkpoint bytes against the generation receipt")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runtime', type=Path)
    parser.add_argument('fixture', type=Path)
    parser.add_argument('mode', choices=('write', 'read'))
    parser.add_argument('--port', type=int, default=18061)
    parser.add_argument('--prompt-tokens', type=int, help='length of a deterministic token-ID prompt')
    parser.add_argument('--chat', action='store_true')
    parser.add_argument('--adapter', type=Path)
    parser.add_argument('--pack', choices=('SQ-10', 'SQ-15', 'SQ-20'), help='verify the served model contract for this pack')
    parser.add_argument('--long-copy', action='store_true', help='test the same 1024-token copy workload with any text engine')
    parser.add_argument('--dspark-long', action='store_true', help='test 1024-token greedy chat with the draft engine')
    parser.add_argument('--verify-export', action='store_true',
                        help='Stream and verify the generated full checkpoint from a --cache-transfer server')
    args = parser.parse_args()
    long_copy = args.long_copy or args.dspark_long
    if long_copy:
        args.chat = True
    if args.prompt_tokens is not None and (args.chat or not 1 <= args.prompt_tokens <= 262139):
        parser.error('--prompt-tokens requires token mode and space for five output tokens')
    token_ids = ([100 + i % 1000 for i in range(args.prompt_tokens)]
                 if args.prompt_tokens is not None else [100, 101, 102])
    limit = 1024 if long_copy else 5
    expected_contract = None
    if args.pack or args.adapter:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experimental/branching_context/src'))
        from context_time_travel.inkling_backend import PACK_HASHES, model_contract_id
        expected_contract = model_contract_id(
            adapter_sha256=hashlib.sha256(args.adapter.read_bytes()).hexdigest() if args.adapter else None,
            weights_sha256=PACK_HASHES[args.pack or 'SQ-20'])
    key = (args.runtime / 'pcc-http.key').read_text().strip()

    fixture = json.loads(args.fixture.read_text()) if args.mode == 'read' else None
    if expected_contract is not None or fixture is not None:
        request = Request(f'http://127.0.0.1:{args.port}/cow/info',
                          headers={'Authorization': 'Bearer ' + key})
        with urlopen(request, timeout=10) as response:
            info = json.load(response)
        if expected_contract is not None and info['model_contract_id'] != expected_contract:
            raise ValueError('Selected pack or adapter differs from the serving endpoint')
        if fixture is not None:
            for field in ('model_contract_id', 'execution_profile_id'):
                if info[field] != fixture['answer']['receipt'][field]:
                    raise ValueError(f'Restart endpoint differs from write fixture in {field}')

    def call(path, payload):
        request = Request(f'http://127.0.0.1:{args.port}/cow/{path}',
                          json.dumps(payload).encode(),
                          {'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'})
        try:
            with urlopen(request, timeout=600 if args.prompt_tokens else 60) as response:
                result = json.load(response)
        except HTTPError as error:
            raise AssertionError(f'{path}: {error.code}: {error.read().decode()}') from error
        if path == 'fork':
            return result
        assert result['event'] == 'complete', result
        if expected_contract is not None:
            assert result['receipt']['model_contract_id'] == expected_contract, result
        return result

    messages = [{'role': 'user', 'content': 'What is 2 + 2? Reply with just the number.'}]
    if long_copy:
        messages[0]['content'] = ('Copy every line verbatim. Start with the first line.\n' +
                                  '\n'.join(f'Item {i}: the sample value is {i}.' for i in range(1, 501)))
    namespace = 'text/'  if args.chat else ''

    def create(branch):
        prompt = dict(messages=messages) if args.chat else dict(token_ids=token_ids)
        return call(namespace + 'create', dict(branch=branch, **prompt,
                                               operation_id='op-' + uuid.uuid4().hex, stream=False))

    def generate(branch, revision):
        prompt = dict(messages=messages) if args.chat else dict(rendered_token_ids=token_ids)
        return call(namespace + 'generate', dict(branch=branch, revision=revision, max_tokens=limit,
                                                 **prompt, settings={'temperature': 0}, stream=False,
                                                 operation_id='op-' + uuid.uuid4().hex))

    if args.mode == 'write':
        suffix = uuid.uuid4().hex
        left, right = 'left-' + suffix, 'right-' + suffix
        left_root = create(left)
        if args.prompt_tokens is not None:
            # Branch from the saved long parent without repeating its prefill and staging.
            saved = left_root['receipt']['snapshot']
            branch = call('fork', dict(name=right, snapshot=saved))
            assert branch['tip'] == saved
            right_revision = branch['revision']
        else:
            right_root = create(right)
            assert left_root['receipt']['snapshot'] == right_root['receipt']['snapshot']
            right_revision = right_root['receipt']['branch']['revision']
        answer = generate(left, left_root['receipt']['branch']['revision'])
        assert 1 <= len(answer['answer']['tokens']) <= limit, answer
        if long_copy:
            assert len(answer['answer']['tokens']) == limit
        if args.dspark_long:
            counters = answer['answer']['counter_deltas']
            assert 0 < counters['drafts_accepted'] <= counters['draft_nodes']
        elif args.chat and not long_copy:
            assert answer['answer']['text'].strip() == '4', answer
        fixture = dict(branch=right, revision=right_revision, answer=answer)
        if args.prompt_tokens is None:
            fixture['root'] = right_root
        args.fixture.write_text(json.dumps(fixture))
        print(f'HTTP create and generate passed with {len(answer["answer"]["tokens"])} tokens')
    else:
        revision = fixture['revision'] if 'revision' in fixture else fixture['root']['receipt']['branch']['revision']
        answer = generate(fixture['branch'], revision)
        expected = fixture['answer']
        assert answer['answer']['tokens'] == expected['answer']['tokens'], answer
        assert answer['receipt']['payload_sha256'] == expected['receipt']['payload_sha256'], answer
        forward = answer['answer']['counter_deltas']['forward_tokens']
        generated = len(answer['answer']['tokens'])
        # The chat renderer appends exactly two assistant framing tokens.
        assert forward == generated + (2 if args.chat else 0), answer
        if args.prompt_tokens is not None:
            assert answer['answer']['counter_deltas']['checkpoint_import_bytes'] > 0, answer
        if long_copy:
            assert len(answer['answer']['tokens']) == limit
        if args.dspark_long:
            counters = answer['answer']['counter_deltas']
            assert 0 < counters['drafts_accepted'] <= counters['draft_nodes']
        elif args.chat and not long_copy:
            assert answer['answer']['text'].strip() == '4', answer
        print(json.dumps(dict(result='restart matched', generated=generated, forward=forward,
                              counters=answer['answer']['counter_deltas'], payload=answer['receipt']['payload_sha256'])))

    if args.verify_export:
        verify_export(args.port, key, answer['receipt'])


if __name__ == '__main__':
    main()
