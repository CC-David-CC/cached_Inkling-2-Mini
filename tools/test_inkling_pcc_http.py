"""Check native PCC HTTP generation and continuation after a service restart."""
import argparse
import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import uuid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runtime', type=Path)
    parser.add_argument('fixture', type=Path)
    parser.add_argument('mode', choices=('write', 'read'))
    parser.add_argument('--port', type=int, default=18061)
    parser.add_argument('--chat', action='store_true')
    parser.add_argument('--adapter', type=Path)
    args = parser.parse_args()
    expected_contract = None
    if args.adapter:
        import hashlib
        from context_time_travel.inkling_backend import model_contract_id
        expected_contract = model_contract_id(
            adapter_sha256=hashlib.sha256(args.adapter.read_bytes()).hexdigest())
    key = (args.runtime / 'pcc-http.key').read_text().strip()

    def call(path, payload):
        request = Request(f'http://127.0.0.1:{args.port}/cow/{path}',
                          json.dumps(payload).encode(),
                          {'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'})
        try:
            with urlopen(request, timeout=60) as response:
                result = json.load(response)
        except HTTPError as error:
            raise AssertionError(f'{path}: {error.code}: {error.read().decode()}') from error
        assert result['event'] == 'complete', result
        if expected_contract is not None:
            assert result['receipt']['model_contract_id'] == expected_contract, result
        return result

    messages = [{'role': 'user', 'content': 'What is 2 + 2? Reply with just the number.'}]
    namespace = 'text/' if args.chat else ''

    def create(branch):
        prompt = dict(messages=messages) if args.chat else dict(token_ids=[100, 101, 102])
        return call(namespace + 'create', dict(branch=branch, **prompt,
                                               operation_id='op-' + uuid.uuid4().hex, stream=False))

    def generate(branch, revision):
        prompt = dict(messages=messages) if args.chat else dict(rendered_token_ids=[100, 101, 102])
        return call(namespace + 'generate', dict(branch=branch, revision=revision, max_tokens=5,
                                                 **prompt, settings={'temperature': 0}, stream=False,
                                                 operation_id='op-' + uuid.uuid4().hex))

    if args.mode == 'write':
        suffix = uuid.uuid4().hex
        left, right = 'left-' + suffix, 'right-' + suffix
        left_root, right_root = create(left), create(right)
        assert left_root['receipt']['snapshot'] == right_root['receipt']['snapshot']
        answer = generate(left, left_root['receipt']['branch']['revision'])
        assert 1 <= len(answer['answer']['tokens']) <= 5, answer
        if args.chat:
            assert answer['answer']['text'].strip() == '4', answer
        args.fixture.write_text(json.dumps(dict(branch=right, root=right_root, answer=answer)))
        print(f'HTTP create and generate passed: {answer["answer"]["tokens"]}')
    else:
        fixture = json.loads(args.fixture.read_text())
        answer = generate(fixture['branch'], fixture['root']['receipt']['branch']['revision'])
        expected = fixture['answer']
        assert answer['answer']['tokens'] == expected['answer']['tokens'], answer
        assert answer['receipt']['payload_sha256'] == expected['receipt']['payload_sha256'], answer
        forward = answer['answer']['counter_deltas']['forward_tokens']
        generated = len(answer['answer']['tokens'])
        # The chat renderer appends exactly two assistant framing tokens.
        assert forward == generated + (2 if args.chat else 0), answer
        if args.chat:
            assert answer['answer']['text'].strip() == '4', answer
        print(f'HTTP continuation after restart matched: {answer["answer"]["tokens"]}')


if __name__ == '__main__':
    main()
