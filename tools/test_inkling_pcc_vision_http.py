"""Check Inkling image upload, branch answer, and exact HTTP restart recovery."""
import argparse
import base64
import hashlib
from io import BytesIO
import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import uuid
import sys

from test_inkling_http import solid_png


def image_cases():
    from PIL import Image

    red, blue = (255, 0, 0), (0, 0, 255)
    for color, left, right in [('red', red, blue), ('blue', blue, red)]:
        raw = base64.b64decode(solid_png(left).split(',', 1)[1])
        yield color, color, 'Is the image red or blue? Answer with one word.', raw

        image = Image.new('RGB', (224, 224), left)
        image.paste(right, (112, 0, 224, 224))
        buffer = BytesIO()
        image.save(buffer, format='PNG')
        yield 'left-' + color, color, 'What color is the LEFT half of the image? Answer with one word.', buffer.getvalue()


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experimental/branching_context/src'))
    from context_time_travel.inkling_backend import SERVING_PACK_HASHES, model_contract_id

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runtime', type=Path)
    parser.add_argument('fixture', type=Path)
    parser.add_argument('mode', choices=('write', 'read'))
    parser.add_argument('--port', type=int, default=18062)
    parser.add_argument('--adapter', type=Path, help='verify the vision contract for this adapter file')
    parser.add_argument('--pack', choices=tuple(SERVING_PACK_HASHES), help='verify the served weight pack')
    args = parser.parse_args()
    expected_contract = None
    if args.adapter or args.pack:
        expected_contract = model_contract_id(
            vision=True,
            adapter_sha256=hashlib.sha256(args.adapter.read_bytes()).hexdigest() if args.adapter else None,
            weights_sha256=SERVING_PACK_HASHES[args.pack or 'SQ-20'],
        )
    key = (args.runtime / 'pcc-http.key').read_text().strip()

    request = Request(f'http://127.0.0.1:{args.port}/cow/info',
                      headers={'Authorization': 'Bearer ' + key})
    with urlopen(request, timeout=10) as response:
        instance = response.headers.get('X-COW-Server-Instance')
        info = json.load(response)
    if not instance:
        raise ValueError('Restart verification requires a server reporting its instance ID')
    if expected_contract and info['model_contract_id'] != expected_contract:
        raise ValueError('Selected pack or adapter differs from the serving endpoint')

    def operation():
        return 'op-' + uuid.uuid4().hex

    def post(path, body, image=False):
        headers = {'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'}
        if image:
            headers.update({'Content-Type': 'image/png',
                            'X-COW-Content-SHA256': hashlib.sha256(body).hexdigest()})
        else:
            body = json.dumps(body).encode()
        req = Request(f'http://127.0.0.1:{args.port}/cow/{path}', body, headers)
        try:
            with urlopen(req, timeout=60) as response:
                return json.load(response)
        except HTTPError as error:
            raise AssertionError(f'{path}: {error.code}: {error.read().decode()}') from error

    def answer(branch, revision, color):
        result = post('text/answer', dict(branch=branch, revision=revision,
                                         operation_id=operation(), max_tokens=32,
                                         settings={'temperature': 0}, stream=False))
        if expected_contract:
            assert result['receipt']['model_contract_id'] == expected_contract, result['receipt']
        assert result['answer']['text'].strip().lower().rstrip('.') == color, result
        assert result['answer']['counter_deltas']['forward_tokens'] == len(result['answer']['tokens']), result
        return result

    if args.mode == 'write':
        namespace = post('namespace', dict(title='Inkling vision test', operation_id=operation()))['namespace_id']
        results = {}
        for case_id, color, question, raw in image_cases():
            uploaded = post('images?namespace=' + namespace, raw, image=True)
            messages = [dict(role='user', content=[
                dict(type='image', image_id=uploaded['image_id']),
                dict(type='text', text=question),
            ])]
            branch = case_id + '-' + uuid.uuid4().hex
            root = post('text/prepare', dict(branch=branch, namespace_id=namespace,
                                            messages=messages, operation_id=operation(), stream=False))
            receipt = root['receipt']
            fork = post('fork', dict(name=branch + '-restart', snapshot=receipt['snapshot'],
                                     source_branch=branch, operation_id=operation()))
            response = answer(branch, receipt['branch']['revision'], color)
            results[case_id] = dict(server_instance=instance, color=color, fork=fork, response=response, root=receipt['snapshot'])
            print(f'{case_id}: {response["answer"]["text"]}')
        assert len({record['root'] for record in results.values()}) == len(results)
        args.fixture.write_text(json.dumps(results))
    else:
        fixtures = json.loads(args.fixture.read_text())
        if set(fixtures) != {case_id for case_id, *_ in image_cases()}:
            raise ValueError('Restart fixture must contain all four vision cases')
        if any(not f.get('server_instance') or f['server_instance'] == instance for f in fixtures.values()):
            raise ValueError('Restart verification requires fixtures from another server instance')
        for case_id, fixture in fixtures.items():
            fork, expected = fixture['fork'], fixture['response']
            if expected_contract and expected['receipt']['model_contract_id'] != expected_contract:
                raise ValueError(f'{case_id} was recorded under a different vision model contract')
            response = answer(fork['name'], fork['revision'], fixture.get('color', case_id))
            assert response['answer']['tokens'] == expected['answer']['tokens'], case_id
            actual_hash = response['receipt']['payload_sha256']
            expected_hash = expected['receipt']['payload_sha256']
            assert actual_hash == expected_hash, (case_id, actual_hash, expected_hash)
            print(f'{case_id} after restart: identical tokens and state hash, zero image or text replay')


if __name__ == '__main__':
    main()
