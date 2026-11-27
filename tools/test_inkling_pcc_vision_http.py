"""Check Inkling image upload, branch answer, and exact HTTP restart recovery."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import uuid

from test_inkling_http import solid_png


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runtime', type=Path)
    parser.add_argument('fixture', type=Path)
    parser.add_argument('mode', choices=('write', 'read'))
    parser.add_argument('--port', type=int, default=18062)
    args = parser.parse_args()
    key = (args.runtime / 'pcc-http.key').read_text().strip()

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
        assert result['answer']['text'].strip().lower().rstrip('.') == color, result
        assert result['answer']['counter_deltas']['forward_tokens'] == len(result['answer']['tokens']), result
        return result

    if args.mode == 'write':
        namespace = post('namespace', dict(title='Inkling vision test', operation_id=operation()))['namespace_id']
        results = {}
        for color, rgb in (('red', (255, 0, 0)), ('blue', (0, 0, 255))):
            raw = base64.b64decode(solid_png(rgb).split(',', 1)[1])
            uploaded = post('images?namespace=' + namespace, raw, image=True)
            messages = [dict(role='user', content=[
                dict(type='image', image_id=uploaded['image_id']),
                dict(type='text', text='Is the image red or blue? Answer with one word.'),
            ])]
            branch = color + '-' + uuid.uuid4().hex
            root = post('text/prepare', dict(branch=branch, namespace_id=namespace,
                                            messages=messages, operation_id=operation(), stream=False))
            receipt = root['receipt']
            fork = post('fork', dict(name=branch + '-restart', snapshot=receipt['snapshot'],
                                     source_branch=branch, operation_id=operation()))
            response = answer(branch, receipt['branch']['revision'], color)
            results[color] = dict(fork=fork, response=response, root=receipt['snapshot'])
            print(f'{color}: {response["answer"]["text"]}')
        assert results['red']['root'] != results['blue']['root']
        args.fixture.write_text(json.dumps(results))
    else:
        for color, fixture in json.loads(args.fixture.read_text()).items():
            fork, expected = fixture['fork'], fixture['response']
            response = answer(fork['name'], fork['revision'], color)
            assert response['answer']['tokens'] == expected['answer']['tokens']
            assert response['receipt']['payload_sha256'] == expected['receipt']['payload_sha256']
            print(f'{color} after restart: identical tokens and state hash, zero image or text replay')


if __name__ == '__main__':
    main()
