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
    args = parser.parse_args()
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
        return result

    def create(branch):
        return call('create', dict(branch=branch, token_ids=[100, 101, 102],
                                   operation_id="op-" + uuid.uuid4().hex, stream=False))

    def generate(branch, revision):
        return call('generate', dict(branch=branch, revision=revision, max_tokens=5,
                                     rendered_token_ids=[100, 101, 102],
                                     settings={'temperature': 0}, stream=False,
                                     operation_id="op-" + uuid.uuid4().hex))

    if args.mode == 'write':
        suffix = uuid.uuid4().hex
        left, right = 'left-' + suffix, 'right-' + suffix
        left_root, right_root = create(left), create(right)
        assert left_root['receipt']['snapshot'] == right_root['receipt']['snapshot']
        answer = generate(left, left_root['receipt']['branch']['revision'])
        assert 1 <= len(answer['answer']['tokens']) <= 5, answer
        args.fixture.write_text(json.dumps(dict(branch=right, root=right_root, answer=answer)))
        print(f'HTTP create and generate passed: {answer["answer"]["tokens"]}')
    else:
        fixture = json.loads(args.fixture.read_text())
        answer = generate(fixture['branch'], fixture['root']['receipt']['branch']['revision'])
        expected = fixture['answer']
        assert answer['answer']['tokens'] == expected['answer']['tokens'], answer
        assert answer['receipt']['payload_sha256'] == expected['receipt']['payload_sha256'], answer
        assert answer["answer"]["counter_deltas"]["forward_tokens"] == len(answer["answer"]["tokens"])
        print(f'HTTP continuation after restart matched: {answer["answer"]["tokens"]}')


if __name__ == '__main__':
    main()
