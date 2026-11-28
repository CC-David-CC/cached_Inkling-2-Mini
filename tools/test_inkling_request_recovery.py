"""Verify rejected requests preserve a live PCC branch and its next generation."""
import argparse
import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from uuid import uuid4


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runtime', type=Path)
    parser.add_argument('--port', type=int, default=18061)
    args = parser.parse_args()
    key = (args.runtime / 'pcc-http.key').read_text().strip()

    def call(path, body=None):
        req = Request(f'http://127.0.0.1:{args.port}/cow/{path}',
                      None if body is None else json.dumps(body).encode(),
                      {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + key})
        with urlopen(req, timeout=60) as response:
            return json.load(response)

    prompt = [100, 101, 102] + [200 + value for value in uuid4().bytes]

    def create():
        return call('create', dict(branch='recovery-' + uuid4().hex, operation_id='op-' + uuid4().hex,
                                   token_ids=prompt, stream=False))['receipt']['branch']

    def generation(head, **changes):
        body = dict(branch=head['name'], revision=head['revision'], operation_id='op-' + uuid4().hex,
                    max_tokens=5, rendered_token_ids=prompt, stream=False,
                    settings={'temperature': 0})
        return call('generate', {**body, **changes})

    control = generation(create())
    head = create()
    rejected = []
    for changes in (dict(max_tokens=-1), dict(revision=head['revision']+1)):
        try:
            generation(head, **changes)
        except HTTPError as error:
            assert 400 <= error.code < 500, error.code
            rejected.append(dict(status=error.code, error=json.load(error)['error']))
        else:
            raise AssertionError(f'Invalid request accepted: {changes}')
        after = call('head?' + urlencode({'branch': head['name']}))
        assert (after['tip'], after['revision']) == (head['tip'], head['revision'])
    answer = generation(head)
    assert answer['answer']['tokens'] == control['answer']['tokens']
    if answer['receipt']['payload_sha256'] != control['receipt']['payload_sha256']:
        repeat = generation(create())
        raise AssertionError(json.dumps(dict(
            error='Recovered checkpoint differs from independent control',
            rejected=rejected, preserved_tip=head['tip'],
            recovered_payload=answer['receipt']['payload_sha256'],
            control_payload=control['receipt']['payload_sha256'],
            repeated_control_payload=repeat['receipt']['payload_sha256'],
            repeated_control_tokens_match=repeat['answer']['tokens'] == control['answer']['tokens'],
            profile=answer['receipt'].get('execution_profile_id'),
        )))
    print(json.dumps(dict(rejected_status=rejected, preserved_tip=head['tip'],
                          surviving_tokens=answer['answer']['tokens'],
                          surviving_payload=answer['receipt']['payload_sha256'])))


if __name__ == '__main__':
    main()
