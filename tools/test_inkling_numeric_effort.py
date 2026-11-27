"""Exercise numeric reasoning effort, streaming channels and continuation over PCC HTTP."""
import argparse
import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import uuid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runtime', type=Path)
    parser.add_argument('--port', type=int, default=18061)
    args = parser.parse_args()
    headers = {'Authorization': 'Bearer ' + (args.runtime / 'pcc-http.key').read_text().strip(),
               'Content-Type': 'application/json'}
    messages = [{'role': 'user', 'content': 'What is 17 times 23? Answer with the number.'}]

    def call(action, **data):
        data['operation_id'] = 'op-' + uuid.uuid4().hex
        request = Request(f'http://127.0.0.1:{args.port}/cow/text/{action}',
                          json.dumps(data).encode(), headers)
        with urlopen(request, timeout=120) as response:
            if data.get('stream'):
                return [json.loads(line) for line in response]
            return json.load(response)

    def create(effort):
        fields = dict(branch='effort-' + uuid.uuid4().hex, messages=messages, reasoning_effort=effort)
        receipt = call('create', stream=False, **fields)['receipt']
        return dict(fields, revision=receipt['branch']['revision'], settings={'temperature': 0})

    for invalid in (-1, 1, 9, 0.991, True, '0.2', None):
        try:
            create(invalid)
        except HTTPError as error:
            detail = json.load(error).get('error')
            assert error.code == 400 and detail == (
                'reasoning_effort must be a JSON number in [0, 0.99]'
            ), (error.code, detail)
        else:
            raise AssertionError(f'Accepted invalid effort {invalid!r}')

    for effort in (0, 0.01, 0.1, 0.543, 0.99):
        events = call('generate', max_tokens=512, stream=True, **create(effort))
        assert events[-1]['event'] == 'complete'
        answer = events[-1]['answer']
        assert answer['text'].strip() == '391'
        for channel, field in [('text', 'text'), ('reasoning', 'reasoning_content')]:
            assert ''.join(e['text'] for e in events if e['event'] == channel) == answer.get(field, '')
        assert bool(answer.get('reasoning_content')) == bool(effort)

    request = create(0.99)
    prefix = call('generate', max_tokens=8, stream=False, **request)
    suffix = call('resume', branch=request['branch'], revision=prefix['receipt']['branch']['revision'],
                  max_tokens=504, stream=False)
    for key in ('text', 'reasoning_content', 'tokens'):
        assert prefix['answer'][key] + suffix['answer'][key] == answer[key]
    print('Numeric effort, streaming and resumed channels passed')


if __name__ == '__main__':
    main()
