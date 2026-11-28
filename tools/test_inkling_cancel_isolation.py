"""Exercise disconnect cancellation with a queued request on a live Inkling PCC server."""
import argparse
import concurrent.futures
import json
import time
import uuid
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

def uid():
    return 'op-' + uuid.uuid4().hex

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runtime', type=Path)
    parser.add_argument('--port', type=int, default=18061)
    parser.add_argument('--trial-reference', help='append a shared reference to both prompts for an isolated run')
    args = parser.parse_args()
    key = (args.runtime / 'pcc-http.key').read_text().strip()

    def call(path, payload=None):
        headers = {'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'}
        return urlopen(Request(f'http://127.0.0.1:{args.port}/cow/' + path, None if payload is None else json.dumps(payload).encode(), headers), timeout=30)

    def setup(prompt, limit):
        if args.trial_reference:
            prompt += '\nTrial reference: ' + args.trial_reference
        messages = [dict(role='user', content=prompt)]
        branch = uid()
        with call('text/create', dict(branch=branch, messages=messages, stream=False, operation_id=uid())) as r:
            head = json.load(r)['receipt']['branch']
        return head, dict(branch=branch, messages=messages, revision=head['revision'], settings={'temperature':0}, operation_id=uid(), max_tokens=limit, stream=True)

    def complete(body):
        with call('text/generate', body) as r:
            for line in r:
                e = json.loads(line)
                if e['event'] == 'complete':
                    return e
        raise RuntimeError('Incomplete response')

    prompt = 'What is 9 + 8? Answer with just the number.'
    _, baseline = setup(prompt, 8)
    ref = complete(baseline)
    assert ref['answer']['text'].strip() == '17'
    for i in range(3):
        orig, cancel = setup('List every integer from one to one thousand in order.', 256)
        _, other = setup(prompt, 8)
        with concurrent.futures.ThreadPoolExecutor(1) as pool:
            with call('text/generate', cancel) as r:
                for line in r:
                    if json.loads(line)['event'] == 'text':
                        break
                else:
                    raise RuntimeError('No output before disconnect')
                future = pool.submit(complete, other)
                deadline = time.monotonic() + 5
                while True:
                    try:
                        with call('operation?operation=' + other['operation_id']) as response:
                            state = json.load(response)['state']
                    except HTTPError as error:
                        if error.code != 404:
                            raise
                    else:
                        assert state != 'completed', state
                        break
                    assert time.monotonic() < deadline, 'Second operation was never registered'
                    time.sleep(.01)
                assert not future.done()
            result = future.result()
        deadline = time.monotonic()+10
        while True:
            with call('operation?operation='+cancel['operation_id']) as r:
                op = json.load(r)
            if op['state']=='cancelled': break
            assert time.monotonic()<deadline, op['state']
            time.sleep(.05)
        with call('head?branch='+cancel['branch']) as r:
            h=json.load(r)
        assert not op['durable'] and (h['revision'],h['tip'])==(orig['revision'],orig['tip'])
        assert result['receipt']['branch']['name']==other['branch']
        assert result['receipt']['payload_sha256']==ref['receipt']['payload_sha256']
        assert result['answer']['tokens']==ref['answer']['tokens']
        print(json.dumps(dict(trial=i,state=op['state'],durable=op['durable'],preserved_tip=h['tip'],surviving_tokens=result['answer']['tokens'],surviving_payload=result['receipt']['payload_sha256'])))


if __name__ == '__main__':
    main()
