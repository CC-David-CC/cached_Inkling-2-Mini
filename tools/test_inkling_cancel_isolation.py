"""Exercise disconnect cancellation with a queued request or a second decoding worker on a live Inkling PCC server."""
import argparse
import concurrent.futures
import json
import time
import threading
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
    parser.add_argument('--other-port', type=int, help='independent server on which the surviving request must emit text before disconnect')
    parser.add_argument('--other-runtime', type=Path, help='read the second server key from this runtime')
    args = parser.parse_args()
    if args.other_runtime is not None and args.other_port is None:
        parser.error('--other-runtime requires --other-port')
    if args.other_port == args.port:
        parser.error('--other-port must identify a different endpoint')
    survivor_port = args.other_port or args.port
    key = (args.runtime / 'pcc-http.key').read_text().strip()
    survivor_key = ((args.other_runtime / 'pcc-http.key').read_text().strip()
                    if args.other_runtime is not None else key)

    def call(path, payload=None, *, port=None):
        request_key = survivor_key if port == survivor_port else key
        headers = {'Authorization': 'Bearer ' + request_key, 'Content-Type': 'application/json'}
        return urlopen(Request(f'http://127.0.0.1:{port or args.port}/cow/' + path, None if payload is None else json.dumps(payload).encode(), headers), timeout=30)

    def setup(prompt, limit, *, port=None):
        if args.trial_reference:
            prompt += '\nTrial reference: ' + args.trial_reference
        messages = [dict(role='user', content=prompt)]
        branch = uid()
        with call('text/create', dict(branch=branch, messages=messages, stream=False, operation_id=uid()), port=port) as r:
            head = json.load(r)['receipt']['branch']
        return head, dict(branch=branch, messages=messages, revision=head['revision'], settings={'temperature':0}, operation_id=uid(), max_tokens=limit, stream=True)

    def complete(body, *, port=None, visible=None):
        with call('text/generate', body, port=port) as r:
            for line in r:
                e = json.loads(line)
                if e['event'] == 'text' and visible is not None:
                    visible.set()
                if e['event'] == 'complete':
                    return e
        raise RuntimeError('Incomplete response')

    prompt = 'What is 9 + 8? Answer with just the number.'
    limit = 128 if args.other_port else 8
    if args.other_port:
        prompt = 'List the integers from 1 to 500 in order, separated by commas.'
    _, baseline = setup(prompt, limit, port=survivor_port)
    ref = complete(baseline, port=survivor_port)
    if not args.other_port:
        assert ref['answer']['text'].strip() == '17'
    for i in range(3):
        orig, cancel = setup('List every integer from one to one thousand in order.', 256)
        _, other = setup(prompt, limit, port=survivor_port)
        visible = threading.Event()
        with concurrent.futures.ThreadPoolExecutor(1) as pool:
            with call('text/generate', cancel) as r:
                for line in r:
                    if json.loads(line)['event'] == 'text':
                        break
                else:
                    raise RuntimeError('No output before disconnect')
                future = pool.submit(complete, other, port=survivor_port, visible=visible)
                if args.other_port:
                    assert visible.wait(10), 'Surviving worker did not emit visible text'
                deadline = time.monotonic() + 5
                while True:
                    try:
                        with call('operation?operation=' + other['operation_id'], port=survivor_port) as response:
                            state = json.load(response)['state']
                    except HTTPError as error:
                        if error.code != 404:
                            raise
                    else:
                        assert state != 'completed', state
                        break
                    assert time.monotonic() < deadline, 'Second operation was never registered'
                    time.sleep(.01)
                if args.other_port:
                    assert state == 'running', state
                    with call('operation?operation=' + cancel['operation_id']) as response:
                        cancelled_state = json.load(response)['state']
                    assert cancelled_state == 'running', cancelled_state
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
