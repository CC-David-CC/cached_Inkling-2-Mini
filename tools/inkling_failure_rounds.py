"""Evaluate all stations, then take one rank-four AdamW step on current failures."""
import argparse
import ctypes
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer
from inkling_train_head import (HEAD_ADAPTER_HEADER, HEAD_ADAPTER_MAGIC, MODEL_SHA256,
                               initialize, load_backend, runtime_profile, torch, train_step)
from inkling_station_rounds import load_station_fixture
from record_inkling_answers import record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ('repo', 'runtime', 'parent', 'out'):
        parser.add_argument(arg, type=Path)
    args = parser.parse_args()
    # The historical eight-step debit below applies to this specific starting adapter.
    if hashlib.sha256(args.parent.read_bytes()).hexdigest() != 'db5147973453ad4f4b600c48af1c0c950a2ea911295faf46dddfeba5a3c54208':
        parser.error('Expected the original eight-step station AdamW rank-4 parent')
    report = args.repo / 'docs/benchmarks/inkling/station-adapter-paired-answers.csv'
    with report.open() as source:
        baseline_correct = {row['station'] for row in csv.DictReader(source)
                            if row['stations-v1-adamw-r1'].strip().lower().strip(' .') == row['expected']}
    corrections, held, system = load_station_fixture(args.repo)
    cases = corrections + held
    ids = [c['id'] for c in cases]
    args.out.mkdir(exist_ok=False)
    tok = Tokenizer.from_file(str(args.runtime/'tokenizer-source.json'))
    def encode(s):
        return tok.encode(s, add_special_tokens=False).ids
    prefix = encode('<|message_system|><|content_text|>' + system +
                    '<|end_message|><|message_system|><|content_text|>Thinking effort level: 0<|end_message|>')
    expected = [c.get('chosen', c.get('expected')) for c in cases]
    suffix = [encode('<|message_user|><|content_text|>' + c['user'] +
                     '<|end_message|><|message_model|><|content_text|>') for c in cases]
    targets = [encode(label+'<|end_message|>') for label in expected]
    costs = [len(prefix)+len(s)+len(t) for s,t in zip(suffix,targets)]
    eval_cost = sum(len(prefix)+len(s)+12 for s in suffix)
    cap = 100_000_000
    # Debit the historical base evaluation and eight steps that produced the starting parent.
    used = eval_cost + 8*sum(costs[:3])
    features = []
    adapter = args.parent
    prior = set(range(3,54))
    exposed = set(range(3))
    history = []
    round_no = 0
    def open_model(path):
        return load_backend(runtime_profile(args.runtime, capacity=133120, adapter=path))
    def pin(backend,lease):
        backend.consume(lease,prefix)
        frontier=(ctypes.c_float*backend.vocab_size)(*backend.logits(lease))
        if not backend.lib.ctt_pin_shared(lease):
            raise RuntimeError("Could not pin the shared station prompt")
        return frontier
    def reset(backend,lease,frontier):
        if not backend.lib.ctt_reuse_shared(lease,len(prefix),frontier,len(frontier)):
            raise RuntimeError("Could not restore the shared station prompt")
    while True:
        assert used+eval_cost <= cap
        used += eval_cost
        backend=open_model(adapter)
        good=set()
        scored=[]
        try:
            with backend.lease() as lease:
                frontier=pin(backend,lease)
                for i,c in enumerate(cases):
                    reset(backend,lease,frontier)
                    backend.consume(lease,suffix[i])
                    with backend.generate_request(lease,12) as gen:
                        reply=list(gen)
                    answer=tok.decode(reply,skip_special_tokens=False).split('<|end_message|>')[0].strip()
                    ok=answer.lower().strip(' .')==expected[i]
                    if ok:good.add(i)
                    scored.append(dict(id=ids[i],question=c['user'],answer=answer,expected=expected[i],
                                       tokens=reply,exposed=i in exposed,outcome='correct' if ok else 'incorrect',
                                       evidence=f'Station {ids[i][8:]} has label {expected[i]}.'))
        finally:
            backend.close()
        run=f'{args.out.name}-r{round_no:02}'
        record(args.repo,run,dict(metadata=dict(model=MODEL_SHA256,engine='Cached native PCC',
               adapter=hashlib.sha256(adapter.read_bytes()).hexdigest(),settings=dict(temperature=0,max_tokens=12),
               scoring="answer.strip().lower().strip(' .') == expected"),context=system,cases=scored))
        failed=sorted(set(range(54))-good)
        entry=dict(run=run,total_steps=8+round_no,score=len(good),original_fixed=len(good & {0,1,2}),
                   gains=[ids[i] for i in sorted(good-prior)],losses=[ids[i] for i in sorted(prior-good)],
                   never_trained=54-len(exposed),charged_tokens=used)
        history.append(entry)
        print(json.dumps(entry),flush=True)
        if round_no==0:
            assert {ids[i] for i in good} == baseline_correct
        # Charge each full sequence once for feature capture, then again for each fit step.
        feature_cost=sum(costs) if not features else 0
        step_cost=sum(costs[i] for i in failed)
        if not failed or used+feature_cost+step_cost+eval_cost>cap:
            entry['stop']='all correct' if not failed else 'token cap'
            break
        if not features:
            used+=feature_cost
            backend=open_model(None)
            try:
                with backend.lease() as lease:
                    frontier=pin(backend,lease)
                    for i in range(54):
                        reset(backend,lease,frontier)
                        block=suffix[i]
                        batch=[]
                        for target in targets[i]:
                            h,raw=backend.head_features(lease,block)
                            batch.append((torch.frombuffer(h,dtype=torch.float32).clone(),
                                          torch.frombuffer(raw,dtype=torch.float32).clone(),target))
                            block=[target]
                        features.append(batch)
                softcap=backend.logit_softcap
            finally:
                backend.close()
            params=initialize(len(features[0][0][1]),'adamw')
            header=HEAD_ADAPTER_HEADER.pack(HEAD_ADAPTER_MAGIC,4,bytes.fromhex(MODEL_SHA256))
            data=adapter.read_bytes()
            assert data.startswith(header)
            weights=torch.from_numpy(np.frombuffer(data[len(header):],dtype=np.float32).copy()).cuda()
            split=params[0].numel()
            with torch.no_grad():
                params[0].copy_(weights[:split].reshape_as(params[0]))
                params[1].copy_(weights[split:].reshape_as(params[1]))
        x,raw,t=zip(*(row for i in failed for row in features[i]))
        entry['trained_next']=[ids[i] for i in failed]
        entry['loss_before_step']=train_step(params,torch.stack(x).cuda(),torch.stack(raw).cuda(),torch.tensor(t,device='cuda'),softcap)
        used+=step_cost
        exposed.update(failed)
        round_no+=1
        prior=good
        adapter=args.out/f'round-{round_no:02}.iklora'
        with adapter.open('xb') as stream:
            stream.write(header)
            for factor in params[:2]:
                stream.write(factor.detach().cpu().numpy().tobytes())
        (args.out/'rounds.json').write_text(json.dumps(history,indent=2))
    (args.out/'rounds.json').write_text(json.dumps(history,indent=2))


if __name__=='__main__':
    main()
