"""Run fixed-rank output-head training and greedy 128K station retention checks."""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path

from inkling_train_head import (HEAD_ADAPTER_HEADER, HEAD_ADAPTER_MAGIC, MODEL_SHA256,
                               initialize, load_backend, parent_factors, runtime_profile, torch, train_step)
from record_inkling_answers import record
from tokenizers import Tokenizer


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def load_station_fixture(repo):
    corrections = repo / 'training/wrong_but_corrected/shared-document-128k'
    retention = repo / 'evaluation/retention/shared-document-128k'
    wrong = rows(corrections / 'examples.jsonl')
    held = rows(retention / 'cases.jsonl')
    if len(wrong) != 3 or len(held) != 51 or len({r['id'] for r in wrong + held}) != 54:
        raise ValueError('Expected 3 correction and 51 retention cases with distinct station IDs')
    system = (corrections / 'system-prompt.txt').read_bytes()
    digest = hashlib.sha256(system).hexdigest()
    baseline = json.loads((retention / 'baseline.json').read_text())
    if baseline['system_prompt_sha256'] != digest or any(r['system_prompt_sha256'] != digest for r in wrong):
        raise ValueError('Station register checksum differs from the recorded fixture')
    document = system.decode('utf-8')
    lines = set(document.splitlines())
    for group, field in ((wrong, 'chosen'), (held, 'expected')):
        for case in group:
            station = case['id'].removeprefix('station-')
            if f"Station {station} has label {case[field]}." not in lines:
                raise ValueError(f"Station register does not support the recorded label for {case['id']}")
    return wrong, held, document


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('runtime', 'repo', 'out'):
        parser.add_argument(key, type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--practice-from', type=Path,
                        help='prior rank-four output adapter used as a frozen parent')
    mode.add_argument('--evaluate', type=Path, help='evaluate an existing adapter only')
    args = parser.parse_args()
    wrong, held, system = load_station_fixture(args.repo)
    args.out.mkdir(exist_ok=False)
    tok = Tokenizer.from_file(str(args.runtime/'tokenizer-source.json'))

    def encode(text):
        return tok.encode(text, add_special_tokens=False).ids

    prefix = encode('<|message_system|><|content_text|>' + system +
                    '<|end_message|><|message_system|><|content_text|>Thinking effort level: 0<|end_message|>')
    train = []
    summary = {}
    stages = [(args.evaluate.stem, args.evaluate)] if args.evaluate else [('base', None)]
    for name, adapter in stages:
        profile = runtime_profile(args.runtime, capacity=133120, adapter=adapter)
        backend = load_backend(profile)
        cases = []
        try:
            with backend.lease() as lease:
                backend.consume(lease, prefix)
                frontier = (ctypes.c_float * backend.vocab_size)(*backend.logits(lease))
                if not backend.lib.ctt_pin_shared(lease):
                    raise RuntimeError("Could not pin the shared station prompt")

                def reset():
                    if not backend.lib.ctt_reuse_shared(lease, len(prefix), frontier, len(frontier)):
                        raise RuntimeError("Could not restore the shared station prompt")

                for case in wrong + held:
                    suffix = encode('<|message_user|><|content_text|>' + case['user'] +
                                    '<|end_message|><|message_model|><|content_text|>')
                    reset()
                    backend.consume(lease, suffix)
                    with backend.generate_request(lease, 12) as gen:
                        ids = list(gen)
                    answer = tok.decode(ids, skip_special_tokens=False).split('<|end_message|>')[0].strip()
                    expected = case.get('chosen', case.get('expected'))
                    success = answer.lower().strip(' .') == expected
                    cases.append(dict(id=case['id'], question=case['user'], answer=answer, expected=expected,
                                      evidence=f"Station {case['id'][8:]} has label {expected}.", tokens=ids,
                                      outcome='correct' if success else 'incorrect', training='chosen' in case))
                    if adapter is None and 'chosen' in case:
                        reset()
                        for target in encode(expected + '<|end_message|>'):
                            hidden, raw = backend.head_features(lease, suffix)
                            train.append((torch.tensor(list(hidden)), torch.tensor(list(raw)), target))
                            suffix = [target]
            cap = backend.logit_softcap
        finally:
            backend.close()
        run = args.out.name + '-' + name
        counts = dict(corrected=sum(r['outcome']=='correct' for r in cases[:3]),
                      retained=sum(r['outcome']=='correct' for r in cases[3:]))
        record(args.repo, run, dict(metadata=dict(model=MODEL_SHA256, engine='Cached native PCC', adapter=profile.get('adapter_sha256'),
               settings=dict(capacity=133120, temperature=0, max_tokens=12, profile=profile['execution_profile_id']),
               scoring="answer.strip().lower().strip(' .') == expected"), context=system, cases=cases))
        summary[name] = counts
        (args.out/'scores.json').write_text(json.dumps(summary, indent=2))
        print(run, counts, flush=True)
        if adapter is not None:
            continue
        x, raw, targets = zip(*train)
        x, raw, targets = torch.stack(x).cuda(), torch.stack(raw).cuda(), torch.tensor(targets, device='cuda')
        if args.practice_from:
            payload = args.practice_from.read_bytes()
            parent_a, parent_b = parent_factors(payload, raw.shape[1], raw.device)
            if parent_a.shape[0] != 4:
                raise ValueError("Station practice requires a rank-four parent")
            parent_scores = raw + (x @ parent_a.T) @ parent_b.T
            factors = initialize(raw.shape[1], 'adamw')
            losses = [train_step(factors, x, parent_scores, targets, cap) for _ in range(8)]
            # QR reduces the SVD of the combined update to an eight by eight matrix.
            combined_b = torch.cat((parent_b, factors[1].detach()), dim=1).double()
            combined_a = torch.cat((parent_a, factors[0].detach())).double()
            left, lr = torch.linalg.qr(combined_b)
            right, rr = torch.linalg.qr(combined_a.T)
            u, singular, vh = torch.linalg.svd(lr @ rr.T)
            for label, a, b in (
                    ('merged-r8', combined_a, combined_b),
                    ('collapsed-r4', vh[:4] @ right.T, (left @ u[:, :4]) * singular[:4])):
                path = args.out/(label + '.iklora')
                with path.open('xb') as stream:
                    stream.write(HEAD_ADAPTER_HEADER.pack(HEAD_ADAPTER_MAGIC, a.shape[0], bytes.fromhex(MODEL_SHA256)))
                    for factor in (a, b):
                        stream.write(factor.float().cpu().contiguous().numpy().tobytes())
                stages.append((label, path))
            (args.out/'practice.json').write_text(json.dumps(dict(parent_sha256=hashlib.sha256(payload).hexdigest(),
                singular_values=singular.tolist(), losses=losses, train_ids=[r['id'] for r in wrong])))
            continue
        for optimizer in ('adamw', 'muon'):
            factors = initialize(raw.shape[1], optimizer)
            for round_id in (1, 2):
                # Continue the same active rank. Reset optimizer history between practice rounds.
                if round_id == 2:
                    factors[2].state.clear()
                losses = [train_step(factors, x, raw, targets, cap) for _ in range(8)]
                label = f'{optimizer}-r{round_id}'
                path = args.out/(label + '.iklora')
                with path.open('xb') as stream:
                    stream.write(HEAD_ADAPTER_HEADER.pack(HEAD_ADAPTER_MAGIC, 4, bytes.fromhex(MODEL_SHA256)))
                    for factor in factors[:2]:
                        stream.write(factor.detach().cpu().numpy().tobytes())
                (args.out/(label+'.json')).write_text(json.dumps(dict(losses=losses, rank=4, train_ids=[r['id'] for r in wrong])))
                stages.append((label, path))


if __name__ == '__main__':
    main()
