#!/usr/bin/env python3
"""Train and verify rank-4 output-head LoRA against the pinned SQ-20 engine."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experimental/branching_context/src'))

from context_time_travel.backend_registry import load_backend
from context_time_travel.inkling_backend import (
    HEAD_ADAPTER_HEADER, HEAD_ADAPTER_MAGIC, HEAD_WIDTH, MODEL_SHA256, runtime_profile,
)

RANK = 4
EXAMPLES = (((100, 101, 102, 103), 19), ((201, 202, 203, 204, 205), 20))


def capture(runtime):
    profile = runtime_profile(runtime)
    backend = load_backend(profile)
    try:
        features, logits = [], []
        for ids, _ in EXAMPLES:
            with backend.lease() as lease:
                hidden, raw = backend.head_features(lease, ids)
                features.append(torch.frombuffer(hidden, dtype=torch.float32).clone())
                logits.append(torch.frombuffer(raw, dtype=torch.float32).clone())
        return (torch.stack(features).cuda(), torch.stack(logits).cuda(),
                torch.tensor([target for _, target in EXAMPLES], device='cuda'),
                backend.logit_softcap, profile['execution_profile_id'])
    finally:
        backend.close()


def initialize(vocab, optimizer_name):
    generator = torch.Generator(device='cuda').manual_seed(3407)
    a = torch.nn.Parameter(torch.randn((RANK, HEAD_WIDTH), device='cuda',
                                      generator=generator) / HEAD_WIDTH**0.5)
    b = torch.nn.Parameter(torch.zeros((vocab, RANK), device='cuda'))
    if optimizer_name == 'muon':
        optimizer = torch.optim.Muon([a, b], lr=0.01, weight_decay=0,
                                     adjust_lr_fn='match_rms_adamw')
    else:
        optimizer = torch.optim.AdamW([a, b], lr=0.01, weight_decay=0, fused=False)
    return a, b, optimizer


def train_step(parameters, features, raw, targets, cap):
    a, b, optimizer = parameters
    optimizer.zero_grad(set_to_none=True)
    scores = cap * torch.tanh((raw + (features @ a.T) @ b.T) / cap)
    loss = torch.nn.functional.cross_entropy(scores, targets)
    loss.backward()
    optimizer.step()
    return loss.item()


def equal_state(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(equal_state(left[key], right[key]) for key in left)
    if isinstance(left, (tuple, list)):
        return len(left) == len(right) and all(equal_state(a, b) for a, b in zip(left, right))
    return left == right


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runtime', type=Path)
    parser.add_argument('output', type=Path, help='new private output directory')
    parser.add_argument('--optimizer', choices=('adamw', 'muon'), default='adamw')
    args = parser.parse_args()
    args.output.mkdir(mode=0o700)
    torch.use_deterministic_algorithms(True)
    features, raw, targets, cap, profile_id = capture(args.runtime.resolve(strict=True))
    initial = initialize(raw.shape[1], args.optimizer)
    previous_b = initial[1].detach().clone()
    losses = [train_step(initial, features, raw, targets, cap) for _ in range(2)]
    assert not torch.equal(initial[1], previous_b), 'LoRA update left B unchanged'
    a, b, optimizer = initial
    metadata = dict(optimizer=args.optimizer, steps=2, profile_id=profile_id,
                    base_sha256=MODEL_SHA256, torch_version=str(torch.__version__),
                    rank=RANK, examples=repr(EXAMPLES))
    checkpoint = args.output / 'step-2.pt'
    torch.save(dict(metadata=metadata, a=a.detach(), b=b.detach(),
                    optimizer=optimizer.state_dict()), checkpoint)
    restored = torch.load(checkpoint, weights_only=True)
    assert restored['metadata'] == metadata
    resumed = initialize(raw.shape[1], args.optimizer)
    with torch.no_grad():
        resumed[0].copy_(restored['a'])
        resumed[1].copy_(restored['b'])
    resumed[2].load_state_dict(restored['optimizer'])
    losses.append(train_step(initial, features, raw, targets, cap))
    resume_loss = train_step(resumed, features, raw, targets, cap)
    assert resume_loss == losses[-1]
    assert all(torch.equal(initial[i], resumed[i]) for i in (0, 1))
    assert equal_state(initial[2].state_dict(), resumed[2].state_dict())

    final = args.output / 'adapter.iklora'
    with final.open('xb') as stream:
        stream.write(HEAD_ADAPTER_HEADER.pack(HEAD_ADAPTER_MAGIC, RANK, bytes.fromhex(MODEL_SHA256)))
        for factor in resumed[:2]:
            stream.write(factor.detach().cpu().contiguous().numpy().tobytes())
    report = dict(metadata, verified_steps=3, losses=losses,
                  exact_optimizer_resume=True, adapter_sha256=hashlib.sha256(final.read_bytes()).hexdigest(),
                  peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
    (args.output / 'receipt.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
