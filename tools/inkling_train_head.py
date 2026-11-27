#!/usr/bin/env python3
"""Train and verify rank-4 output-head LoRA against the pinned SQ-20 engine."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

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


def save_training_checkpoint(path, state):
    """Publish only a complete optimizer checkpoint."""
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        staged = Path(stream.name)
        try:
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            staged.unlink(missing_ok=True)
            raise
    try:
        os.replace(staged, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        staged.unlink(missing_ok=True)


def restore_parameters(parameters, checkpoint):
    for name, target in zip(('a', 'b'), parameters[:2]):
        value = checkpoint[name]
        if not isinstance(value, torch.Tensor) or value.shape != target.shape or value.dtype != target.dtype:
            raise ValueError(f'Checkpoint factor {name} must have shape {tuple(target.shape)} and dtype {target.dtype}')
    with torch.no_grad():
        parameters[0].copy_(checkpoint['a'])
        parameters[1].copy_(checkpoint['b'])
    parameters[2].load_state_dict(checkpoint['optimizer'])


def parent_factors(payload, vocab, device):
    magic, rank, base = HEAD_ADAPTER_HEADER.unpack_from(payload)
    if (magic != HEAD_ADAPTER_MAGIC or base.hex() != MODEL_SHA256 or not 1 <= rank <= 16 - RANK
            or len(payload) != HEAD_ADAPTER_HEADER.size + rank * (HEAD_WIDTH + vocab) * 4):
        raise ValueError('Invalid parent adapter or combined rank exceeds the serving limit')
    weights = torch.frombuffer(bytearray(payload[HEAD_ADAPTER_HEADER.size:]), dtype=torch.float32).to(device)
    split = rank * HEAD_WIDTH
    return weights[:split].reshape(rank, HEAD_WIDTH), weights[split:].reshape(vocab, rank)


@torch.no_grad()
def export_adapter(output, parameters, parent=None, collapse=False):
    a, b = parameters[:2]
    if parent is not None:
        a = torch.cat((parent[0], a), dim=0)
        b = torch.cat((parent[1], b), dim=1)
    if collapse:
        # Factor the small core of B @ A without materializing the dense head.
        left, lb = torch.linalg.qr(b.double())
        right, ra = torch.linalg.qr(a.T.double())
        u, singular, vh = torch.linalg.svd(lb @ ra.T)
        b = ((left @ u[:, :RANK]) * singular[:RANK]).float()
        a = (vh[:RANK] @ right.T).float()
    final = output / 'adapter.iklora'
    with final.open('xb') as stream:
        stream.write(HEAD_ADAPTER_HEADER.pack(HEAD_ADAPTER_MAGIC, a.shape[0], bytes.fromhex(MODEL_SHA256)))
        for factor in (a, b):
            stream.write(factor.detach().cpu().contiguous().numpy().tobytes())
    return hashlib.sha256(final.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runtime', type=Path)
    parser.add_argument('output', type=Path, help='new private output directory')
    parser.add_argument('--optimizer', choices=('adamw', 'muon'), default=None)
    parser.add_argument('--resume', type=Path, help='continue a saved optimizer checkpoint')
    parser.add_argument('--additional-steps', type=int, default=1, help='updates after --resume')
    parser.add_argument('--parent-adapter', type=Path, help='train a fresh rank-4 residual over a frozen head adapter')
    parser.add_argument('--collapse', action='store_true', help='export the parent plus residual at rank 4')
    args = parser.parse_args()
    if args.collapse and args.parent_adapter is None:
        parser.error('--collapse requires --parent-adapter')
    parent_payload = args.parent_adapter.read_bytes() if args.parent_adapter else None
    parent_hash = hashlib.sha256(parent_payload).hexdigest() if parent_payload is not None else None
    if args.additional_steps < 1 or (not args.resume and args.additional_steps != 1):
        parser.error('--additional-steps requires --resume and a positive count')
    restored = torch.load(args.resume, weights_only=True) if args.resume else None
    if restored is not None:
        saved = restored['metadata']
        if saved['optimizer'] not in ('adamw', 'muon') or args.optimizer not in (None, saved['optimizer']):
            parser.error('checkpoint optimizer mismatch')
        if saved.get('parent_sha256') != parent_hash:
            parser.error('resume requires the original parent adapter')
        args.optimizer = saved['optimizer']
    else:
        args.optimizer = args.optimizer or 'adamw'
    args.output.mkdir(mode=0o700)
    torch.use_deterministic_algorithms(True)
    features, raw, targets, cap, profile_id = capture(args.runtime.resolve(strict=True))
    parent = None
    if parent_payload is not None:
        parent = parent_factors(parent_payload, raw.shape[1], raw.device)
        raw = raw + (features @ parent[0].T) @ parent[1].T
    initial = initialize(raw.shape[1], args.optimizer)
    if args.resume:
        expected = dict(profile_id=profile_id, rank=RANK, base_sha256=MODEL_SHA256,
                        torch_version=str(torch.__version__), examples=repr(EXAMPLES))
        if any(saved.get(k) != v for k, v in expected.items()):
            raise ValueError('Training checkpoint identity differs from this run')
        restore_parameters(initial, restored)
        losses = [train_step(initial, features, raw, targets, cap) for _ in range(args.additional_steps)]
        steps = saved['steps'] + args.additional_steps
        metadata = dict(saved, steps=steps)
        save_training_checkpoint(args.output / f'step-{steps}.pt',
                                 dict(metadata=metadata, a=initial[0].detach(), b=initial[1].detach(),
                                      optimizer=initial[2].state_dict()))
        report = dict(metadata, collapsed=args.collapse, resumed_from_step=saved['steps'], losses=losses,
                      adapter_sha256=export_adapter(args.output, initial, parent, args.collapse))
        (args.output / 'receipt.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report))
        return
    previous_b = initial[1].detach().clone()
    losses = [train_step(initial, features, raw, targets, cap) for _ in range(2)]
    assert not torch.equal(initial[1], previous_b), 'LoRA update left B unchanged'
    a, b, optimizer = initial
    metadata = dict(optimizer=args.optimizer, steps=2, profile_id=profile_id,
                    base_sha256=MODEL_SHA256, torch_version=str(torch.__version__),
                    rank=RANK, examples=repr(EXAMPLES))
    if parent is not None:
        metadata['parent_sha256'] = parent_hash
    checkpoint = args.output / 'step-2.pt'
    save_training_checkpoint(checkpoint, dict(metadata=metadata, a=a.detach(), b=b.detach(),
                                              optimizer=optimizer.state_dict()))
    restored = torch.load(checkpoint, weights_only=True)
    assert restored['metadata'] == metadata
    resumed = initialize(raw.shape[1], args.optimizer)
    restore_parameters(resumed, restored)
    losses.append(train_step(initial, features, raw, targets, cap))
    resume_loss = train_step(resumed, features, raw, targets, cap)
    assert resume_loss == losses[-1]
    assert all(torch.equal(initial[i], resumed[i]) for i in (0, 1))
    assert equal_state(initial[2].state_dict(), resumed[2].state_dict())

    adapter_hash = export_adapter(args.output, resumed, parent, args.collapse)
    report = dict(metadata, collapsed=args.collapse, verified_steps=3, losses=losses,
                  exact_optimizer_resume=True, adapter_sha256=adapter_hash,
                  peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
    (args.output / 'receipt.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
