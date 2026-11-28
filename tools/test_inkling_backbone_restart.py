"""Check full-backbone query LoRA gradients, process restart and factor export."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from torch import nn
from torch.nn import functional as F


class QueryAdapter(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.a = nn.Parameter(torch.randn(4, base.in_features, device=base.weight.device) * 0.01)
        self.b = nn.Parameter(torch.zeros(base.out_features, 4, device=base.weight.device))

    def forward(self, x):
        delta = F.linear(F.linear(x.float(), self.a), self.b)
        return self.base(x) + delta.to(x.dtype)


def run_steps(checkpoint, optimizer, work, phase):
    from tokenizers import Tokenizer
    from transformers import Inkling2LoopForConditionalGeneration

    torch.use_deterministic_algorithms(True)
    model, info = Inkling2LoopForConditionalGeneration.from_pretrained(
        checkpoint, local_files_only=True, device_map={'': 'cuda:0'}, dtype=torch.bfloat16,
        output_loading_info=True, attn_implementation='eager')
    if any(info.values()):
        raise ValueError(f'Incomplete model load: {list(k for k, v in info.items() if v)}')
    model.eval().requires_grad_(False)
    torch.manual_seed(960)
    query = model.model.language_model.layers[0].self_attn.q_proj
    query.linear = QueryAdapter(query.linear)
    factors = [query.linear.a, query.linear.b]
    if optimizer == 'muon':
        opt = torch.optim.Muon(factors, lr=0.0001, weight_decay=0, adjust_lr_fn='match_rms_adamw')
    else:
        opt = torch.optim.AdamW(factors, lr=0.0001)
    tokenizer = Tokenizer.from_file(str(checkpoint / 'tokenizer.json'))
    ids = tokenizer.encode('A Python tuple is immutable.', add_special_tokens=False).ids
    inputs = torch.tensor([ids], device='cuda')
    if phase == 'resume':
        saved = torch.load(work / 'first.pt', weights_only=True)
        with torch.no_grad():
            query.linear.a.copy_(saved['a'])
            query.linear.b.copy_(saved['b'])
        opt.load_state_dict(saved['optimizer'])
    for step in range(2 if phase == 'resume' else 1, 3):
        opt.zero_grad(set_to_none=True)
        logits = model(input_ids=inputs, use_cache=False).logits
        loss = F.cross_entropy(logits[:, :-1, :200058].float().flatten(0, 1), inputs[:, 1:].flatten())
        loss.backward()
        for factor in factors:
            assert factor.grad is not None and torch.isfinite(factor.grad).all()
        assert sum(p.grad is not None for p in model.parameters()) == 2
        old = query.linear.b.detach().clone()
        opt.step()
        changed = int(torch.count_nonzero(old != query.linear.b))
        assert torch.isfinite(loss) and changed and all(torch.isfinite(x).all() for x in factors)
        state = dict(a=factors[0].detach().cpu(), b=factors[1].detach().cpu(),
                     optimizer=opt.state_dict(), loss=loss.item())
        torch.save(state, work / ('first.pt' if step == 1 else f'{phase}.pt'))
        print(json.dumps(dict(phase=phase, step=step, loss=loss.item(), changed_b=changed)), flush=True)


def main():
    from safetensors.torch import load_file, save_file
    from inkling_train_head import equal_state

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--optimizer', choices=('adamw', 'muon'), required=True)
    parser.add_argument('--export', type=Path, metavar='FILE',
                        help='Keep the verified safetensors adapter at a new path.')
    parser.add_argument('--worker', nargs=2, metavar=('PHASE', 'DIRECTORY'), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        phase, directory = args.worker
        run_steps(args.checkpoint, args.optimizer, Path(directory), phase)
        return
    if args.export and (args.export.exists() or not args.export.parent.is_dir()):
        parser.error('--export requires a new filename in an existing directory')
    with tempfile.TemporaryDirectory(dir=args.export.parent if args.export else None) as directory:
        work = Path(directory)
        for phase in ('direct', 'resume'):
            subprocess.run([sys.executable, __file__, *sys.argv[1:], '--worker', phase, directory], check=True)
        direct, resumed = [torch.load(work / f'{phase}.pt', map_location='cpu', weights_only=True)
                           for phase in ('direct', 'resume')]
        assert equal_state(direct, resumed)
        factors = {f'base_model.model.model.llm.layers.0.attn.wq_du.lora_{k.upper()}.weight': resumed[k]
                   for k in ('a', 'b')}
        export = str(work / 'adapter_model.safetensors')
        save_file(factors, export)
        assert equal_state(load_file(export), factors)
        result = dict(optimizer=args.optimizer, resume_exact=True, export_exact=True)
        if args.export:
            os.link(export, args.export)
            result['export'] = str(args.export)
        print(json.dumps(result))


if __name__ == '__main__':
    main()
