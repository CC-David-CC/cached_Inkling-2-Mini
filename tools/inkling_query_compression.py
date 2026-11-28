"""Compare two unit-scale backbone query rounds with a one-shot SVD merge."""
import argparse
import hashlib,json,torch
from pathlib import Path
from safetensors.torch import load_file
from safetensors import safe_open
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('target_weights', type=Path)
parser.add_argument('adapters', nargs=2, type=Path, metavar='ADAPTER')
args = parser.parse_args()
adapters = args.adapters
a=[];b=[]
for path in adapters:
    factors=load_file(path)
    prefix='base_model.model.model.llm.layers.0.attn.wq_du.lora_'
    a.append(factors[prefix+'A.weight'])
    b.append(factors[prefix+'B.weight'])
with safe_open(args.target_weights,framework='pt') as f:
    base=f.get_tensor('model.llm.layers.0.attn.wq_du.weight')
def digest(t):return hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
parent=(base.float()+b[0]@a[0]).bfloat16()
target=(parent.float()+b[1]@a[1]).bfloat16()
with safe_open(adapters[1], framework='pt') as f:
    if (f.metadata() or {}).get('parent_projection_sha256') != digest(parent):
        raise ValueError('Second adapter requires a different merged parent')
# Rank-8 update: compute its singular values via an 8x8 core.
q,r=torch.linalg.qr(torch.cat(b,dim=1).double())
z,t=torch.linalg.qr(torch.cat(a,dim=0).T.double())
u,s,v=torch.linalg.svd(r@t.T,full_matrices=False)
def merged(rank):
    delta=(q@u[:,:rank]*s[:rank])@(v[:rank]@z.T)
    return (base.double()+delta).bfloat16()
first_changed=parent!=base
survived=first_changed & (parent==target)
results=[]
for rank in (4,8):
    approx=merged(rank)
    results.append(dict(rank=rank,discarded_frobenius_fraction=float(s[rank:].norm()/s.norm()),bf16_different_from_sequential=int((approx!=target).sum()),exactly_retained_among_surviving_round1=int(((approx==target)&survived).sum()),sha256=digest(approx)))
print(json.dumps(dict(rank4_vs_rank8_bf16_differences=int((merged(4)!=merged(8)).sum()),elements=base.numel(),singular_values=s.tolist(),round1_changed=int(first_changed.sum()),round1_survived_sequential=int(survived.sum()),sequential_sha256=digest(target),one_shot=results),indent=2))
