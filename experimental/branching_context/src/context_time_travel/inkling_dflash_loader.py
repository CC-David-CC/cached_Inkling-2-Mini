"""Load the pinned Inkling DFlash2 reference implementation and inference weights."""
from functools import partial
import hashlib
import json
from pathlib import Path
from types import ModuleType

from safetensors import safe_open
from safetensors.torch import load_file
import torch

from .inkling_dflash_context import propose_draft_block
from .real import digest_file

REFERENCE = 'f55b7fe0a4c0b3073e0f9cdce547cce29f4b8e2168c4d2818760007c43b7651e'
CONFIG = 'd094e435eb939e01fca6297f637c5af35e3da4b552f8e1ab0a181cd278a1cefa'
WEIGHTS = '217980ab00d116693fa5946df7bbfd967caf5fac1f529afd541fbd470ddb54ee'
TARGET = '65e60f51aa1284d4fb022584908c315270334d991828bb774af60584523539d1'


def load_proposer(reference, draft_directory, target_weights, *, device, stage_target=False):
    """Return a callable accepting BF16 target context and a greedy anchor token ID."""
    reference = Path(reference)
    source = reference.read_bytes()
    draft_directory = Path(draft_directory)
    config = (draft_directory / 'config.json').read_bytes()
    weights = draft_directory / 'model.safetensors'
    if hashlib.sha256(source).hexdigest() != REFERENCE or hashlib.sha256(config).hexdigest() != CONFIG:
        raise ValueError('Inkling DFlash2 reference or config fingerprint mismatch')
    if digest_file(weights) != WEIGHTS or digest_file(target_weights) != TARGET:
        raise ValueError('Inkling DFlash2 checkpoint fingerprint mismatch')

    module = ModuleType('inkling_dflash2_reference')
    exec(compile(source, str(reference), 'exec'), module.__dict__)
    model = module.DFlash2DraftModel(module.Qwen3Config.from_dict(json.loads(config)))
    state = load_file(weights, device='cpu')
    for name in ('predecessor_codebook', 'successor_codebook'):
        key = 'candidate_selector.' + name
        state[key + '.weight'] = state.pop(key)
    model.to(dtype=torch.bfloat16).load_state_dict(state, strict=True)
    del state
    model.to(device).eval()
    placement = 'cpu' if stage_target else device
    with safe_open(target_weights, framework='pt') as target:
        embedding = target.get_tensor('model.llm.embed.weight').to(placement)
        norm = target.get_tensor('model.llm.embed_norm.weight').to(device)
        head = target.get_tensor('model.llm.unembed.weight')[:200058].to(placement)
    return partial(propose_draft_block, model, embedding=embedding, embedding_norm=norm, head=head)
