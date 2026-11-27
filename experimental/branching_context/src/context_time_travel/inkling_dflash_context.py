"""Restore Inkling DFlash2's four tapped layers into a BF16 drafter context."""
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import torch


def _validate_context(context, prompt_tokens, saved_tokens):
    if context.ndim != 3 or context.shape[0] != 1 or context.shape[2] != 4 * 2304:
        raise ValueError("Expected one Inkling DFlash2 context with four 2304-wide taps")
    if context.dtype != torch.bfloat16 or not context.is_contiguous():
        raise ValueError("Expected contiguous BF16 drafter context")
    if not 0 < prompt_tokens <= saved_tokens <= context.shape[1]:
        raise ValueError("Invalid checkpoint prompt or saved length")


def save_hidden_context(path, context, *, prompt_tokens, saved_tokens):
    """Atomically publish prefill and continuation taps in the restore layout."""
    _validate_context(context, prompt_tokens, saved_tokens)
    path = Path(path)
    layers = context.detach()[0].view(-1, 4, 2304)
    stream = NamedTemporaryFile(dir=path.parent, delete=False)
    temporary = Path(stream.name)
    try:
        with stream:
            for layer in range(4):
                layers[:prompt_tokens, layer].contiguous().cpu().view(torch.uint16).numpy().tofile(stream)
            layers[prompt_tokens:saved_tokens].cpu().view(torch.uint16).numpy().tofile(stream)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def restore_hidden_context(path, context, *, prompt_tokens, saved_tokens):
    """Load layer-major prefill taps and token-major continuation taps."""
    _validate_context(context, prompt_tokens, saved_tokens)
    path = Path(path)
    elements = saved_tokens * context.shape[2]
    if path.stat().st_size != elements * 2:
        raise ValueError("Drafter checkpoint size differs from saved length")
    packed = torch.from_file(str(path), size=elements, dtype=torch.uint16).view(torch.bfloat16)
    destination = context[0].view(-1, 4, 2304)
    split = prompt_tokens * context.shape[2]
    destination[:prompt_tokens].copy_(packed[:split].reshape(4, prompt_tokens, 2304).transpose(0, 1))
    destination[prompt_tokens:saved_tokens].copy_(packed[split:].reshape(-1, 4, 2304))
