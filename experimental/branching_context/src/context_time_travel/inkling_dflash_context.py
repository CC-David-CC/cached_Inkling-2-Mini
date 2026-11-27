"""Restore Inkling DFlash2's four tapped layers into a BF16 drafter context."""
import hashlib
import math
import mmap
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


def _write_rows(stream, rows, digest):
    # Bound transient contiguous GPU and host copies for long prompts.
    values_per_row = math.prod(rows.shape[1:])
    rows_per_chunk = (16 << 20) // (values_per_row * rows.element_size())
    for chunk in rows.split(rows_per_chunk):
        data = memoryview(chunk.contiguous().cpu().view(torch.uint16).numpy()).cast("B")
        stream.write(data)
        digest.update(data)


def save_hidden_context(path, context, *, prompt_tokens, saved_tokens):
    """Publish the compact checkpoint and return its SHA-256 for the caller’s manifest."""
    _validate_context(context, prompt_tokens, saved_tokens)
    path = Path(path)
    layers = context.detach()[0].view(-1, 4, 2304)
    digest = hashlib.sha256()
    stream = NamedTemporaryFile(dir=path.parent, delete=False)
    temporary = Path(stream.name)
    try:
        with stream:
            for layer in range(4):
                _write_rows(stream, layers[:prompt_tokens, layer], digest)
            if saved_tokens > prompt_tokens:
                _write_rows(stream, layers[prompt_tokens:saved_tokens], digest)
        os.replace(temporary, path)
        return digest.hexdigest()
    finally:
        temporary.unlink(missing_ok=True)


def restore_hidden_context(path, context, *, prompt_tokens, saved_tokens, expected_sha256=None):
    """Load layer-major prefill taps and token-major continuation taps."""
    _validate_context(context, prompt_tokens, saved_tokens)
    path = Path(path)
    elements = saved_tokens * context.shape[2]
    with path.open("rb") as stream:
        if os.fstat(stream.fileno()).st_size != elements * 2:
            raise ValueError("Drafter checkpoint size differs from saved length")
        # The descriptor binds validation and the GPU copy to one file even if
        # another writer atomically publishes a newer checkpoint at this path.
        with mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_COPY) as mapped:
            if expected_sha256 is not None and hashlib.sha256(mapped).hexdigest() != expected_sha256:
                raise ValueError("Drafter checkpoint digest mismatch")
            packed = torch.frombuffer(mapped, dtype=torch.bfloat16)
            try:
                destination = context[0].view(-1, 4, 2304)
                split = prompt_tokens * context.shape[2]
                destination[:prompt_tokens].copy_(packed[:split].reshape(4, prompt_tokens, 2304).transpose(0, 1))
                destination[prompt_tokens:saved_tokens].copy_(packed[split:].reshape(-1, 4, 2304))
            finally:
                del packed


def verify_greedy_block(backend, lease, context, proposal, *, remaining, stop_tokens=()):
    """Commit the target's greedy anchor and matching draft tokens within the request limit."""
    position = backend.position(lease) + 1
    _validate_context(context, position, position)
    if type(remaining) is not int or remaining < 1 or len(proposal) > 7:
        raise ValueError("Invalid DFlash2 block size or request limit")
    if any(type(token) is not int or not 0 <= token < backend.vocab_size for token in proposal):
        raise ValueError("Invalid DFlash2 proposal token")
    limit = min(remaining, context.shape[1] - position, backend.context_capacity - position)
    if limit < 1:
        raise ValueError("DFlash2 context is full")

    committed = []
    for index in range(min(limit, len(proposal) + 1)):
        logits = backend.logits(lease)
        token = max(range(len(logits)), key=logits.__getitem__)
        if index and proposal[index - 1] != token:
            break
        taps = backend.consume_dflash_taps(lease, [token])
        context[0, position + index].copy_(torch.frombuffer(taps, dtype=torch.float32))
        committed.append(token)
        if token in stop_tokens:
            break
    return committed
