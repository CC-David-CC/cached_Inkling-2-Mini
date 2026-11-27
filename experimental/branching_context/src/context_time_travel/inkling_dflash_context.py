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


def write_hidden_context(stream, context, *, prompt_tokens, saved_tokens):
    """Append checkpoint rows to a binary stream and return their SHA-256."""
    _validate_context(context, prompt_tokens, saved_tokens)
    layers = context.detach()[0].view(-1, 4, 2304)
    digest = hashlib.sha256()
    for layer in range(4):
        _write_rows(stream, layers[:prompt_tokens, layer], digest)
    if saved_tokens > prompt_tokens:
        _write_rows(stream, layers[prompt_tokens:saved_tokens], digest)
    return digest.hexdigest()


def save_hidden_context(path, context, *, prompt_tokens, saved_tokens):
    """Publish the compact checkpoint and return its SHA-256 for the caller’s manifest."""
    path = Path(path)
    stream = NamedTemporaryFile(dir=path.parent, delete=False)
    temporary = Path(stream.name)
    try:
        with stream:
            digest = write_hidden_context(stream, context, prompt_tokens=prompt_tokens, saved_tokens=saved_tokens)
        os.replace(temporary, path)
        return digest
    finally:
        temporary.unlink(missing_ok=True)


def restore_hidden_buffer(buffer, context, *, prompt_tokens, saved_tokens, expected_sha256=None):
    """Restore from a writable contiguous byte buffer, including a slice of a RAM checkpoint."""
    _validate_context(context, prompt_tokens, saved_tokens)
    with memoryview(buffer) as view:
        if view.nbytes != saved_tokens * context.shape[2] * 2:
            raise ValueError("Drafter checkpoint size differs from saved length")
        if view.readonly or not view.c_contiguous:
            raise ValueError("Expected writable contiguous drafter checkpoint storage")
        if expected_sha256 is not None and hashlib.sha256(view).hexdigest() != expected_sha256:
            raise ValueError("Drafter checkpoint digest mismatch")
        packed = torch.frombuffer(view, dtype=torch.bfloat16)
        destination = context[0].view(-1, 4, 2304)
        split = prompt_tokens * context.shape[2]
        destination[:prompt_tokens].copy_(packed[:split].reshape(4, prompt_tokens, 2304).transpose(0, 1))
        destination[prompt_tokens:saved_tokens].copy_(packed[split:].reshape(-1, 4, 2304))


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
            restore_hidden_buffer(mapped, context, prompt_tokens=prompt_tokens, saved_tokens=saved_tokens,
                                  expected_sha256=expected_sha256)


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
        token = backend.select_greedy(lease)
        if index and proposal[index - 1] != token:
            break
        taps = backend.consume_dflash_taps(lease, [token])
        context[0, position + index].copy_(torch.frombuffer(taps, dtype=torch.float32))
        committed.append(token)
        if token in stop_tokens or backend.is_eog(token):
            break
    return committed


@torch.inference_mode()
def select_draft_path(drafter, hidden, head, anchor):
    """Select seven Inkling DFlash2 successors from the 16 candidates at each position."""
    unary, candidates = (hidden @ head.T).float().topk(16, dim=-1)
    selector = drafter.candidate_selector
    projected = selector.hidden_projection(hidden)
    predecessor = anchor
    path = []
    for position in range(7):
        choices = candidates[:, position]
        edge = (selector.predecessor_codebook(predecessor)[:, None, :]
                * projected[:, position:position + 1, :]
                * selector.successor_codebook(choices)).float().sum(dim=-1)
        scores = 30 * torch.tanh((unary[:, position] + edge) / 30)
        predecessor = choices.gather(-1, scores.argmax(-1, keepdim=True))[:, 0]
        path.append(predecessor)
    return torch.stack(path, dim=1)


@torch.inference_mode()
def propose_draft_block(drafter, context, anchor, embedding, embedding_norm, head):
    """Propose seven successors with resident or host-staged target weights."""
    device = context.device
    positions = torch.arange(context.shape[1] + 8, device=device).unsqueeze(0)
    ids = torch.full((1, 8), 200056, dtype=torch.long, device=embedding.device)
    ids[0, 0] = anchor
    noise = torch.nn.functional.rms_norm(
        embedding[ids].to(device, torch.float32), (2304,), embedding_norm.to(device, torch.float32), 1e-6
    ).to(torch.bfloat16)
    hidden = drafter(position_ids=positions, noise_embedding=noise, target_hidden=context)
    # Stage the output head after attention releases its temporary buffers.
    return select_draft_path(drafter, hidden[:, 1:], head.to(device), ids[:, 0].to(device))[0].tolist()
