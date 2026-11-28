"""Bounded in-process attestations for exact immutable checkpoint owners.

This is only a host-side digest memo for RAM objects. It does not certify disk
contents or native/GPU state, and it does not replace canonical binding checks.
"""
from collections import OrderedDict
from contextlib import contextmanager
import hashlib
import math
import struct
import threading
import weakref


MAX_ATTESTED_OWNERS = 32
_LOCK = threading.RLock()
_OWNERS = OrderedDict()


_export = threading.local()
_restore_verification = 'fast'


def set_restore_verification(mode):
    """Select the server-wide digest policy before starting worker threads."""
    if mode not in ('fast', 'full'):
        raise ValueError('Expected fast or full restore verification')
    global _restore_verification
    _restore_verification = mode


@contextmanager
def checkpoint_export_scope():
    previous = getattr(_export, 'active', False)
    previous_payload = getattr(_export, 'payload', None)
    _export.active = True
    _export.payload = None
    try:
        yield
    finally:
        _export.active = previous
        _export.payload = previous_payload


def exported_payload_digest(blob):
    digest = hashlib.sha256(blob)
    if type(blob) is bytes and getattr(_export, 'active', False):
        _export.payload = (blob, digest)
    return digest.digest()


def _compute_digest(blob, frontier, *, reuse_export=True):
    state_identity = blob if isinstance(blob, bytes) else blob.identity
    payload = getattr(_export, 'payload', None)
    if reuse_export and payload is not None and blob is payload[0]:
        digest = payload[1].copy()
    else:
        digest = hashlib.sha256(state_identity)
    digest.update(struct.pack('<' + str(len(frontier)) + 'f', *frontier))
    return digest.hexdigest()


def _eligible(owner):
    blob = getattr(owner, 'blob', None)
    frontier = getattr(owner, 'frontier', None)
    try:
        return (type(blob) is bytes and bool(blob) and type(frontier) is tuple and bool(frontier)
                and all(type(value) in (int, float) and math.isfinite(value) for value in frontier)
                and type(getattr(owner, 'checksum', None)) is str)
    except (OverflowError, TypeError, ValueError):
        return False


def _drop(key, owner_ref):
    with _LOCK:
        entry = _OWNERS.get(key)
        if entry is not None and entry[0] is owner_ref:
            del _OWNERS[key]


def _remember(owner, digest):
    if not _eligible(owner) or owner.checksum != digest:
        return
    key = id(owner)
    try:
        owner_ref = weakref.ref(owner, lambda ref, key=key: _drop(key, ref))
    except TypeError:
        # Weak-reference support is an optimization constraint, not part of
        # the existing checkpoint admission contract. Keep hashing normally.
        return
    payload = getattr(_export, "payload", None)
    raw_digest = payload[1].digest() if payload is not None and payload[0] is owner.blob else None
    entry = (owner_ref, owner.blob, owner.frontier, owner.checksum, digest, raw_digest)
    with _LOCK:
        _OWNERS[key] = entry
        _OWNERS.move_to_end(key)
        while len(_OWNERS) > MAX_ATTESTED_OWNERS:
            _OWNERS.popitem(last=False)


def restore_payload_digest(blob):
    """Reuse a scoped export hash only for the very same immutable bytes."""
    if _restore_verification == 'fast' and type(blob) is bytes:
        with _LOCK:
            for owner_ref, payload, _, _, _, raw_digest in reversed(_OWNERS.values()):
                if payload is blob and raw_digest is not None and owner_ref() is not None:
                    return raw_digest
    return hashlib.sha256(blob).digest()


def attested_owner_for_digest(checksum):
    """Find a live immutable RAM reference for bytewise on-disk validation."""
    if _restore_verification == 'full':
        return None
    with _LOCK:
        for owner_ref, blob, frontier, expected, digest, _ in reversed(_OWNERS.values()):
            if checksum == expected == digest:
                owner = owner_ref()
                if owner is not None and owner.blob is blob and owner.frontier is frontier:
                    return owner
    return None


def verified_checkpoint_digest(owner):
    """Return the actual RAM digest, reusing only the exact certified owner."""
    if _restore_verification == 'full':
        return _compute_digest(owner.blob, owner.frontier, reuse_export=False)
    key = id(owner)
    with _LOCK:
        entry = _OWNERS.get(key)
        if entry is not None:
            owner_ref, blob, frontier, checksum, digest, _ = entry
            # The retained tuple was eligible when certified: its elements
            # were exact immutable builtin numbers, and the strong reference
            # below guarantees this exact tuple is still present. Rechecking
            # every element here would scan large logit frontiers per lookup.
            if (owner_ref() is owner and owner.blob is blob and owner.frontier is frontier
                    and type(owner.checksum) is str and owner.checksum == checksum == digest):
                _OWNERS.move_to_end(key)
                return digest
            del _OWNERS[key]
    digest = _compute_digest(owner.blob, owner.frontier)
    if digest == getattr(owner, 'checksum', None):
        _remember(owner, digest)
    return digest


def _attest_factory_result(owner):
    """Hash an exact factory result, set its checksum, then attest that owner."""
    digest = _compute_digest(owner.blob, owner.frontier)
    object.__setattr__(owner, 'checksum', digest)
    _remember(owner, digest)
    return digest


def attested_owner_count():
    """Bounded diagnostics for tests; not a persistence or health metric."""
    with _LOCK:
        return len(_OWNERS)
