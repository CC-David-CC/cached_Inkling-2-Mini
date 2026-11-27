"""Pinned Inkling 2 Mini token-ID backend with exact CUDA state transfers."""
from contextlib import contextmanager
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import struct
from tempfile import TemporaryDirectory
from threading import RLock

from .checkpoint_cache import CheckpointCache
from .input_spans import InputHistory
from .real import NativeBackend, digest_file
from .store import IntegrityError, InvalidOperation

MODEL_SHA256 = '0bbfd52344f0dd4779a7807b5f73554a812f24c035e6877b63cd0d499f7e1b97'
TOKENIZER_SHA256 = '7f332727f9a3f093dd452a39cc978e6757627e5dad1b0673aa4e75174f236dd6'
VISION_SHA256 = '6589eff47f9c20bc3a032faa3d449f10bf4f04a84cc6b2ee8a44d51d2b6fc677'
VISION_EXECUTION = dict(input_projection='inkling-absolute-image-rows-v1',
                        embedding_width=2304, vision_weights_sha256=VISION_SHA256)
EXECUTION = dict(capacity=8192, prefill_chunk=512, kv='fp16', draft='off',
                 state_format='inkling-cuda-fp16-v1')


def model_contract_id(vision=False):
    identity = dict(architecture='inkling2_loop', weights_sha256=MODEL_SHA256,
                    input_kind='mixed' if vision else 'token_ids', adapters=[],
                    vision=VISION_SHA256 if vision else False,
                    tokenizer_sha256=TOKENIZER_SHA256)
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


class InklingBackend(NativeBackend):
    context_capacity = EXECUTION['capacity']

    def __init__(self, profile):
        self._lock = RLock()
        self._leased = False
        self._close_pending = False
        self.vision_path = profile.get('vision_path')
        tokenizer_path = Path(profile['tokenizer_path'])
        if digest_file(tokenizer_path) != TOKENIZER_SHA256:
            raise IntegrityError('Inkling tokenizer fingerprint mismatch')
        tokenizer = json.loads(tokenizer_path.read_bytes())
        # ByteLevel BPE uses printable Unicode stand-ins for the 256 raw bytes.
        printable = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
        missing = [byte for byte in range(256) if byte not in printable]
        decoder = {chr(byte): byte for byte in printable}
        decoder.update({chr(256 + index): byte for index, byte in enumerate(missing)})
        self._pieces = {
            index: bytes(decoder[char] for char in piece)
            for piece, index in tokenizer['model']['vocab'].items()
        }
        self._pieces.update({entry['id']: entry['content'].encode('utf-8')
                             for entry in tokenizer['added_tokens']})
        super().__init__(profile)

    def piece(self, token):
        try:
            return self._pieces[token]
        except KeyError as error:
            raise IntegrityError('Unknown Inkling token ID') from error

    def _load_model(self):
        self.lib.ctt_model_work.argtypes = [C.c_void_p]
        self.lib.ctt_model_work.restype = C.c_uint64
        self.lib.ctt_import_checkpoint.argtypes = [
            C.c_void_p, C.c_void_p, C.c_size_t, C.c_int,
            C.POINTER(C.c_float), C.c_size_t,
        ]
        self.lib.ctt_import_checkpoint.restype = C.c_size_t
        self.lib.ctt_image_pad.argtypes = [C.c_void_p]
        self.lib.ctt_image_pad.restype = C.c_int
        self.lib.ctt_encode_image.argtypes = [C.c_void_p, C.c_char_p, C.c_char_p]
        self.lib.ctt_encode_image.restype = C.c_void_p
        self.lib.ctt_image_data.argtypes = [C.c_void_p, C.POINTER(C.c_int),
                                            C.POINTER(C.c_int), C.POINTER(C.c_size_t)]
        self.lib.ctt_image_data.restype = C.c_void_p
        self.lib.ctt_free_image.argtypes = [C.c_void_p]
        self.lib.ctt_free_image.restype = None
        self.lib.ctt_consume_mixed.argtypes = [
            C.c_void_p, C.POINTER(C.c_int32), C.POINTER(C.c_int32),
            C.c_void_p, C.c_int, C.c_int,
        ]
        self.lib.ctt_consume_mixed.restype = C.c_int
        return super()._load_model()

    def validate_execution_profile(self):
        profile = self.profile
        execution = profile['execution']
        if self.vision_path:
            if (any(execution.get(key) != value for key, value in VISION_EXECUTION.items())
                    or digest_file(self.vision_path) != VISION_SHA256):
                raise IntegrityError('Inkling vision profile or weight fingerprint mismatch')
        elif any(key in execution for key in VISION_EXECUTION):
            raise IntegrityError('Inkling vision profile requires the pinned vision weights')
        if (profile['model_weights_sha256'] != MODEL_SHA256
                or profile['model_contract_id'] != model_contract_id(bool(self.vision_path))
                or any(profile['execution'].get(key) != value for key, value in EXECUTION.items())
                or profile['bridge_library'] not in profile['native_build']['sha256']
                or not Path(profile['bridge_library']).is_absolute()):
            raise IntegrityError('Unsupported Inkling model or execution profile')
        super().validate_execution_profile()

    def new_cache(self):
        return CheckpointCache(byte_limit=2 * 1024**3, entry_limit=32)

    def contract_fields(self):
        return dict(attention='full-and-sliding', state_kind='looped-kv-shortconv',
                    tokenizer_identity=TOKENIZER_SHA256,
                    input_kind='mixed' if self.vision_path else 'token_ids')

    @property
    def image_pad(self):
        return self.lib.ctt_image_pad(self.model)

    @property
    def encoder_profile_id(self):
        return self.profile['execution_profile_id']

    def encode_image(self, data):
        if not self.vision_path or not 0 < len(data) <= 16 * 1024**2:
            raise InvalidOperation('Inkling vision requires an enabled tower and bounded image bytes')
        with self._lock, TemporaryDirectory(prefix='inkling-image-') as directory:
            path = Path(directory) / 'input'
            path.write_bytes(data)
            image = self.lib.ctt_encode_image(self.model, os.fsencode(self.vision_path), os.fsencode(path))
            if not image:
                raise InvalidOperation(self.lib.ctt_error().decode())
            try:
                nx, ny, size = C.c_int(), C.c_int(), C.c_size_t()
                payload = self.lib.ctt_image_data(image, C.byref(nx), C.byref(ny), C.byref(size))
                width = VISION_EXECUTION['embedding_width']
                header = struct.pack('<5i', 0x31455653, nx.value * ny.value, nx.value, ny.value, width)
                return header + C.string_at(payload, size.value)
            finally:
                self.lib.ctt_free_image(image)

    def consume_inputs(self, lease, spans, *, embeddings):
        history = InputHistory(spans)
        if not self.vision_path or not 0 < len(history) <= self.context_capacity - self.position(lease) - 1:
            raise InvalidOperation('Invalid Inkling image input capacity or profile')
        indices = []
        chunks = []
        row = 0
        for span in history.spans:
            if span['kind'] == 'text':
                indices.extend([-1] * len(span['tokens']))
                continue
            digest = span['embedding_sha256']
            embedding = embeddings.get(digest)
            if (span['encoder_profile_id'] != self.encoder_profile_id
                    or span['pad_token_id'] != self.image_pad
                    or embedding is None or embedding.sha256 != digest
                    or embedding.rows != span['rows']
                    or embedding.width != VISION_EXECUTION['embedding_width']):
                raise IntegrityError('Inkling image embedding identity or dimensions mismatch')
            chunks.append(embedding.data)
            indices.extend(range(row, row + embedding.rows))
            row += embedding.rows
        ids = (C.c_int32 * len(history))(*history.native_tokens)
        placements = (C.c_int32 * len(history))(*indices)
        data = b''.join(chunks)
        before = self.lib.ctt_work(lease)
        if self.lib.ctt_consume_mixed(lease, ids, placements, C.c_char_p(data), row, len(history)):
            raise RuntimeError(self.lib.ctt_error().decode())
        delta = self.lib.ctt_work(lease) - before
        if delta != len(history):
            raise IntegrityError('Inkling mixed native work counter mismatch')
        self.stats['consumed_tokens'] += delta

    @contextmanager
    def lease(self):
        with self._lock:
            if not self.model:
                raise RuntimeError('Inkling backend is closed')
            if self._leased:
                raise RuntimeError('Inkling backend already has an active lease')
            self._leased = True
            try:
                with super().lease() as lease:
                    yield lease
            finally:
                self._leased = False
                if self._close_pending:
                    super().close()

    def work_counters(self):
        with self._lock:
            if not self.model:
                raise RuntimeError('Inkling backend is closed')
            return dict(consumed_tokens=self.lib.ctt_model_work(self.model))

    def restore(self, lease, blob, *, history=None, frontier=None):
        if not blob or not history or frontier is None or len(frontier) != self.vocab_size:
            raise IntegrityError('Incomplete Inkling checkpoint')
        row = (C.c_float * self.vocab_size)(*frontier)
        if self.lib.ctt_import_checkpoint(lease, C.c_char_p(blob), len(blob),
                                          len(history), row, len(row)) != len(blob):
            self.close()
            raise IntegrityError('Inkling state transfer failed')
        self.stats['checkpoint_import_bytes'] += len(blob)
        return blob

    def close(self):
        with self._lock:
            if self._leased:
                self._close_pending = True
            else:
                super().close()
