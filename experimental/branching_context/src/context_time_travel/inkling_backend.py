"""Pinned Inkling 2 Mini token-ID backend with exact CUDA state transfers."""
from contextlib import contextmanager
import ctypes as C
import hashlib
import json
from pathlib import Path
from threading import RLock

from .checkpoint_cache import CheckpointCache
from .real import NativeBackend, digest_file
from .store import IntegrityError

MODEL_SHA256 = '0bbfd52344f0dd4779a7807b5f73554a812f24c035e6877b63cd0d499f7e1b97'
TOKENIZER_SHA256 = '7f332727f9a3f093dd452a39cc978e6757627e5dad1b0673aa4e75174f236dd6'
EXECUTION = dict(capacity=8192, prefill_chunk=512, kv='fp16', draft='off',
                 state_format='inkling-cuda-fp16-v1')


def model_contract_id():
    identity = dict(architecture='inkling2_loop', weights_sha256=MODEL_SHA256,
                    input_kind='token_ids', adapters=[], vision=False,
                    tokenizer_sha256=TOKENIZER_SHA256)
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


class InklingBackend(NativeBackend):
    context_capacity = EXECUTION['capacity']

    def __init__(self, profile):
        self._lock = RLock()
        self._leased = False
        self._close_pending = False
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
        return super()._load_model()

    def validate_execution_profile(self):
        profile = self.profile
        if (profile['model_weights_sha256'] != MODEL_SHA256
                or profile['model_contract_id'] != model_contract_id()
                or any(profile['execution'].get(key) != value for key, value in EXECUTION.items())
                or profile['bridge_library'] not in profile['native_build']['sha256']
                or not Path(profile['bridge_library']).is_absolute()):
            raise IntegrityError('Unsupported Inkling model or execution profile')
        super().validate_execution_profile()

    def new_cache(self):
        return CheckpointCache(byte_limit=2 * 1024**3, entry_limit=32)

    def contract_fields(self):
        return dict(attention='full-and-sliding', state_kind='looped-kv-shortconv',
                    tokenizer_identity=TOKENIZER_SHA256)

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
