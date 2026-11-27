"""Pinned Inkling 2 Mini token-ID backend with exact CUDA state transfers."""
from contextlib import contextmanager
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
from tempfile import TemporaryDirectory
from threading import RLock

from .digest_attestation import exported_payload_digest
from .checkpoint_cache import CheckpointCache
from .input_spans import InputHistory
from .real import NativeBackend, digest_file
from .store import IntegrityError, InvalidOperation

MODEL_SHA256 = '0bbfd52344f0dd4779a7807b5f73554a812f24c035e6877b63cd0d499f7e1b97'
TOKENIZER_SHA256 = '7f332727f9a3f093dd452a39cc978e6757627e5dad1b0673aa4e75174f236dd6'
VISION_SHA256 = '6589eff47f9c20bc3a032faa3d449f10bf4f04a84cc6b2ee8a44d51d2b6fc677'
GRAMMAR_TOKENIZER_SHA256 = '14132b1c19a0a44f3330db134fe6be44c68e22ddb6873cddd752a036e10b0f20'
PACK_HASHES = {
    'SQ-10': 'ac4d452ad03e4930abd9d46f9c5eec22b2f75a1da45a1e49bd24ccd58b065934',
    'SQ-15': '2ea32eed533afba257e2cbee7b073dc2c3da80fe55d1d2d794779b5498e30737',
    'SQ-20': MODEL_SHA256,
}
HEAD_WIDTH = 2304
HEAD_ADAPTER_HEADER = struct.Struct('<8sI32s')
HEAD_ADAPTER_MAGIC = b'IKLORA1\0'
VISION_EXECUTION = dict(input_projection='inkling-absolute-image-rows-v1',
                        embedding_width=2304, vision_weights_sha256=VISION_SHA256)
EXECUTION = dict(capacity=8192, prefill_chunk=512, kv='fp16', draft='off',
                 state_format='inkling-cuda-fp16-v1',
                 native_request_protocol='consumed-greedy-ack-v1',
                 generated_operator='normal-verifier-decode',
                 eos_tokens=[199999, 200006], GPU=[0])


def model_contract_id(vision=False, adapter_sha256=None, weights_sha256=MODEL_SHA256):
    identity = dict(architecture='inkling2_loop', weights_sha256=weights_sha256,
                    input_kind='mixed' if vision else 'token_ids',
                    adapters=[adapter_sha256] if adapter_sha256 else [],
                    vision=VISION_SHA256 if vision else False,
                    tokenizer_sha256=TOKENIZER_SHA256)
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def runtime_profile(runtime, *, vision=False, normalizer_id=None, adapter=None, capacity=8192, grammar=False, draft=False, gpus=1, pack="SQ-20"):
    if pack not in PACK_HASHES:
        raise ValueError('Unknown Inkling pack')
    weights = PACK_HASHES[pack]
    bridge = str(runtime / 'native-build/libinkling_pcc_bridge.so')
    build = {'sha256': {bridge: digest_file(bridge)}}
    if type(capacity) is not int or not 8192 <= capacity <= 133120:
        raise ValueError('Inkling capacity must be between 8192 and 133120 tokens')
    execution = dict(EXECUTION, capacity=capacity,
                     cublas_workspace_config=os.getenv('CUBLAS_WORKSPACE_CONFIG'),
                     cuda_cublas_compute_type=os.getenv('GGML_CUDA_CUBLAS_COMPUTE_TYPE', 'auto'),
                     nvidia_tf32_override=os.getenv('NVIDIA_TF32_OVERRIDE', 'unset'))
    if type(gpus) is not int or gpus not in (1, 2, 3):
        raise ValueError('Inkling requires one, two or three GPUs')
    # EP2 uses both devices inside one coordinator worker on logical GPU 0.
    if gpus == 2:
        execution['expert_parallel_devices'] = [0, 1]
    elif gpus == 3:
        execution['layer_split_devices'] = [0, 1, 2]
    if 'INKLING_EXPERT_VRAM_MIB' in os.environ:
        value = os.environ['INKLING_EXPERT_VRAM_MIB']
        if not value.isascii() or not value.isdecimal() or int(value) > 0xffffffff:
            raise ValueError('INKLING_EXPERT_VRAM_MIB must be an unsigned 32-bit MiB count')
        execution['expert_vram_mib'] = value
    if 'STRATA_INK_EXACT' in os.environ:
        execution['cold_expert_lane'] = os.environ['STRATA_INK_EXACT']
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        execution['cuda_visible_devices'] = os.environ['CUDA_VISIBLE_DEVICES']
    hardware = subprocess.check_output([
        'nvidia-smi', '--query-gpu=name,uuid,driver_version', '--format=csv,noheader'
    ], text=True).strip()
    if os.environ.get('INKLING_EXACT_HEAD') == '1':
        execution['exact_single_row_head'] = True
    extra = {}
    if draft:
        if vision or adapter or grammar:
            raise ValueError("DSpark currently supports unadapted text generation")
        execution.update(draft="dspark", state_format="inkling-cuda-fp16-dspark-v1")
    if grammar:
        execution.update(selected_token_protocol="consumed-selected-receipt-v1",
                         grammar_tokenizer_sha256=GRAMMAR_TOKENIZER_SHA256)
    if vision:
        execution.update(VISION_EXECUTION)
        if normalizer_id is not None:
            execution['vision_normalizer_id'] = normalizer_id
        extra['vision_path'] = str(runtime / 'Inkling-2-Mini-Vision.strata')
    if adapter is not None:
        adapter = adapter.resolve(strict=True)
        extra['adapter_path'] = str(adapter)
        extra['adapter_sha256'] = digest_file(adapter)
    identity = dict(build=build, execution=execution, hardware=hardware)
    return dict(backend='inkling-cuda', bridge_library=bridge, native_build=build,
                model_path=str(runtime / f'Inkling-2-Mini-{pack}.strata'),
                tokenizer_path=str(runtime / 'tokenizer-source.json'),
                model_weights_sha256=weights, model_contract_id=model_contract_id(vision, extra.get('adapter_sha256'), weights),
                execution=execution, hardware=hardware,
                execution_profile_id=hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest(),
                **extra)


class InklingBackend(NativeBackend):
    exports_portable_bytes = True
    context_capacity = EXECUTION['capacity']

    def __init__(self, profile):
        self.context_capacity = profile['execution']['capacity']
        self.draft = profile['execution']['draft'] == 'dspark'
        self._lock = RLock()
        self._leased = False
        self._pinned_digest = None
        self._exported_digest = None
        self._close_pending = False
        self.vision_path = profile.get('vision_path')
        self.adapter_sha256 = profile.get('adapter_sha256')
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
        if 'expert_vram_mib' in self.profile['execution']:
            supported = getattr(self.lib, 'ctt_supports_expert_vram_budget', None)
            if supported is None or supported() != 1:
                raise IntegrityError('Rebuild the Inkling bridge to apply the expert VRAM budget')
        self.lib.ctt_model_work.argtypes = [C.c_void_p]
        self.lib.ctt_model_work.restype = C.c_uint64
        self._drafts_accepted = getattr(self.lib, "ctt_drafts_accepted", None)
        if self._drafts_accepted is not None:
            self._drafts_accepted.argtypes = [C.c_void_p]
            self._drafts_accepted.restype = C.c_uint64
        self.lib.ctt_import_checkpoint.argtypes = [
            C.c_void_p, C.c_void_p, C.c_size_t, C.c_int,
            C.POINTER(C.c_float), C.c_size_t,
        ]
        self.lib.ctt_import_checkpoint.restype = C.c_size_t
        self.lib.ctt_select_greedy.argtypes = [C.c_void_p]
        self.lib.ctt_select_greedy.restype = C.c_int
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
        self.lib.ctt_head_features.argtypes = [
            C.c_void_p, C.POINTER(C.c_int32), C.c_int,
            C.POINTER(C.c_float), C.POINTER(C.c_float),
        ]
        self.lib.ctt_head_features.restype = C.c_int
        self.lib.ctt_logit_softcap.argtypes = [C.c_void_p]
        self.lib.ctt_logit_softcap.restype = C.c_float
        self.lib.ctt_attach_head_lora.argtypes = [C.c_void_p, C.c_int, C.c_void_p, C.c_void_p]
        self.lib.ctt_attach_head_lora.restype = C.c_int
        execution = self.profile['execution']
        gpu_count = len(execution.get('layer_split_devices', execution.get('expert_parallel_devices', [0])))
        if gpu_count > 1 and self.draft:
            raise ValueError('Multi-GPU DSpark requires separate qualification')
        if self.context_capacity == 8192 and not self.draft and gpu_count == 1:
            model = super()._load_model()
        else:
            self.lib.ctt_pin_shared.argtypes = [C.c_void_p]
            self.lib.ctt_pin_shared.restype = C.c_int
            self.lib.ctt_reuse_shared.argtypes = [C.c_void_p, C.c_int, C.POINTER(C.c_float), C.c_size_t]
            self.lib.ctt_reuse_shared.restype = C.c_int
            self.lib.ctt_reuse_exported.argtypes = self.lib.ctt_reuse_shared.argtypes
            self.lib.ctt_reuse_exported.restype = C.c_int
            load = self.lib.ctt_load_capacity
            load.argtypes = [C.c_char_p, C.c_int]
            load.restype = C.c_void_p
            if self.draft:
                load = self.lib.ctt_load_profile
                load.argtypes = [C.c_char_p, C.c_int, C.c_bool]
                load.restype = C.c_void_p
                self.lib.ctt_draft_block.argtypes = [C.c_void_p, C.c_int, C.POINTER(C.c_int32), C.POINTER(C.c_int)]
                self.lib.ctt_draft_block.restype = C.c_int
                model = load(os.fsencode(self.profile['model_path']), self.context_capacity, True)
            elif gpu_count > 1:
                load = self.lib.ctt_load_devices
                load.argtypes = [C.c_char_p, C.c_int, C.c_bool, C.c_int]
                load.restype = C.c_void_p
                model = load(os.fsencode(self.profile['model_path']), self.context_capacity, False, gpu_count)
            else:
                model = load(os.fsencode(self.profile['model_path']), self.context_capacity)
        if model and self.adapter_sha256:
            try:
                self._attach_adapter(model)
            except BaseException:
                self.lib.ctt_unload(model)
                raise
        return model

    def _attach_adapter(self, model):
        vocab = self.lib.ctt_vocab(model)
        maximum = HEAD_ADAPTER_HEADER.size + 16 * (HEAD_WIDTH + vocab) * 4
        with open(self.profile['adapter_path'], 'rb') as stream:
            data = stream.read(maximum + 1)
        if len(data) < HEAD_ADAPTER_HEADER.size or hashlib.sha256(data).hexdigest() != self.adapter_sha256:
            raise IntegrityError('Inkling adapter fingerprint mismatch')
        magic, rank, base = HEAD_ADAPTER_HEADER.unpack_from(data)
        split = HEAD_ADAPTER_HEADER.size + rank * HEAD_WIDTH * 4
        if (magic != HEAD_ADAPTER_MAGIC or base.hex() != self.profile['model_weights_sha256'] or not 1 <= rank <= 16
                or len(data) != split + rank * vocab * 4):
            raise IntegrityError('Invalid Inkling head LoRA geometry or base identity')
        if self.lib.ctt_attach_head_lora(model, rank, C.c_char_p(data[HEAD_ADAPTER_HEADER.size:split]),
                                         C.c_char_p(data[split:])):
            raise IntegrityError('Inkling native LoRA attach failed')

    def validate_grammar_engine(self, engine):
        from .grammar_first import GrammarEngine
        if not isinstance(engine, GrammarEngine) or engine.tokenizer_sha256 != GRAMMAR_TOKENIZER_SHA256:
            raise IntegrityError('Inkling grammar tokenizer fingerprint mismatch')

    def consume_generated(self, lease, ids):
        if len(ids) != 1:
            raise InvalidOperation('Inkling decode requires one selected token')
        self.consume(lease, ids)

    def head_features(self, lease, tokens):
        if self.adapter_sha256:
            raise InvalidOperation('Head training requires the unadapted Inkling profile')
        if not tokens or len(tokens) > self.context_capacity - self.position(lease) - 1:
            raise InvalidOperation('Invalid Inkling head training context')
        ids = (C.c_int32 * len(tokens))(*tokens)
        features = (C.c_float * HEAD_WIDTH)()
        logits = (C.c_float * self.vocab_size)()
        if self.lib.ctt_head_features(lease, ids, len(ids), features, logits):
            raise RuntimeError(self.lib.ctt_error().decode())
        return features, logits

    @property
    def logit_softcap(self):
        return self.lib.ctt_logit_softcap(self.model)

    def validate_execution_profile(self):
        profile = self.profile
        execution = profile['execution']
        if 'cublas_workspace_config' not in execution:
            raise IntegrityError('Legacy Inkling profile has no recorded cuBLAS workspace setting')
        if execution['cublas_workspace_config'] != os.getenv('CUBLAS_WORKSPACE_CONFIG'):
            raise IntegrityError('cuBLAS workspace setting differs from the saved profile')
        if execution.get('expert_vram_mib') != os.environ.get('INKLING_EXPERT_VRAM_MIB'):
            raise IntegrityError('Expert VRAM budget differs from the saved profile')
        if execution.get('cold_expert_lane') != os.environ.get('STRATA_INK_EXACT'):
            raise IntegrityError('Cold expert execution mode differs from the saved profile')
        if self.vision_path:
            if (any(execution.get(key) != value for key, value in VISION_EXECUTION.items())
                    or digest_file(self.vision_path) != VISION_SHA256):
                raise IntegrityError('Inkling vision profile or weight fingerprint mismatch')
        elif any(key in execution for key in VISION_EXECUTION):
            raise IntegrityError('Inkling vision profile requires the pinned vision weights')
        if bool(profile.get('adapter_path')) != bool(self.adapter_sha256):
            raise IntegrityError('Inkling adapter path and identity must be provided together')
        if 'layer_split_devices' in execution:
            split = execution['layer_split_devices']
            if (split != [0, 1, 2] or any(type(i) is not int for i in split)
                    or 'expert_parallel_devices' in execution):
                raise IntegrityError('Invalid Inkling layer split')
        devices = execution.get('expert_parallel_devices', [0])
        if devices not in ([0], [0, 1]) or any(type(i) is not int for i in devices):
            raise IntegrityError('Invalid Inkling GPU placement')
        required = dict(EXECUTION)
        if self.draft:
            required.update(draft='dspark', state_format='inkling-cuda-fp16-dspark-v1')
            if self.adapter_sha256 or self.vision_path:
                raise IntegrityError('Unsupported DSpark model configuration')
        weights = profile['model_weights_sha256']
        if (weights not in PACK_HASHES.values()
                or profile['model_contract_id'] != model_contract_id(bool(self.vision_path), self.adapter_sha256, weights)
                or type(self.context_capacity) is not int or not 8192 <= self.context_capacity <= 133120
                or any(execution.get(key) != value for key, value in required.items() if key != 'capacity')
                or profile['bridge_library'] not in profile['native_build']['sha256']
                or not Path(profile['bridge_library']).is_absolute()):
            raise IntegrityError('Unsupported Inkling model or execution profile')
        super().validate_execution_profile()

    def new_cache(self):
        default_limit = max(2 * 1024**3, 3 * (self.context_capacity * 32768 + 180 * 1024**2))
        return CheckpointCache(byte_limit=self.profile.get('ram_cache_bytes', default_limit), entry_limit=32)

    def contract_fields(self):
        return dict(attention='full-and-sliding', state_kind='looped-kv-shortconv',
                    tokenizer_identity=TOKENIZER_SHA256,
                    input_kind='mixed' if self.vision_path else 'token_ids',
                    adapters=(self.adapter_sha256,) if self.adapter_sha256 else ())

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

    @contextmanager
    def generate_request(self, lease, max_steps):
        def generate():
            if self.draft:
                block = (C.c_int32 * 9)()
                nodes = C.c_int()
                remaining = max_steps
                while remaining:
                    count = self.lib.ctt_draft_block(lease, remaining, block, C.byref(nodes))
                    if not 1 <= count <= min(9, remaining):
                        raise IntegrityError(self.lib.ctt_error().decode() or 'Invalid DSpark block')
                    self.stats['draft_nodes'] = self.stats.get('draft_nodes', 0) + nodes.value
                    remaining -= count
                    yield from block[:count]
                    if self.is_eog(block[count - 1]):
                        break
                return
            for _ in range(max_steps):
                token = self.lib.ctt_select_greedy(lease)
                if token < 0:
                    raise IntegrityError(self.lib.ctt_error().decode())
                self.consume(lease, (token,))
                yield token
                if self.is_eog(token):
                    break

        request = generate()
        try:
            yield request
        finally:
            request.close()

    def work_counters(self):
        with self._lock:
            if not self.model:
                raise RuntimeError('Inkling backend is closed')
            tokens = self.lib.ctt_model_work(self.model)
            counters = dict(consumed_tokens=tokens, forward_tokens=tokens)
            if self.context_capacity > 8192:
                counters.update(resident_parent_restores=self.stats.get('resident_parent_restores', 0),
                                resident_child_restores=self.stats.get('resident_child_restores', 0),
                                checkpoint_import_bytes=self.stats['checkpoint_import_bytes'])
            if self.draft:
                if self._drafts_accepted is not None:
                    counters["drafts_accepted"] = self._drafts_accepted(self.model)
                counters["draft_nodes"] = self.stats.get("draft_nodes", 0)
            return counters

    def export(self, lease):
        blob = super().export(lease)
        if self.context_capacity > 8192:
            self._exported_digest = exported_payload_digest(blob)
            if self.lib.ctt_pin_shared(lease):
                self._pinned_digest = self._exported_digest
        return blob

    def restore(self, lease, blob, *, history=None, frontier=None):
        if not blob or not history or frontier is None or len(frontier) != self.vocab_size:
            raise IntegrityError('Incomplete Inkling checkpoint')
        row = (C.c_float * self.vocab_size)(*frontier)
        if self.context_capacity > 8192:
            digest = hashlib.sha256(blob).digest()
            if digest == self._pinned_digest and self.lib.ctt_reuse_shared(lease, len(history), row, len(row)):
                self.stats['resident_parent_restores'] = self.stats.get('resident_parent_restores', 0) + 1
                return blob
            if digest == self._exported_digest and self.lib.ctt_reuse_exported(lease, len(history), row, len(row)):
                self.stats['resident_child_restores'] = self.stats.get('resident_child_restores', 0) + 1
                return blob
            self._pinned_digest = None
        if self.lib.ctt_import_checkpoint(lease, C.c_char_p(blob), len(blob),
                                          len(history), row, len(row)) != len(blob):
            self.close()
            raise IntegrityError('Inkling state transfer failed')
        if self.context_capacity > 8192 and self.lib.ctt_pin_shared(lease):
            self._pinned_digest = digest
        self.stats['checkpoint_import_bytes'] += len(blob)
        return blob

    def close(self):
        with self._lock:
            if self._leased:
                self._close_pending = True
            else:
                super().close()
