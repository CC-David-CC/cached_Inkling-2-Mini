"""Adapt PCC image admission and rendering to Inkling's pinned native vision tower."""
from concurrent.futures import CancelledError
import hashlib
from types import SimpleNamespace

from context_time_travel.inkling_backend import VISION_EXECUTION
from context_time_travel.store import InvalidOperation
from context_time_travel.vision_embeddings import parse_sve

from .cow_vision import CowVisionPromptEncoder
from .cow_vision_preparer import CowVisionPreparer


class InklingVisionPromptEncoder(CowVisionPromptEncoder):
    markers = ('<|content_image|>', '<|unused_200054|>', '<|end_message|>')


class InklingVisionPreparer(CowVisionPreparer):
    def __init__(self, store, normalizer, renderer, profile):
        identity = profile['execution_profile_id']
        descriptor = SimpleNamespace(identity=identity, width=VISION_EXECUTION['embedding_width'],
                                     max_rows=1024)
        super().__init__(store, normalizer, renderer, descriptor,
                         original_bytes_limit=256 * 1024**2)
        self.encoder_profile_id = identity

    def validate(self, request):
        if isinstance(request, dict) and request.get('schema') == 2:
            raise InvalidOperation('Inkling image upload currently requires a new prepared prompt')
        return super().validate(request)

    def supports_worker(self, model):
        return (model.profile.get('backend') == 'inkling-cuda'
                and model.profile['execution_profile_id'] == self.encoder_profile_id
                and model.profile['execution'].get('vision_normalizer_id') == self.normalizer.identity
                and type(getattr(model, 'gpu_id', None)) is int)

    def encode_on_worker(self, model, raw, *, pool, disk, cancelled):
        if cancelled is not None and cancelled():
            raise CancelledError('Inkling vision encoding cancelled')
        with pool.reserve(20 + 4 * self.encoder.width * self.encoder.max_rows):
            payload = model.backend.encode_image(raw)
            embedding = parse_sve(payload, expected_sha256=hashlib.sha256(payload).hexdigest(),
                                  expected_width=self.encoder.width, max_rows=self.encoder.max_rows)
        if cancelled is not None and cancelled():
            raise CancelledError('Inkling vision encoding cancelled')
        return SimpleNamespace(embedding=embedding,
                               receipt=dict(backend='inkling-cuda', rows=embedding.rows,
                                            embedding_sha256=embedding.sha256))
