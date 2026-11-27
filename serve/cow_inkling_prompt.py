"""Pinned Inkling byte-level tokenizer and greedy text prompt rendering."""
from functools import partial
import hashlib
from pathlib import Path
from importlib.util import module_from_spec, spec_from_file_location

from context_time_travel.inkling_backend import TOKENIZER_SHA256
from context_time_travel.real import digest_file

from .cow_prompt_encoder import CowPromptEncoder
from .frontend import ChatTemplate


TOKENIZER_IMPLEMENTATION_SHA256 = '2199b8c20e0012879dd5c944b8d2b0deccbcee94395485a59737d5e64d39a7de'
TEMPLATE_SHA256 = '53a9035f3255945704b5857657cfb00c1165090c0641c4f782a0cbf268cc16dc'


class InklingPromptEncoder(CowPromptEncoder):
    def render_prompt(self, messages, tools, kwargs):
        rendered = super().render_prompt(messages, tools, kwargs)
        if kwargs.get('add_generation_prompt') and kwargs.get('reasoning_effort') == 0:
            rendered += '<|message_model|><|content_text|>'
        return rendered

    def question_frame(self, canonical, controller, content):
        if not isinstance(content, str) or self.root_mode(canonical, controller) != 'context_only':
            raise ValueError('Inkling question requires a closed context root')
        return self._question_tokens(content)

    def followup_frame(self, canonical, controller, content):
        if controller.get('status') != 'stopped' or not isinstance(content, str):
            raise ValueError('Inkling follow-up requires a stopped answer')
        if not canonical or canonical[-1] not in controller.get('policy', {}).get('eos_tokens', []):
            raise ValueError('Inkling follow-up requires a consumed stop token')
        return self._question_tokens(content)

    def _question_tokens(self, content):
        anchor = [dict(role='system', content='Reference.')]
        messages = anchor + [dict(role='user', content=content)]
        prefix = tuple(self.encode_rendered(anchor, self.render(anchor, add_generation_prompt=False), False))
        full = tuple(self.encode_rendered(messages, self.render(messages, add_generation_prompt=True), True))
        if full[:len(prefix)] != prefix:
            raise ValueError('Inkling question changed the context prefix')
        return full[len(prefix):]

    def root_mode(self, canonical, controller):
        if controller.get('status') == 'idle':
            opener = tuple(self.tok.encode('<|message_model|><|content_text|>', parse_special=True))
            if tuple(canonical[-len(opener):]) == opener:
                return 'prepared_answer'
            if canonical and canonical[-1] == self.tok.special_tokens['<|end_message|>']:
                return 'context_only'
        return 'unsupported'


class InklingTextDecoder:
    """Render the visible channel of a non-thinking assistant completion."""
    def __init__(self, tokenizer):
        self._tokenizer = tokenizer
        self._framing = {tokenizer.special_tokens[tag] for tag in (
            '<|message_model|>', '<|content_text|>', '<|end_message|>',
        )}

    def token_bytes(self, token):
        return b'' if token in self._framing else self._tokenizer.token_bytes(token)


def text_components(runtime, *, vision=False):
    source = runtime / 'strata-source/terminal/tools/strata_tokenizer.py'
    tokenizer_path = runtime / 'tokenizer-source.json'
    template_path = runtime / 'tokenizer/chat_template.jinja'
    for path, expected in ((source, TOKENIZER_IMPLEMENTATION_SHA256),
                           (tokenizer_path, TOKENIZER_SHA256), (template_path, TEMPLATE_SHA256)):
        if digest_file(path) != expected:
            raise ValueError(f'Inkling frontend fingerprint mismatch: {path.name}')

    spec = spec_from_file_location('_inkling_strata_tokenizer', source)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    tokenizer = module.Tokenizer.from_hf(tokenizer_path, pre='o200k',
                                         special_ids={'bos': 200028, 'eos': 200006, 'pad': 199999})
    encoder = InklingPromptEncoder(tokenizer, ChatTemplate(template_path), {'reasoning_effort': 0})
    components = dict(tokenizer=InklingTextDecoder(tokenizer), render=encoder.render,
                      encode_rendered=encoder.encode_rendered)
    renderer_id = hashlib.sha256(Path(__file__).read_bytes()
                                 + bytes.fromhex(TOKENIZER_IMPLEMENTATION_SHA256)
                                 + bytes.fromhex(TOKENIZER_SHA256)
                                 + bytes.fromhex(TEMPLATE_SHA256)).hexdigest()
    components.update(renderer_id=renderer_id, initial_root_mode=encoder.root_mode,
                      append_initial_user_suffix=encoder.question_frame, append_user_suffix=encoder.followup_frame)
    if vision:
        from .cow_inkling_vision import InklingVisionPromptEncoder
        opener = tuple(tokenizer.encode('<|message_model|><|content_text|>', parse_special=True))
        components.update(renderer_id=renderer_id,
                          mixed_renderer=InklingVisionPromptEncoder(encoder, renderer_id=renderer_id),
                          initial_root_mode=partial(initial_root_mode, opener=opener))
    return components


def initial_root_mode(tokens, controller, *, opener):
    if controller.get('status') == 'idle' and tuple(tokens[-len(opener):]) == opener:
        return 'prepared_answer'
    return 'unsupported'
