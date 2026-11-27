"""Pinned Inkling byte-level tokenizer and greedy text prompt rendering."""
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
        if kwargs.get('add_generation_prompt'):
            rendered += '<|message_model|><|content_text|>'
        return rendered


class InklingTextDecoder:
    """Render the visible channel of a non-thinking assistant completion."""
    def __init__(self, tokenizer):
        self._tokenizer = tokenizer
        self._framing = {tokenizer.special_tokens[tag] for tag in (
            '<|message_model|>', '<|content_text|>', '<|end_message|>',
        )}

    def token_bytes(self, token):
        return b'' if token in self._framing else self._tokenizer.token_bytes(token)


def text_components(runtime):
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
    encoder = InklingPromptEncoder(tokenizer, ChatTemplate(template_path), {'reasoning_effort': 'none'})
    return dict(tokenizer=InklingTextDecoder(tokenizer), render=encoder.render,
                encode_rendered=encoder.encode_rendered)
