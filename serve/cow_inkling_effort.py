"""Numeric effort rendering and native Inkling text channel presentation."""
import codecs
from copy import copy

from .cow_text_adapter import CowTextAdapter
from .cow_inkling_prompt import InklingTextDecoder


class InklingChannels:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.special = tokenizer.special_tokens
        self.channel = 'text'
        self.decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')

    def push(self, token, terminal=False):
        channel = self.channel
        if token == self.special['<|content_thinking|>']:
            self.channel = 'reasoning'
        elif token in (self.special['<|content_text|>'], self.special['<|message_model|>'],
                       self.special['<|end_message|>']) or terminal:
            self.channel = 'text'
        else:
            return channel, self.decoder.decode(self.tokenizer.token_bytes(token))
        tail = self.decoder.decode(b'', final=True)
        self.decoder.reset()
        return channel, tail


class InklingWorkspaceDecoder(InklingTextDecoder):
    def visible_assistant_text(self, tokens, eos):
        channels = InklingChannels(self._tokenizer)
        parts = []
        for token in tokens:
            terminal = token in eos
            channel, text = channels.push(token, terminal=terminal)
            if channel == 'text':
                parts.append(text)
            if terminal:
                break
        return ''.join(parts)


class InklingChannelRequest:
    def __init__(self, view, tokenizer, channel="text", pending=b""):
        self.request = view.request
        self.snapshot = view.snapshot
        self.durable = view.durable
        self.input_preparation = getattr(view, 'input_preparation', None)
        self.tokenizer = tokenizer
        self.initial_channel = channel
        self.pending = pending
        self.channels = self.new_channels()

    def new_channels(self):
        channels = InklingChannels(self.tokenizer)
        channels.channel = self.initial_channel
        channels.decoder.setstate((self.pending, 0))
        return channels

    def decode_event(self, event):
        if event.get('event') != 'token':
            return [event]
        token = event['token']
        terminal = event.get('turn_complete', False)
        channel, piece = self.channels.push(token, terminal)
        events = [dict(event=channel, text=piece, token_id=token)] if piece else []
        if terminal:
            events.append(dict(event='turn_complete', token_id=token))
        return events

    def text_result(self):
        return self.channel_result()['text']

    def channel_result(self):
        channels = self.new_channels()
        parts = {'text': [], 'reasoning_content': []}
        eos = {self.tokenizer.special_tokens[tag] for tag in
               ('<|content_model_end_sampling|>', '<|endoftext|>')}
        for token in self.snapshot.result()['tokens']:
            channel, piece = channels.push(token, token in eos)
            parts['reasoning_content' if channel == 'reasoning' else 'text'].append(piece)
        return {name: ''.join(pieces) for name, pieces in parts.items()}


class InklingTextAdapter(CowTextAdapter):
    def with_reasoning_effort(self, effort):
        if type(effort) not in (float, int) or not 0 <= effort <= 0.99:
            raise ValueError('reasoning_effort must be a JSON number in [0, 0.99]')
        if effort == 0:
            return self
        adapter = copy(self)
        encoder = copy(self.render.__self__)
        encoder.template_kwargs = dict(encoder.template_kwargs, reasoning_effort=effort)
        adapter.render = encoder.render
        adapter.encode_rendered = encoder.encode_rendered
        return adapter

    def _present_turn(self, view, options):
        if options.get('text_stops') or options.get('grammar_state'):
            return view
        return InklingChannelRequest(view, self.render.__self__.tok)

    def ask(self, *args, **kwargs):
        return self._present_turn(super().ask(*args, **kwargs), kwargs)

    def answer(self, *args, **kwargs):
        return self._present_turn(super().answer(*args, **kwargs), kwargs)

    def append_user(self, *args, **kwargs):
        return self._present_turn(super().append_user(*args, **kwargs), kwargs)

    def generate(self, *args, **kwargs):
        if self.render.__self__.template_kwargs['reasoning_effort'] == 0:
            return self._present_turn(super().generate(*args, **kwargs), kwargs)
        if kwargs.get('text_stops') or kwargs.get('grammar_state'):
            raise ValueError('Positive effort with a grammar or text stops is unsupported')
        view = super().generate(*args, **kwargs)
        return InklingChannelRequest(view, self.render.__self__.tok)

    def resume(self, branch, revision, operation_id, max_tokens, **kwargs):
        _, ids, controller = self.token_adapter.source_state(branch, revision, operation_id)
        ids = self._text_tail(ids)
        tokenizer = self.render.__self__.tok
        special = tokenizer.special_tokens
        # The last user frame ends before the generated assistant channel markers.
        # If the request stopped on the model header, its channel is still undecided.
        channel = 'text'
        boundaries = {special[tag] for tag in (
            '<|message_user|>', '<|message_model|>', '<|content_text|>', '<|end_message|>',
        )}
        for token in reversed(ids):
            if token in boundaries:
                break
            if token == special['<|content_thinking|>']:
                channel = 'reasoning'
                break
        view = super().resume(branch, revision, operation_id, max_tokens, **kwargs)
        if (controller.get('text_stop') or controller.get('constraint') or
                self.render.__self__.tok.special_tokens['<|message_user|>'] not in ids):
            return view
        return InklingChannelRequest(view, tokenizer, channel,
                                     bytes.fromhex(controller.get('output_pending_hex', '')))
