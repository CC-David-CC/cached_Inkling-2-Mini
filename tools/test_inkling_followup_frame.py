"""Verify text-stop follow-up framing against the pinned template on CPU."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'experimental/branching_context/src')]

from serve.cow_inkling_prompt import text_components


def verify(runtime):
    encoder = text_components(runtime)['render'].__self__
    messages = [
        dict(role='user', content='Print two lines.'),
        dict(role='assistant', content='ORANGE\n'),
        dict(role='user', content='What comes next?'),
    ]
    rendered = encoder.render(messages, add_generation_prompt=True)
    expected = tuple(encoder.encode_rendered(messages, rendered, True))
    end = encoder.tok.special_tokens['<|content_model_end_sampling|>']
    close = encoder.tok.special_tokens['<|end_message|>']
    boundary = expected.index(end)
    assert expected[boundary - 1] == close
    for consumed, matched in ((boundary - 1, True), (boundary, True), (boundary + 1, False)):
        parent = expected[:consumed]
        controller = dict(status='stopped', policy=dict(eos_tokens=[end]),
                          text_stop=dict(stopped=matched))
        suffix = encoder.followup_frame(parent, controller, messages[-1]['content'])
        assert parent + suffix == expected, f'Frame mismatch at token {consumed}'
    try:
        encoder.followup_frame((), dict(status='stopped', policy=dict(eos_tokens=[end]),
                                        text_stop=dict(stopped=True)), messages[-1]['content'])
    except ValueError:
        pass
    else:
        raise AssertionError('Accepted a text stop with empty token history')
    for status in ('active', 'stopped'):
        controller = dict(status=status, policy=dict(eos_tokens=[end]))
        try:
            encoder.followup_frame(expected[:boundary - 1], controller, messages[-1]['content'])
        except ValueError:
            continue
        raise AssertionError(f'Accepted an unfinished {status} frame')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runtime', type=Path)
    verify(parser.parse_args().runtime)
    print('All six follow-up frame cases passed')
