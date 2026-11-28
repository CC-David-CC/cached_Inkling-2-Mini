import unittest
from unittest.mock import patch

from serve.cow_qwen35_prompt import Qwen35PromptEncoder, Service


class QwenSystemPrefixTests(unittest.TestCase):
    def test_user_only_default_system_preamble_does_not_break_prefix(self):
        encoder = object.__new__(Qwen35PromptEncoder)
        supplied = '<|im_start|>system\nSupplied instructions<|im_end|>\n'
        default = '<|im_start|>system\nDefault instructions<|im_end|>\n'

        def render(instance, messages, tools, kwargs):
            prefix = supplied if len(messages) == 2 else default
            return prefix + '<|im_start|>user\n' + messages[-1]['content'] + '<|im_end|>\n'

        with patch.object(Service, 'render_prompt', side_effect=render):
            result = encoder.render_prompt(
                [{'role': 'system', 'content': 'Supplied instructions'}], [], {})
        self.assertEqual(result, supplied)
