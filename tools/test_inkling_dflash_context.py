"""CPU regression tests for Inkling DFlash2 checkpoint layout and publication."""
from array import array
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, call, patch
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experimental/branching_context/src"))
from context_time_travel.inkling_dflash_context import restore_hidden_context, save_hidden_context, verify_greedy_block


class HiddenContextTest(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "context"
        self.context = torch.arange(4 * 9216).remainder(997).to(torch.bfloat16).reshape(1, 4, 9216)

    def test_layout_and_restore_preserve_unused_capacity(self):
        for prompt in (1, 3):
            with self.subTest(prompt=prompt):
                digest = save_hidden_context(self.path, self.context, prompt_tokens=prompt, saved_tokens=3)
                self.assertEqual(digest, hashlib.sha256(self.path.read_bytes()).hexdigest())
                taps = self.context[0].reshape(4, 4, 2304)
                expected = torch.cat((taps[:prompt].transpose(0, 1).reshape(-1),
                                      taps[prompt:3].reshape(-1)))
                self.assertEqual(self.path.read_bytes(), expected.view(torch.uint16).numpy().tobytes())
                destination = torch.full_like(self.context, -1)
                restore_hidden_context(self.path, destination, prompt_tokens=prompt, saved_tokens=3,
                                       expected_sha256=digest)
                self.assertTrue(torch.equal(destination[:, :3], self.context[:, :3]))
                self.assertTrue((destination[:, 3:] == -1).all())

    def test_truncated_or_extended_file_leaves_destination_untouched(self):
        for delta in (-2, 2):
            with self.subTest(delta=delta):
                self.path.write_bytes(bytes(9216 * 2 + delta))
                destination = self.context.clone()
                with self.assertRaises(ValueError):
                    restore_hidden_context(self.path, destination, prompt_tokens=1, saved_tokens=1)
                self.assertTrue(torch.equal(destination, self.context))

    def test_corruption_rejected_before_copy(self):
        save_hidden_context(self.path, self.context, prompt_tokens=2, saved_tokens=3)
        raw = self.path.read_bytes()
        expected = hashlib.sha256(raw).hexdigest()
        destination = torch.zeros_like(self.context)
        restore_hidden_context(self.path, destination, prompt_tokens=2, saved_tokens=3,
                               expected_sha256=expected)
        self.path.write_bytes(bytes([raw[0] ^ 1]) + raw[1:])
        destination.zero_()
        with self.assertRaisesRegex(ValueError, "digest mismatch"):
            restore_hidden_context(self.path, destination, prompt_tokens=2, saved_tokens=3,
                                   expected_sha256=expected)
        self.assertEqual(destination.count_nonzero().item(), 0)

    def test_failed_publication_preserves_previous_checkpoint(self):
        self.path.write_bytes(b"previous checkpoint")
        with patch("context_time_travel.inkling_dflash_context.os.replace", side_effect=OSError):
            with self.assertRaises(OSError):
                save_hidden_context(self.path, self.context, prompt_tokens=1, saved_tokens=2)
        self.assertEqual(self.path.read_bytes(), b"previous checkpoint")
        self.assertEqual(list(self.path.parent.iterdir()), [self.path])


class GreedyVerifierTest(unittest.TestCase):
    def test_stop_at_anchor_or_accepted_successor(self):
        for stop, expected in [(10, [10]), (20, [10, 20])]:
            for native in (False, True):
                with self.subTest(stop=stop, native=native):
                    backend = Mock(vocab_size=64, context_capacity=8)
                    backend.position.return_value = 0
                    backend.is_eog.side_effect = lambda token: native and token == stop
                    backend.select_greedy.side_effect = [10, 20, 30]
                    backend.consume_dflash_taps.return_value = array("f", [1.0]) * 9216
                    ctx = torch.zeros((1, 8, 9216), dtype=torch.bfloat16)
                    block = verify_greedy_block(backend, object(), ctx, [20, 30],
                                                remaining=7, stop_tokens=() if native else (stop,))
                    self.assertEqual(block, expected)
                    self.assertEqual(backend.consume_dflash_taps.call_count, len(expected))
                    self.assertEqual(ctx[:, 1 + len(expected):].count_nonzero().item(), 0)

    def test_rejection_preserves_next_position_for_target_continuation(self):
        lease = object()
        backend = Mock(vocab_size=64, context_capacity=8)
        backend.is_eog.return_value = False
        backend.position.side_effect = [0, 2]
        backend.select_greedy.side_effect = [10, 20, 30, 30]
        taps = array("f", [1.0]) * 9216
        backend.consume_dflash_taps.return_value = taps
        context = torch.zeros((1, 8, 9216), dtype=torch.bfloat16)

        committed = verify_greedy_block(backend, lease, context, [20, 42], remaining=7)
        self.assertEqual(committed, [10, 20])
        self.assertTrue((context[0, 1:3] == 1).all())
        self.assertEqual(context[0, 3:].count_nonzero().item(), 0)

        continuation = verify_greedy_block(backend, lease, context, [], remaining=1)
        self.assertEqual(continuation, [30])
        self.assertEqual(backend.consume_dflash_taps.call_args_list,
                         [call(lease, [10]), call(lease, [20]), call(lease, [30])])
        self.assertTrue((context[0, 3] == 1).all())


if __name__ == "__main__":
    unittest.main()
