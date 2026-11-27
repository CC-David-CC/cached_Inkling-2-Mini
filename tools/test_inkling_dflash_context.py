"""CPU regression tests for Inkling DFlash2 checkpoint layout and publication."""
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experimental/branching_context/src"))
from context_time_travel.inkling_dflash_context import restore_hidden_context, save_hidden_context


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


if __name__ == "__main__":
    unittest.main()
