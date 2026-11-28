"""An authenticated RAM checkpoint can check each stored byte without rehashing."""
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from context_time_travel.real import BranchingModel, Checkpoint
from context_time_travel.store import IntegrityError


class ResidentPayloadTests(unittest.TestCase):
    def test_corruption_and_missing_owner(self):
        blob = b'x' * (8 * 1024**2 + 3)
        owner = Checkpoint.create('model', 'profile', 7, blob, (1.,))
        manifest = dict(payload_sha256=owner.checksum, state_bytes=len(blob), schema=1)
        with tempfile.TemporaryDirectory() as directory:
            model = SimpleNamespace(profile={'backend': 'inkling-cuda'}, backend=SimpleNamespace(vocab_size=1),
                                    checkpoint_directory=Path(directory))
            path = Path(directory) / (owner.checksum + '.state')
            payload = blob + struct.pack('<f', 1.)
            path.write_bytes(payload)
            def validate():
                return BranchingModel._validate_checkpoint_payload(model, manifest)
            with patch('context_time_travel.real.digest_file', side_effect=AssertionError('unexpected hash')):
                self.assertEqual(validate(), manifest)
                for offset in (0, 8*1024**2, len(payload)-1):
                    with path.open('r+b') as file:
                        file.seek(offset)
                        file.write(b'!')
                    with self.assertRaises(IntegrityError):
                        validate()
                    path.write_bytes(payload)
                for bad in (payload[:-1], payload + b'!'):
                    path.write_bytes(bad)
                    with self.assertRaises(IntegrityError):
                        validate()
            path.write_bytes(payload)
            del owner
            self.assertEqual(validate(), manifest)
            path.write_bytes(b'!' + payload[1:])
            with self.assertRaises(IntegrityError):
                validate()
