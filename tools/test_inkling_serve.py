"""Exercise the native launcher CLI without loading model weights."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


LAUNCHER = Path(__file__).with_name("inkling_serve.py")


class RuntimeDirectoryTests(unittest.TestCase):
    def test_runtime_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = root / "runtime"
            server = runtime / "strata-source/terminal/serve/server.py"
            server.parent.mkdir(parents=True)
            server.write_text(
                "import json, sys\n"
                "from pathlib import Path\n"
                "config = Path(sys.argv[sys.argv.index('--config') + 1])\n"
                "print(json.dumps(json.loads(config.read_text())['args']))\n"
            )
            (runtime / "Inkling-2-Mini-Vision.strata").touch()
            alias = root / "runtime link"
            alias.symlink_to(runtime, target_is_directory=True)
            plain = root / "regular file"
            plain.touch()
            for path in (root / "missing", plain, runtime, alias):
                for vision in (False, True):
                    with self.subTest(path=path.name, vision=vision):
                        command = [sys.executable, str(LAUNCHER), str(path)]
                        if vision:
                            command.append("--vision")
                        result = subprocess.run(command, capture_output=True, text=True, timeout=10)
                        if path.is_dir():
                            self.assertEqual(result.returncode, 0, result.stderr)
                            self.assertEqual("--vision" in json.loads(result.stdout), vision)
                        else:
                            self.assertEqual(result.returncode, 2, result.stderr)
                            self.assertIn("runtime directory", result.stderr)
                            self.assertIn(str(path), result.stderr)
                            self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
