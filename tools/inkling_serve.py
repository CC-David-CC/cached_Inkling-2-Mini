#!/usr/bin/env python3
"""Launch the pinned Strata HTTP adapter for a single Inkling request."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="engine, checkpoint, tokenizer and Strata checkout directory")
    parser.add_argument("--port", type=int, default=18060)
    args = parser.parse_args()
    root = args.directory.resolve()
    terminal = root / "strata-source" / "terminal"

    with TemporaryDirectory(prefix="inkling-serve-", dir=root) as temporary:
        config = Path(temporary) / "config.json"
        config.write_text(json.dumps({
            "arch": "inkling2_loop",
            "exe": str(root / "strata-inkling"),
            "args": [
                "--native", str(root / "Inkling-2-Mini-SQ-20.strata"),
                "--gpus", "1", "--parallel", "1", "--max-context", "8192",
                "--kv", "fp16", "--draft", "off",
            ],
            "cwd": str(root),
            "tokenizer": str(root / "tokenizer"),
            "model_name": "Inkling-2-Mini",
            "log": str(root / "engine.log"),
            "host": "127.0.0.1",
            "port": args.port,
        }), encoding="utf-8")
        return subprocess.call([
            sys.executable, str(terminal / "serve" / "server.py"),
            "--engine", "strata", "--config", str(config),
            "--host", "127.0.0.1", "--port", str(args.port),
            "--parallel", "1", "--strict-model", "--offline", "--no-api-key",
        ], cwd=terminal)


if __name__ == "__main__":
    sys.exit(main())
