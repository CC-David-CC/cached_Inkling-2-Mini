#!/usr/bin/env python3
"""Launch the pinned Strata HTTP adapter for a single Inkling request."""
import argparse
import json
from pathlib import Path
import signal
import subprocess
import sys
from tempfile import TemporaryDirectory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="engine, checkpoint, tokenizer and Strata checkout directory")
    parser.add_argument("--port", type=int, default=18060)
    parser.add_argument("--vision", action="store_true", help="load the SQ-20 companion vision tower")
    parser.add_argument("--draft", choices=("off", "dspark"), default="off",
                        help="select the native Inkling drafter (default: off)")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    root = args.directory.resolve()
    terminal = root / "strata-source" / "terminal"

    engine_args = [
        "--native", str(root / "Inkling-2-Mini-SQ-20.strata"),
        "--gpus", "1", "--parallel", "1", "--max-context", "8192",
        "--kv", "fp16", "--draft", args.draft,
    ]
    if args.vision:
        vision = root / "Inkling-2-Mini-Vision.strata"
        if not vision.is_file():
            parser.error(f"vision checkpoint missing: {vision}")
        engine_args.extend(("--vision", str(vision)))

    with TemporaryDirectory(prefix="inkling-serve-", dir=root) as temporary:
        config = Path(temporary) / "config.json"
        config.write_text(json.dumps({
            "arch": "inkling2_loop",
            "exe": str(root / "strata-inkling"),
            "args": engine_args,
            "cwd": str(root),
            "tokenizer": str(root / "tokenizer"),
            "model_name": "Inkling-2-Mini",
            "log": str(root / ("engine.log" if args.port == 18060 else f"engine-{args.port}.log")),
            "host": "127.0.0.1",
            "port": args.port,
        }), encoding="utf-8")
        with subprocess.Popen([
            sys.executable, str(terminal / "serve" / "server.py"),
            "--engine", "strata", "--config", str(config),
            "--host", "127.0.0.1", "--port", str(args.port),
            "--parallel", "1", "--strict-model", "--offline", "--no-api-key",
        ], cwd=terminal, start_new_session=True) as server:
            def forward_shutdown(signum, _frame):
                if server.poll() is None:
                    server.send_signal(signum)

            signal.signal(signal.SIGINT, forward_shutdown)
            signal.signal(signal.SIGTERM, forward_shutdown)
            return server.wait()


if __name__ == "__main__":
    sys.exit(main())
