#!/usr/bin/env python3
"""Run the Inkling CUDA backend through PCC's local token API."""
import argparse
import json
import os
from pathlib import Path
import secrets
import signal
import stat
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'experimental/branching_context/src')]

from context_time_travel.backend_registry import load_backend
from context_time_travel.coordinator import ContextCoordinator
from context_time_travel.inkling_backend import runtime_profile
from context_time_travel.pools import PoolConfig
from context_time_travel.real import BranchingModel
from serve.cow_adapter import CowAdapter
from serve.cow_http import make_server


def local_key(path):
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        pass
    else:
        with os.fdopen(fd, 'w', encoding='ascii') as stream:
            stream.write(secrets.token_urlsafe(32) + '\n')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, encoding='ascii') as stream:
        details = os.fstat(stream.fileno())
        if not stat.S_ISREG(details.st_mode) or details.st_mode & 0o077 or details.st_uid != os.getuid():
            raise PermissionError('The PCC API key file must be private and owned by this account')
        key = stream.read(256).strip()
        if not 32 <= len(key) < 256 or not key.isascii() or any(char.isspace() for char in key):
            raise ValueError('Invalid PCC API key')
    return key


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runtime', type=Path)
    parser.add_argument('--store', type=Path)
    parser.add_argument('--port', type=int, default=18061)
    args = parser.parse_args()
    runtime = args.runtime.resolve(strict=True)
    directory = args.store or runtime / 'pcc-http-store'
    key = local_key(runtime / 'pcc-http.key')
    profile = runtime_profile(runtime)

    def factory(gpu):
        backend = load_backend(profile, gpu_id=gpu)
        try:
            return BranchingModel(directory, backend, profile)
        except BaseException:
            backend.close()
            raise

    config = PoolConfig(gpus=(0,), ram_bytes=4 * 1024**3, disk_bytes=16 * 1024**3,
                        min_available_bytes=8 * 1024**3)
    coordinator = ContextCoordinator(directory, config, factory)
    try:
        coordinator.ready()
        adapter = CowAdapter(coordinator)
        with make_server(adapter, '127.0.0.1', args.port, key,
                         connection_info={name: profile[name] for name in
                                          ('model_contract_id', 'execution_profile_id')}
                         | {'eos_tokens': profile['execution']['eos_tokens']}) as server:
            def stop(_signum, _frame):
                raise KeyboardInterrupt

            signal.signal(signal.SIGTERM, stop)
            print(json.dumps({'event': 'ready', 'address': server.server_address}), flush=True)
            server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        coordinator.close()


if __name__ == '__main__':
    main()
