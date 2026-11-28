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
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'experimental/branching_context/src')]

from context_time_travel.backend_registry import load_backend
from context_time_travel.coordinator import ContextCoordinator
from context_time_travel.inkling_backend import runtime_profile
from context_time_travel.pools import PoolConfig
from context_time_travel.real import BranchingModel
from context_time_travel.store import Store
from serve.cow_adapter import CowAdapter
from serve.cow_http import make_server
from serve.cow_workspace import CowWorkspace
from serve.cow_inkling_prompt import text_components
from serve.cow_inkling_effort import InklingTextAdapter, InklingWorkspaceDecoder
from serve.cow_vision import CowImageNormalizer
from serve.cow_inkling_vision import InklingVisionPreparer


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
    storage = parser.add_mutually_exclusive_group()
    storage.add_argument('--store', type=Path, help='persistent checkpoint and branch directory')
    storage.add_argument('--persistent', action='store_true', help='use the profile’s runtime checkpoint directory')
    parser.add_argument('--greedy-text-stops', action='store_true', help='Enable checkpointed stop strings for undrafted greedy text')
    parser.add_argument('--serial-followup', action='store_true', help='Consume text generation request inputs one token at a time after loading the parent')
    parser.add_argument('--cache-transfer', action='store_true',
                        help='Enable authenticated export of saved checkpoints to matching execution profiles')
    parser.add_argument('--suffix-checkpoints', action='store_true', help='Save eligible text branches as parent-bound suffixes')
    parser.add_argument('--ram-cache-gib', type=int, help='resident checkpoint pool capacity in GiB')
    parser.add_argument('--adapter', type=Path, help='trained Inkling head LoRA artifact')
    parser.add_argument('--grammar-store', type=Path, help='enable BranchWrite inline grammar registration')
    parser.add_argument('--capacity', type=int, default=8192, help='total prompt and generation token budget')
    parser.add_argument('--pack', choices=('SQ-10', 'SQ-15', 'SQ-20'), default='SQ-20')
    parser.add_argument('--kv-offload', choices=('off', 'auto', 'on'), default='off',
                        help='Place global KV in GPU memory, split automatically, or use host RAM')
    parser.add_argument('--require-gpu-experts', action='store_true',
                        help='Reject startup if the loaded Inkling model has any offloaded experts')
    parser.add_argument('--native-expert-selection', action='store_true',
                        help='Keep native cold-expert selection when STRATA_INK_EXACT is unset')
    parser.add_argument('--gpus', type=int, choices=(1, 2, 3), default=1)
    parser.add_argument('--dspark', action='store_true', help='experimental fused DSpark for greedy text requests')
    parser.add_argument('--dflash2', nargs=3, type=Path, metavar=('REFERENCE', 'DRAFT_DIR', 'TARGET'),
                        help='experimental greedy DFlash2 using pinned local assets')
    parser.add_argument('--dflash2-resident', action='store_true', help='keep DFlash2 target embedding and head on GPU')
    parser.add_argument('--vision', action='store_true')
    parser.add_argument('--port', type=int, default=18061, help='TCP port (0 selects an available port)')
    args = parser.parse_args()
    if not 0 <= args.port <= 65535:
        parser.error('--port must be between 0 and 65535')
    if args.ram_cache_gib is not None and args.ram_cache_gib <= 0:
        parser.error('--ram-cache-gib must be positive')
    if args.suffix_checkpoints and (args.dspark or args.dflash2 or args.vision or args.capacity <= 8192 or args.gpus != 1):
        parser.error('Suffix checkpoints require undrafted single-GPU text with capacity above 8192')
    if (args.pack == 'SQ-20' and args.gpus == 1 and not args.dspark and not args.dflash2
            and not args.native_expert_selection):
        os.environ.setdefault('STRATA_INK_EXACT', '1')
    if args.store is not None or args.persistent:
        serve(args, None)
    else:
        # Keep PCC's atomic publication and branch semantics inside a private tmpfs store.
        with tempfile.TemporaryDirectory(prefix='cached-inkling-', dir='/dev/shm') as temporary:
            serve(args, Path(temporary))


def serve(args, temporary):
    runtime = args.runtime.resolve(strict=True)
    key = local_key(runtime / 'pcc-http.key')
    normalizer = CowImageNormalizer() if args.vision else None
    profile = runtime_profile(runtime, vision=args.vision, capacity=args.capacity, serial_followup=args.serial_followup, greedy_text_stops=args.greedy_text_stops, draft=args.dspark, gpus=args.gpus, kv_offload=args.kv_offload, pack=args.pack, dflash2=args.dflash2, dflash2_resident=args.dflash2_resident, grammar=bool(args.grammar_store),
                              normalizer_id=normalizer.identity if normalizer else None, adapter=args.adapter)
    directory = temporary or args.store or (
        runtime / 'pcc-stores' / profile['model_contract_id'] / profile['execution_profile_id'])
    ram_bytes = (args.ram_cache_gib * 1024**3 if args.ram_cache_gib is not None
                 else max(8 * 1024**3, args.capacity * 262144))
    profile['require_gpu_experts'] = args.require_gpu_experts
    profile['ram_cache_bytes'] = ram_bytes
    profile['suffix_checkpoints'] = args.suffix_checkpoints
    store_bytes = max(16 * 1024**3, args.capacity * 524288)
    if temporary is not None:
        fs = os.statvfs(temporary)
        store_bytes = min(store_bytes, fs.f_bavail * fs.f_frsize)
    components = text_components(runtime, vision=args.vision)
    catalog = None
    if args.grammar_store:
        from serve.cow_grammar_disk import DurableGrammarCatalog
        catalog = DurableGrammarCatalog(args.grammar_store, components['render'].__self__.tok,
                                        profile['execution']['eos_tokens'])
    preparer = (InklingVisionPreparer(Store(directory), normalizer, components['mixed_renderer'], profile)
                if args.vision else None)

    def factory(gpu):
        backend = load_backend(profile, gpu_id=gpu)
        try:
            return BranchingModel(directory, backend, profile)
        except BaseException:
            backend.close()
            raise

    config = PoolConfig(gpus=(0,), ram_bytes=ram_bytes,
                        disk_bytes=store_bytes, cache_transfer=args.cache_transfer,
                        min_available_bytes=8 * 1024**3)
    coordinator = ContextCoordinator(directory, config, factory, input_preparer=preparer)
    try:
        coordinator.ready()
        adapter = CowAdapter(coordinator)
        text = InklingTextAdapter(adapter, **components, vision_preparer=preparer)
        workspace = CowWorkspace(coordinator.store, InklingWorkspaceDecoder(text.render.__self__.tok),
                                 root_mode_resolver=text.root_mode_for_projection)
        with make_server(adapter, '127.0.0.1', args.port, key, text_adapter=text, workspace=workspace,
                         grammar_catalog=catalog, connection_info={name: profile[name] for name in
                                          ('model_contract_id', 'execution_profile_id')}
                         | {'eos_tokens': profile['execution']['eos_tokens']}) as server:
            def stop(_signum, _frame):
                raise KeyboardInterrupt

            signal.signal(signal.SIGTERM, stop)
            print(json.dumps({'event': 'ready', 'address': server.server_address, 'store': str(directory),
                              'storage': 'temporary RAM' if temporary else 'persistent',
                              'ram_cache_bytes': ram_bytes, 'store_limit_bytes': store_bytes}), flush=True)
            server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        coordinator.close()
        if catalog is not None:
            catalog.close()


if __name__ == '__main__':
    main()
