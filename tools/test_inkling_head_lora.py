"""Exercise trained Inkling output-head adapters through PCC on CUDA."""
import argparse
import hashlib
from pathlib import Path

from context_time_travel.backend_registry import load_backend
from context_time_travel.inkling_backend import PACK_HASHES, runtime_profile
from context_time_travel.real import BranchingModel
from test_inkling_pcc import assert_rejected, exact_row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runtime', type=Path)
    parser.add_argument('workspace', type=Path)
    parser.add_argument('adamw', type=Path)
    parser.add_argument('muon', type=Path)
    parser.add_argument('--pack', choices=tuple(PACK_HASHES), default='SQ-20')
    args = parser.parse_args()
    runtime = args.runtime.resolve()
    ids = (100, 101, 102, 103)
    results = {}
    snapshots = {}

    def capture(name, adapter=None):
        profile = runtime_profile(runtime, adapter=adapter, pack=args.pack)
        backend = load_backend(profile)
        try:
            model = BranchingModel(args.workspace, backend, profile)
            before = backend.work_counters()['consumed_tokens']
            for snapshot in snapshots.values():
                assert_rejected(lambda: model.restore_checkpoint(snapshot),
                                'Snapshot model contract mismatch')
            assert backend.work_counters()['consumed_tokens'] == before
            root = model.create(ids)
            frontier = exact_row(model.next_logits(root))
            saved = model.checkpoint(root)
            assert saved['payload_path'].is_file()
            receipt = model.restore_checkpoint(root)
            assert receipt['forward_tokens'] == 0
            assert exact_row(model.next_logits(root)) == frontier
            with backend.lease() as lease:
                backend.consume(lease, ids)
                assert exact_row(backend.logits(lease)) == frontier
            print(f'{name}: frontier SHA-256 {hashlib.sha256(frontier).hexdigest()}', flush=True)
            return root, frontier
        finally:
            backend.close()

    for name, adapter in (('base', None), ('adamw', args.adamw), ('muon', args.muon)):
        snapshots[name], results[name] = capture(name, adapter)
    assert len(set(snapshots.values())) == 3
    assert len(set(results.values())) == 3

    base_root = snapshots.pop('base')
    restored_root, restored = capture('base restored')
    assert restored_root == base_root and restored == results['base']
    print('Both adapter frontiers are reproducible, checkpoints are isolated, and disabling restores base logits')


if __name__ == '__main__':
    main()
