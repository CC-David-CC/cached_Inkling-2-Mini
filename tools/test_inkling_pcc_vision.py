"""Verify vision identity and exact PCC restore on llm-60 in separate processes."""
import argparse
import base64
from pathlib import Path

from context_time_travel.backend_registry import load_backend
from context_time_travel.input_spans import InputHistory
from context_time_travel.pools import DiskCheckpointPool, SharedCheckpointPool
from context_time_travel.real import BranchingModel
from context_time_travel.vision_artifacts import ImageArtifacts
from test_inkling_http import solid_png
from test_inkling_pcc import exact_row, profile


def verify_images(model, workspace):
    backend = model.backend
    ram = SharedCheckpointPool(32 * 1024**2)
    disk = DiskCheckpointPool(model.store, 64 * 1024**2)
    artifacts = ImageArtifacts(model.store, ram, disk,
                               encoder_profile_id=backend.encoder_profile_id,
                               expected_width=2304, original_bytes_limit=32 * 1024**2)
    roots, frontiers = [], []
    try:
        for name, rgb in (('red', (255, 0, 0)), ('blue', (0, 0, 255))):
            picture = base64.b64decode(solid_png(rgb).split(',', 1)[1])
            with artifacts.prepare(original=picture, normalized=picture, encode=backend.encode_image,
                                   pad_token_id=backend.image_pad) as item:
                spans = [{'kind': 'text', 'tokens': [100, 101]},
                         item.span, {'kind': 'text', 'tokens': [102, 103]}]
                with backend.lease() as lease:
                    backend.consume_inputs(lease, spans, embeddings={item.embedding.sha256: item.embedding})
                    frontier = exact_row(backend.logits(lease))
                    backend.consume(lease, (106,))
                    expected = exact_row(backend.logits(lease))
                root = model.create_inputs(spans, artifacts=artifacts)
                assert isinstance(model.history(root), InputHistory)
                assert exact_row(model.next_logits(root)) == frontier
                assert model.create_inputs(spans, artifacts=artifacts) == root
                before = backend.work_counters()['consumed_tokens']
                model.feed(root, (104, 105))
                assert exact_row(model.next_logits(model.feed(root, (106,)))) == expected
                assert exact_row(model.next_logits(root)) == frontier
                assert backend.work_counters()['consumed_tokens'] - before == 3
                model.checkpoint(root)
                (workspace / f'{name}.root').write_text(root)
                (workspace / f'{name}.frontier').write_bytes(frontier)
                (workspace / f'{name}.next').write_bytes(expected)
                roots.append(root)
                frontiers.append(frontier)
        assert roots[0] != roots[1] and frontiers[0] != frontiers[1]
        assert model.history(roots[0]).native_tokens == model.history(roots[1]).native_tokens
        print('Two image identities and exact RAM continuations verified with identical text and pad IDs')
    finally:
        ram.close()


def verify_restore(model, workspace):
    for name in ('red', 'blue'):
        root = (workspace / f'{name}.root').read_text()
        receipt = model.restore_checkpoint(root)
        assert receipt['forward_tokens'] == 0
        assert exact_row(model.next_logits(root)) == (workspace / f'{name}.frontier').read_bytes()
        branch = model.feed(root, (106,))
        assert exact_row(model.next_logits(branch)) == (workspace / f'{name}.next').read_bytes()
    assert model.backend.work_counters()['consumed_tokens'] == 2
    print('Fresh vision checkpoint restore verified: 200058 exact logits per frontier, zero image or prefix replay')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runtime', type=Path)
    parser.add_argument('workspace', type=Path)
    parser.add_argument('mode', choices=('write', 'read'))
    args = parser.parse_args()
    config = profile(args.runtime.resolve(), vision=True)
    backend = load_backend(config)
    try:
        model = BranchingModel(args.workspace / 'pcc', backend, config)
        (verify_images if args.mode == 'write' else verify_restore)(model, args.workspace)
    finally:
        backend.close()


if __name__ == '__main__':
    main()
