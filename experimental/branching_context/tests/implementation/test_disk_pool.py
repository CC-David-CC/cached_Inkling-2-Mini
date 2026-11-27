"""Real filesystem pool/retention tests; synthetic checkpoint payloads."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from context_time_travel.checkpoint_cache import validate_manifest
from context_time_travel.store import IntegrityError, Store, _json_bytes
from context_time_travel.pools import DiskCheckpointPool, PoolFull


class DiskPoolTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.store = Store(self.tmp.name)
        self.root = self.store.create_root('m',(1,),{'status':'idle'})
        self.tip = self.store.append(self.root,(2,),{'status':'idle'})
        self.store.fork('main',self.tip)
        self.pool = DiskCheckpointPool(self.store,64)

    def test_history_roots_tips_and_saved_points_do_not_permanently_pin_numerical_copies(self):
        later = self.store.append(self.tip,(3,),{'status':'idle'})
        self.store.move('main',0,later,'later')
        self.assertEqual(self.store.checkpoint_roots(),set())
        self.assertTrue({self.root,later} <= self.store.retained_snapshots())
        self.store.retain('fork-point',self.tip)
        self.assertEqual(self.store.checkpoint_roots(),set())
        self.assertTrue({self.root,self.tip,later} <= self.store.retained_snapshots())

    def test_read_lease_is_a_disk_eviction_pin(self):
        later = self.store.append(self.tip,(3,),{'status':'idle'})
        self.store.move('main',0,later,'later')
        with self.store.read_lease(self.tip): self.assertIn(self.tip,self.store.checkpoint_roots())
        self.assertNotIn(self.tip,self.store.checkpoint_roots())

    def test_orphan_payload_collection_never_deletes_canonical_history(self):
        file = self.pool.directory/('0'*64+'.state');file.write_bytes(b'x'*60)
        with self.pool.reserve(32):
            self.assertFalse(file.exists())
            self.assertEqual(self.pool.metrics()['reserved_bytes'],32)
        self.assertEqual(self.store.get_snapshot(self.tip)['inputs'],[2])

    def test_suffix_manifest_requires_independently_bound_parent(self):
        parent = dict(parent_snapshot=self.root, parent_consumed=1,
                      parent_payload_sha256='a' * 64)
        required = dict(schema=5, snapshot=self.tip, consumed=2, **parent)
        manifest = dict(required, positions='absolute-zero-based', input_identity='b' * 64,
                        model_contract_id='model', execution_profile_id='profile', logits_count=8,
                        state_bytes=32, payload_sha256='c' * 64)

        def envelope(values):
            return dict(manifest=values, checksum=hashlib.sha256(_json_bytes(values)).hexdigest())

        self.assertEqual(validate_manifest(envelope(manifest), required), manifest)
        for key in parent:
            incomplete = dict(required)
            del incomplete[key]
            with self.subTest(missing=key), self.assertRaises(IntegrityError):
                validate_manifest(envelope(manifest), incomplete)
        for changes in (dict(parent_snapshot=self.tip), dict(parent_payload_sha256='z' * 64),
                        dict(parent_consumed=True), dict(parent_consumed=2)):
            with self.subTest(changes=changes), self.assertRaises(IntegrityError):
                validate_manifest(envelope(dict(manifest, **changes)), dict(required, **changes))
        with self.assertRaises(IntegrityError):
            validate_manifest(envelope(dict(manifest, parent_payload_sha256='d' * 64)), required)

    def test_suffix_read_lease_protects_parent_and_both_become_evictable(self):
        directory = self.pool.directory
        sibling = self.store.append(self.root, (3,), {'status': 'idle'})
        for snapshot, data, binding in (
                (self.root, b'parent', {'schema': 1}),
                (sibling, b'sibling', {'schema': 5, 'parent_snapshot': self.root}),
                (self.tip, b'child', {'schema': 5, 'parent_snapshot': self.root})):
            digest = hashlib.sha256(data).hexdigest()
            (directory / (digest + '.state')).write_bytes(data)
            manifest = dict(binding, payload_sha256=digest)
            (directory / (snapshot + '.json')).write_text(json.dumps({'manifest': manifest}))
        with self.store.read_lease(self.tip):
            self.pool.collect()
            self.assertEqual({path.stem for path in directory.glob('*.json')}, {self.root, self.tip})
            for data in (b'parent', b'child'):
                self.assertTrue((directory / (hashlib.sha256(data).hexdigest() + '.state')).exists())
            self.assertEqual(len(list(directory.glob('*.state'))), 2)
        self.pool.collect()
        self.assertEqual(list(directory.glob('*.json')), [])
        self.assertEqual(list(directory.glob('*.state')), [])

    def test_cumulative_manifest_binds_logical_parent_and_full_base_separately(self):
        grandchild = self.store.append(self.tip, (3,), {'status': 'idle'})
        required = dict(schema=6, snapshot=grandchild, consumed=3, parent_consumed=1,
                        parent_snapshot=self.root, logical_parent_snapshot=self.tip,
                        parent_payload_sha256='a' * 64)
        manifest = dict(required, positions='absolute-zero-based', input_identity='b' * 64,
                        model_contract_id='model', execution_profile_id='profile', logits_count=8,
                        state_bytes=32, payload_sha256='c' * 64)

        def envelope(values):
            return dict(manifest=values, checksum=hashlib.sha256(_json_bytes(values)).hexdigest())

        self.assertEqual(validate_manifest(envelope(manifest), required), manifest)
        for key in ('parent_snapshot', 'parent_payload_sha256', 'logical_parent_snapshot'):
            with self.subTest(key=key), self.assertRaises(IntegrityError):
                validate_manifest(envelope(dict(manifest, **{key: 'd' * 64})), required)
        for invalid in (self.root, grandchild, 'invalid'):
            with self.subTest(invalid=invalid), self.assertRaises(IntegrityError):
                validate_manifest(envelope(dict(manifest, logical_parent_snapshot=invalid)),
                                  dict(required, logical_parent_snapshot=invalid))

    def test_cumulative_suffix_retains_base_after_intermediate_checkpoint_eviction(self):
        grandchild = self.store.append(self.tip, (3,), {'status': 'idle'})
        directory = self.pool.directory
        for snapshot, data, binding in (
                (self.root, b'base', {'schema': 1}),
                (self.tip, b'intermediate', {'schema': 5, 'parent_snapshot': self.root}),
                (grandchild, b'cumulative', {'schema': 6, 'parent_snapshot': self.root,
                                             'logical_parent_snapshot': self.tip})):
            digest = hashlib.sha256(data).hexdigest()
            (directory / (digest + '.state')).write_bytes(data)
            manifest = dict(binding, payload_sha256=digest)
            (directory / (snapshot + '.json')).write_text(json.dumps({'manifest': manifest}))
        with self.store.read_lease(grandchild):
            self.pool.collect()
            self.assertEqual({p.stem for p in directory.glob('*.json')}, {self.root, grandchild})
            self.assertEqual(self.store.get_snapshot(grandchild)['parent'], self.tip)
            self.assertEqual(self.store.get_snapshot(self.tip)['parent'], self.root)
            for data in (b'base', b'cumulative'):
                self.assertTrue((directory / (hashlib.sha256(data).hexdigest() + '.state')).exists())
        self.pool.collect()
        self.assertFalse(list(directory.glob('*.state')))

    def test_failed_reservation_releases_and_does_not_exceed_limit(self):
        with self.assertRaises(RuntimeError):
            with self.pool.reserve(64):
                with self.assertRaises(PoolFull): self.pool.reserve(1)
                raise RuntimeError('write failed')
        self.assertEqual(self.pool.metrics()['reserved_bytes'],0)


if __name__ == '__main__': unittest.main()
