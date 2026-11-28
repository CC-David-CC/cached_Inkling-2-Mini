"""Opt-in single-lease Inkling backend retaining target and drafter context together."""
from contextlib import contextmanager
from io import BytesIO
import struct


from .inkling_backend import InklingBackend
from .store import IntegrityError


class DFlashBackend(InklingBackend):
    header = struct.Struct('<8sQQ')
    magic = b'IKDFROW1'

    def __init__(self, profile):
        super().__init__(profile)
        try:
            # Load the native target first: PyTorch import order affects its CUDA logits.
            import torch
            from .inkling_dflash_loader import load_proposer

            self.torch = torch
            stage_target = profile['execution'].get('dflash2_target_placement', 'host') == 'host'
            self.propose = load_proposer(*profile['dflash2_assets'], device='cuda:0', stage_target=stage_target)
            self.rows = torch.empty((1, self.context_capacity, 9216), dtype=torch.bfloat16, device='cuda:0')
        except BaseException:
            self.close()
            raise

    def consume(self, lease, ids):
        torch = self.torch
        start = self.position(lease) + 1
        taps = self.consume_dflash_taps(lease, ids)
        rows = torch.frombuffer(taps, dtype=torch.float32).reshape(4, len(ids), 2304)
        self.rows[0, start:start + len(ids)].view(-1, 4, 2304).copy_(rows.transpose(0, 1))

    @contextmanager
    def generate_request(self, lease, max_steps):
        def generate():
            remaining = max_steps
            while remaining:
                anchor = self.select_greedy(lease)
                # A successor can only be committed after the anchor.
                path = []
                if remaining > 1 and not self.is_eog(anchor):
                    path = self.propose(self.rows[:, :self.position(lease) + 1], anchor)
                self.stats['draft_nodes'] = self.stats.get('draft_nodes', 0) + len(path)
                for index, token in enumerate([anchor, *path]):
                    if index and token != self.select_greedy(lease):
                        break
                    self.consume(lease, [token])
                    if index:
                        self.stats['drafts_accepted'] = self.stats.get('drafts_accepted', 0) + 1
                    remaining -= 1
                    yield token
                    if remaining == 0 or self.is_eog(token):
                        return

        request = generate()
        try:
            yield request
        finally:
            request.close()

    def work_counters(self):
        counters = super().work_counters()
        for key in ('draft_nodes', 'drafts_accepted'):
            counters[key] = self.stats.get(key, 0)
        return counters

    def export(self, lease):
        torch = self.torch
        size = self.lib.ctt_state_size(lease)
        if size == 0:
            raise IntegrityError('Missing DFlash2 target state')
        count = self.position(lease) + 1
        offset = self.header.size + size
        with BytesIO() as stream:
            stream.seek(offset + count * 9216 * 2 - 1)
            stream.write(b'\0')
            with stream.getbuffer() as view:
                self.header.pack_into(view, 0, self.magic, size, count)
                with view[self.header.size:offset] as target:
                    self._export_into(lease, target)
                    self._pin_export(lease, target)
                for chunk in self.rows[0, :count].split(512):
                    data = memoryview(chunk.cpu().view(torch.uint16).numpy()).cast('B')
                    view[offset:offset + len(data)] = data
                    offset += len(data)
            return stream.getvalue()

    def restore(self, lease, blob, *, history=None, frontier=None):
        torch = self.torch
        if len(blob) < self.header.size or not history:
            raise IntegrityError('Missing DFlash2 checkpoint')
        magic, size, count = self.header.unpack_from(blob)
        offset = self.header.size + size
        if (magic != self.magic or size == 0 or count != len(history) or count > self.context_capacity
                or len(blob) != offset + count * 9216 * 2):
            raise IntegrityError('Invalid DFlash2 checkpoint layout')
        super().restore(lease, blob[self.header.size:offset], history=history, frontier=frontier)
        view = memoryview(blob)[offset:]
        for start in range(0, count, 512):
            end = min(start + 512, count)
            data = bytearray(view[start * 18432:end * 18432])
            self.rows[0, start:end].copy_(torch.frombuffer(data, dtype=torch.bfloat16).view(-1, 9216))
        return blob
