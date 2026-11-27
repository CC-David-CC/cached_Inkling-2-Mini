"""Compare single-process and restarted output-head updates through the training CLI."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from inkling_train_head import equal_state, torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runtime', type=Path)
    parser.add_argument('--optimizer', choices=('adamw', 'muon'), required=True)
    args = parser.parse_args()
    trainer = Path(__file__).with_name('inkling_train_head.py')
    command = [sys.executable, str(trainer), str(args.runtime.resolve())]

    with tempfile.TemporaryDirectory() as workspace:
        root = Path(workspace)
        original, resumed, direct, chained = [root / name for name in ('original', 'resumed', 'direct', 'chained')]

        def train(output, *options):
            subprocess.run([*command, str(output), *map(str, options)], check=True)
            return json.loads((output / 'receipt.json').read_text())

        baseline = train(original, '--optimizer', args.optimizer)
        checkpoint = original / 'step-2.pt'
        third = train(resumed, '--resume', checkpoint)
        assert baseline['losses'][-1] == third['losses'][0]
        assert (original / 'adapter.iklora').read_bytes() == (resumed / 'adapter.iklora').read_bytes()

        uninterrupted = train(direct, '--resume', checkpoint, '--additional-steps', 2)
        fourth = train(chained, '--resume', resumed / 'step-3.pt')
        assert uninterrupted['losses'] == third['losses'] + fourth['losses']
        assert (direct / 'adapter.iklora').read_bytes() == (chained / 'adapter.iklora').read_bytes()
        states = [torch.load(path / 'step-4.pt', map_location='cpu', weights_only=True)
                  for path in (direct, chained)]
        assert states[0]['metadata']['steps'] == 4 and equal_state(*states)
        print(f'{args.optimizer}: CLI restart matched losses, adapter bytes and complete step-four state')


if __name__ == '__main__':
    main()
