"""Compare fresh-process and uninterrupted AdamW or Muon output-head updates."""
import argparse
from pathlib import Path
import subprocess
import sys
import tempfile

from inkling_train_head import capture, equal_state, initialize, save_training_checkpoint, torch, train_step


def run_stage(runtime, optimizer, directory, resume):
    torch.use_deterministic_algorithms(True)
    features, logits, targets, cap, profile = capture(runtime)
    a, b, opt = initialize(logits.shape[1], optimizer)

    def state():
        return dict(profile=profile, a=a.detach(), b=b.detach(), optimizer=opt.state_dict())

    def step():
        return train_step((a, b, opt), features, logits, targets, cap)

    if resume:
        saved = torch.load(directory / 'step2.pt', weights_only=True)
        assert saved['profile'] == profile
        with torch.no_grad():
            a.copy_(saved['a'])
            b.copy_(saved['b'])
        opt.load_state_dict(saved['optimizer'])
    else:
        step()
        step()
        save_training_checkpoint(directory / 'step2.pt', state())

    result = dict(loss=step(), **state())
    if resume:
        reference = torch.load(directory / 'reference.pt', weights_only=True)
        assert equal_state(reference, result), 'Fresh-process update differs'
        print(f'{optimizer}: loss, factors and optimizer matched; third-step loss {result["loss"]}')
    else:
        save_training_checkpoint(directory / 'reference.pt', result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runtime', type=Path)
    parser.add_argument('--optimizer', choices=('adamw', 'muon'), required=True)
    parser.add_argument('--stage', choices=('reference', 'resume'), help=argparse.SUPPRESS)
    parser.add_argument('--workspace', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.stage:
        run_stage(args.runtime.resolve(), args.optimizer, args.workspace, args.stage == 'resume')
        return
    with tempfile.TemporaryDirectory() as workspace:
        for stage in ('reference', 'resume'):
            subprocess.run([sys.executable, __file__, str(args.runtime), '--optimizer', args.optimizer,
                            '--workspace', workspace, '--stage', stage], check=True)


if __name__ == '__main__':
    main()
