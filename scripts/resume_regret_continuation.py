"""Run the regret trainer for the step-8,000 continuation, allowing only the input-location change.

Adapted from scripts/resume_regret_tillicum.py in qingy1337/anytok (zett-modal-20k, fdbd3ea), which
guarded the source run's Modal-to-Tillicum handoff at step 5,288. Here the run moves between
Tillicum accounts at step 8,000: the trainer, inputs and every training setting stay the same, and
only the two input paths (model, warmup_checkpoint) may differ. With --check, the protected
checkpoint list may also shrink to the steps a short check run is allowed to protect.

    python resume_regret_continuation.py --code <staged tree> [--check] -- <trainer arguments>
"""
import argparse
import importlib.util
import os
from pathlib import Path
import re
import sys

STEP = 8000
TRAINER_SHA256 = 'c411e683c2251047ad497512aea1ca948ffb0bece398ce5fd21992782e18ce8b'
REGRET_REWARD = 'paired_pre_model_update_transferred_bpb_minus_current_canonical_bpb'
LOCATIONS = {'model', 'warmup_checkpoint'}


def redact(text):
    user = os.environ.get('USER', '')
    return re.sub(r'(?<=/)' + re.escape(user) + r'(?![\w.-])', '<user>', text) if user else text


def handoff(saved, requested, checkpoint_step, check=False):
    """Return the event recording an allowed handoff, or None if nothing changed."""
    if saved == requested:
        return None
    if checkpoint_step != STEP:
        raise ValueError('Settings may differ only when resuming the step-8,000 restart')
    for config in (saved, requested):
        identity = (config.get('reward_objective'), config.get('policy_reward'),
                    config.get('policy', {}).get('entropy_weight'), config.get('training_script_sha256'),
                    config.get('memory_execution'))
        if identity != ('regret', REGRET_REWARD, 0.0, TRAINER_SHA256, 'optimized'):
            raise ValueError('Only the entropy-zero regret run with its Tillicum trainer may continue')
    changed = {key for key in set(saved) | set(requested) if saved.get(key) != requested.get(key)}
    allowed = LOCATIONS | ({'protected_steps'} if check else set())
    if not changed <= allowed:
        raise ValueError('Material training settings changed: ' + ', '.join(sorted(changed - allowed)))
    for key in changed & LOCATIONS:
        if Path(saved[key]).name != Path(requested[key]).name:
            raise ValueError('Input changed beyond its location: ' + key)
    if 'protected_steps' in changed:
        if not set(requested.get('protected_steps', [])) <= set(saved.get('protected_steps', [])):
            raise ValueError('A check run may only protect a subset of the saved milestones')
    return dict(stage='execution_migration', step=checkpoint_step,
                reason='Continued on another Tillicum account: same trainer, inputs and settings; '
                       'only the input locations differ',
                changed=sorted(changed),
                locations={key: redact(requested[key]) for key in sorted(changed & LOCATIONS)},
                check_run=check)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--code', type=Path, required=True)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('trainer_args', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    sys.path[:0] = [str(args.code / 'scripts'), str(args.code)]
    source = args.code / 'scripts/train_joint_smollm_input.py'
    spec = importlib.util.spec_from_file_location('regret_continuation_trainer', source)
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    original = trainer.check_resume_settings

    def checked(saved, requested, allow_microbatch, checkpoint_step, allow_min_chars=False):
        if saved == requested:
            return original(saved, requested, allow_microbatch, checkpoint_step, allow_min_chars)
        return handoff(saved, requested, checkpoint_step, args.check)

    trainer.check_resume_settings = checked
    argv = args.trainer_args[1:] if args.trainer_args[:1] == ['--'] else args.trainer_args
    if '--resume' not in argv or argv[argv.index('--reward-objective') + 1] != 'regret':
        raise ValueError('Explicit regret resume required')
    sys.argv = [str(source), *argv]
    trainer.main()


if __name__ == '__main__':
    main()
