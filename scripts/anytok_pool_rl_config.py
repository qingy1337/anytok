"""Names and bounded policy settings for the adversarial pool experiment."""
import os

EXPERIMENT = os.environ.get('ANYTOK_RL_EXPERIMENT', 'original')
if EXPERIMENT not in ['original', 'regret']:
    raise ValueError('Unknown joint RL experiment')
PROFILES = ('trenton', 'github', 'group-pytorch', 'grufaller', 'mix345-net',
            'neural-net-1411', 'queues', 'stars', 'transformer',
            'blackpantherx86', 'anushavenkatachalam', 'cron', 'pandas', 'code-group2-evcc')
assert not set(PROFILES) & {'general', 'research-everettcc'}
VOLUME_NAME = 'anytok-pool-rl-20k'
APP_NAME = 'anytok-pool-settings-rl-20k-b300'
STATE_DIR = 'modal_results/anytok_pool_settings_rl_20k'
RUN_PREFIX = 'settings-run'
SMOKE_PREFIX = 'settings-smoke'
FINAL_STEP = 20000
POLICY = {'kind':'shared_settings_mlp', 'hidden_sizes':[64,64], 'learning_rate': 0.001,
          'entropy_weight': 0.01, 'baseline_decay': 0.99, 'seed': 43}
TRAINING_SOURCE = 'runtime/anytok_pool_settings_rl_20k'
REWARD_OBJECTIVE = 'raw_bpb'
MIN_PROFILE_BALANCE = 10.
if EXPERIMENT == 'regret':
    # Shared persistent input/ is reused; all new checkpoints have a new prefix.
    PROFILES = ('queues', 'stars', 'transformer', 'aurora', 'anushavenkatachalam',
                'cron', 'pandas', 'code-group2-evcc', 'mix345-net', 'github',
                'grufaller', 'group-pytorch', 'li004330', 'transformers-impl',
                'qingy1337', 'remote-worker', 'gildmember', 'data-pipelines',
                'csproject', 'assignment2-testcases', 'trenton', 'definitive', 'deployment')
    APP_NAME = 'anytok-pool-regret-rl-20k-b300'
    STATE_DIR = 'modal_results/anytok_pool_regret_rl_20k'
    RUN_PREFIX = 'regret-run'
    SMOKE_PREFIX = 'regret-smoke'
    TRAINING_SOURCE = 'runtime/anytok_pool_regret_rl_20k'
    REWARD_OBJECTIVE = 'paired_gap_bpb'
    MIN_PROFILE_BALANCE = 5.
    POLICY = {**POLICY, 'entropy_weight': 0.}
assert not set(PROFILES) & {'general', 'research-everettcc'}
