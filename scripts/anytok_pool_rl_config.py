"""Names and bounded policy settings for the adversarial pool experiment."""
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
