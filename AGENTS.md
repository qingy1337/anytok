# Long-running experiments

Keep long-running training, evaluation and tokenizer-generation jobs visible in
`plans/run-dashboard.json`. Reuse the adapters in `monitoring/`; the web
entry point is `python scripts/run_dashboard.py` (default `0.0.0.0:1337`). Record platform, profiles,
completed/total progress and log locations. Prefer atomic JSON status updates
and timestamped JSONL events. Do not add job-control actions to the dashboard.

`general` and `research-everettcc` are permanently disabled Modal profiles;
never use them. For this experiment the user requested no explicit CPU or
memory reservations. Modal credits are estimated from the $30 monthly
allowance minus applied credits; zero monthly usage does not guarantee that a
workspace is currently allowed to run jobs.

Keep corpora, tokenizer files and model checkpoints in persistent cloud
storage. Transfer large artifacts directly between cloud services; only
metadata should pass through the local caller.

# Experiment fidelity

Preserve the agreed evaluation data, complete tokenizer coverage and scoring
protocol. Do not substitute subsets, proxies or cheaper experimental designs
to save time or compute. Material changes require the user's explicit approval.
