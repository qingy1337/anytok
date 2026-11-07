# Live run dashboard

Run from the repository root:

```bash
python scripts/run_dashboard.py
```

Open <http://127.0.0.1:1337/> on the host, or `http://<host-ip>:1337/` remotely.
The default server binds to `0.0.0.0:1337`. The web UI follows the existing
AnyTok dashboards: locally bundled Manrope, pale background, fine gray borders,
restrained color, and a green progress bar. It needs only Python's standard
library plus the existing Modal CLI for live metadata. The terminal UI and its
Textual dependency have been removed.

The deployed server runs in tmux session `anytok-run-web` using the default
command above. Stopping the dashboard leaves all experiment jobs running.

The experiment selector includes the current 1,000-tokenizer pool, 20k ZeTT
training and the checkpoint evaluation grid. Add future runs to
`plans/run-dashboard.json`; adapters read artifacts without changing jobs.

The uniform-pool training run includes a Training plots panel directly below
its progress card. PNG and PDF browser links are
`/plots/anytok_pool_learning.png` and `/plots/anytok_pool_learning.pdf` on the
same port 1337 server. A local plotting worker checks the saved logs every
minute and atomically refreshes these files when logs change. It uses the
existing Matplotlib plotting environment and starts no cloud jobs. Failed
refreshes preserve the previous plots and display a refresh error in the panel.

- Unique trained and collected counts avoid double-counting cross-profile copies.
- Profiles show platform, finished/assigned counts, workers and estimated credits.
- Select a profile to filter workers and logs; select a worker to inspect its log.
  Clear selection returns to all profiles. Pause following freezes the log view.
- Artifacts refresh every 2 seconds; active runs' Modal metadata/logs every 30
  seconds; billing every 5 minutes. Multiple browser tabs share one poller.
  Failed queries retain the last valid state and show staleness.
- Cache: `modal_results/run_dashboard/remote.json`. Completed experiments use
  their saved results without repeatedly querying their old Modal profiles.
- ETA requires at least three comparable algorithm/pretokenization finishes.
  Overdue workers say `Beyond observed range`; full-run ETA additionally needs
  shard assignments. Estimates assume the configured profile concurrency.
- Credits assume $30/month minus applied credits; billing may lag. Unused
  allowance does not confirm a workspace can start jobs. Disabled profiles
  `general` and `research-everettcc` are never queried.
- The HTTP server exposes only UI assets, `/healthz` and the read-only JSON API
  `/api/state`. It does not serve the repository or accept job-control commands.
  API filters: `?run=tokenizer-pool&profile=qxli2&worker=<container-id>`.
- Timestamped events retain their recorded times; updating a file never makes
  its older, untimestamped events appear current.

For cached/offline inspection: `python scripts/run_dashboard.py --offline`.
`--host`, `--port` and `--config` allow another binding or run registry.

## Register another run

`tokenizer_pool` reads our pool plan, manifests, logs and coordinator state.
`markers` counts unique completion-marker identities across profile directories.
`progress_json` reads an atomic JSON status file. A local example:

```json
{
  "id": "local-training",
  "name": "Local training",
  "adapter": "progress_json",
  "platform": "Local · RTX 6000",
  "status": "results/local/status.json",
  "completed_field": "step",
  "total": 20000,
  "unit": "steps",
  "profiles": [],
  "logs": ["results/local/train.log"]
}
```

The status file can report `stage`, `elapsed_seconds`, `eta_seconds` and
`workers`. Each worker can report `id`, `profile`, `task`, `state`, `started_at`,
`elapsed_seconds` and `eta_seconds`. Generic reported ETAs are displayed as
provided. Configure `eta_seconds_field`, `elapsed_field` or `workers_field`
for other field names. Dotted JSON field paths are supported.

For Modal runs, supply `profiles` and exact `app_names` to enable live container
and log polling. Local runs with no app names perform no Modal queries.
Paths are relative to the repository, or may be absolute. The adapter registry
is intentionally small; it is not a second job scheduler.

Verification:

```bash
python -m unittest discover -s tests -p test_run_dashboard.py -v
```

Covers deduplication, partial writes, retries, conservative ETAs, disabled
profiles, read-only polling, stale metadata, real log timestamps, generic local
workers, HTTP routes, filesystem isolation and rejection of mutation requests.
