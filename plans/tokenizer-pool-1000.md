# AnyTok tokenizer pool

The saved plan is `plans/tokenizer-pool-1000-v1.json`. It contains 1,000
configurations sampled with seed 20261105, the pinned corpus recipe, exact
byte quotas, and hashes of the plan and patched tokenizer wheel.

- Algorithm: BPE, WordPiece, or Unigram, with equal probabilities.
- Vocabulary target: uniform over 32,000 through 128,000 in 1,000-token steps.
- Pretokenization: regex, 256-character line chunks, or no regex, equally likely.
- BPE noise: uniform integer millionths from 0 to 750,000; random selection
  from all eligible pairs, without permanent top-pair exclusion.
- Domain mixture: Dirichlet(1) across all nine domains, with no additional
  percentage bounds. Within-domain source proportions follow the corpus recipe.
- Corpus size: 200,000,000 raw UTF-8 bytes per tokenizer. UTF-8 boundaries can
  leave up to three bytes unused per source; manifests record actual usage.

The actual sample contains 332 BPE, 328 WordPiece, and 340 Unigram models.
There are 352 regex, 322 line-chunk, and 326 no-regex configurations. These
counts arise from independent sampling, rather than enforced balancing.

WordPiece uses a BPE-trained vocabulary and greedy WordPiece encoding, as in
our existing generator. Unigram and BPE retain their existing trainer defaults.
The runtime wheel additionally applies
`patches/tokenizers-0.22.2-unigram-literal-unk.patch` and
`patches/tokenizers-0.22.2-unigram-stack-drop.patch`. This excludes the
trainer's NaN-scored temporary `<UNK>` entry from surface matching: literal
`<UNK>` in ordinary corpus text otherwise causes an upstream NaN panic. It
preserves corpus text and sampled pretokenization, with no new length cap.
The second fix clears backtrace ownership links before freeing a lattice,
preventing recursive stack overflow on long paths. Its regression fails on the
original implementation; all 22 Unigram Rust unit tests pass with both fixes.
Each manifest records the actual wheel and worker hashes; the first five
successful pilot models used the original wheel. Failed jobs are isolated in
subprocesses so a Rust panic does not repeatedly crash and restart a Modal runner.

The seed determines sampled configurations and BPE merge randomness; it is
not an assertion of bitwise determinism for all upstream trainers.

## Run and resume

Use the `group-cron` workspace, where the source corpus already resides.
`general` and `research-everettcc` are disabled and must never be used.

```bash
python scripts/modal_credits.py group-cron
python scripts/tokenizer_pool.py
MODAL_PROFILE=group-cron modal run --detach scripts/modal_generate_tokenizer_pool.py --stage all
```

The pilot consists of 12 members of the saved pool: all nine algorithm and
pretokenization combinations, plus high-noise BPE cases for each pretokenizer.
They count toward the final 1,000. The `all` stage runs the pilot and, if it succeeds and fits the budget estimate, continues to full generation. Re-running skips completed members; a reused
file's configuration, provenance and checksum are checked before it is returned.
Failed configurations are preserved and reported, rather than replaced by
fresh samples.

Workers have no explicit CPU, memory or GPU reservation. Each uses four Rayon
threads, and up to 100 separate containers can train concurrently. The full run
uses waves of at most 100 tasks, rechecks billing, and estimates remaining cost
from successful pilot/group measurements. It leaves a $1 margin below the
assumed $30 monthly credit allowance. Estimates include process CPU time and
peak RSS times elapsed time, plus a 50% margin and $0.001 per pending task.
Billing can lag, and this guard is not a hard spending cap.

## Persistent output

Volume: `anytok-tokenizer-pool` in `group-cron`.
Pool directory: `pools/1f5d0be379e32101/`.

- `plan.json`: all 1,000 configurations and the full corpus recipe.
- `source/`: generation scripts.
- `artifacts/`: patched tokenizer wheel (uploaded by the full-stage launcher).
- `tokenizers/0000/` through `tokenizers/0999/`: `tokenizer.json` and
  `manifest.json` for each successfully completed model.

Manifests contain configuration, corpus document hashes and byte counts,
actual vocabulary size, round-trip validation, tokenizer checksum and size,
training time, CPU time, peak RSS, and an approximate process cost.

Local progress is in `modal_results/tokenizer_pool/status.json` and
`events.jsonl`. Completed local manifests are mirrored there without downloading
tokenizer files. The CLI credit helper excludes disabled profiles and can emit
JSON for monitoring.

## Verification

```bash
PYTHONPATH=/tmp/anytok-pool-fixed-v2 RAYON_NUM_THREADS=4 \
  python -m unittest discover -s tests -p test_tokenizer_pool.py -v
```

The integration test uses an isolated installation of the patched wheel at the
shown path. It trains all nine combinations on a toy corpus, checks target
vocabulary size and exact decoding, and exercises 75% noise for BPE. Separate
plan checks reproduce the saved sample and verify every quota against available
source data. Every production model also validates decoding on snippets from
all consumed source buckets and a multilingual synthetic example.

## Multiple profiles

The user authorized splitting across enabled profiles and using up to 100
parallel containers per profile. `scripts/run_tokenizer_pool_farm.py` coordinates
this run under tmux, records assignments, checks billing and workspace app
creation, imports the corpus, trains disjoint shards, and gathers successful
models into the original group-cron output Volume. Profile allocation and budget
calculations have dedicated tests. A failed shard is reported and its successful
models are collected; failed configurations are never silently replaced.

`general` and `research-everettcc` are permanently excluded. `aurora` currently
rejects app creation with a spend-limit error, despite showing no monthly usage,
so this run excludes it too. Other zero-usage accounts are checked for app
creation access before selection; $30 minus monthly usage is an estimate of
remaining allowance, not proof that a workspace is usable.

The first two shards cover 348 non-pilot regex configurations, split evenly
between pandas and qingy1337. They run independently of the outstanding
unrestricted diagnostics. Queueing by family does not change the saved pool;
training the LM should wait for the full pool rather than using the fast subset.
`scripts/collect_tokenizer_pool.py` gathers those initial shards automatically.
The farm queues every remaining index in `farm-queue.json` and dispatches
measured families immediately. A slow diagnostic gates only its own family;
its first valid saved manifest can supply a cost estimate before the entire
shard is collected. Profiles with running shards are excluded from new
assignments, and saved `farm-round-*.json` assignments are disjoint. A process
lock prevents two new coordinators running concurrently. The original BPE
diagnostics are adopted without restarting either worker; their old coordinator
is paused locally and retired once its diagnostic launcher exits.

The normal coordinator command is:

```bash
python scripts/run_tokenizer_pool_farm.py
```

It checks verified profile budgets, preserves the $1 billing margin, and trains
up to 100 models per profile at once. All 640 remaining configurations were
queued on November 5: 537 measured configurations can dispatch immediately;
103 BPE/no-regex configurations become eligible as soon as either existing
BPE diagnostic supplies a successful measurement. The original two running
indices remain reserved and are never duplicated. The source workspace
`group-cron` always uses its original corpus; corpus import runs only for other
workspaces. The coordinator gathers successful shards directly into group-cron
and publishes the verified full index itself; it needs no separate finalizer.
Do not restart a live coordinator or its launchers: detached remote work may
continue and requires explicit reconciliation before another launch. Progress is in
`modal_results/tokenizer_pool/farm-status.json` and per-profile directories.

`scripts/modal_tokenizer_pool_transfer.py` prepares just the source prefixes
needed by the maximum quota for each of the 23 buckets: 428,481,413 compressed
bytes total. The last document is kept intact, so every tokenizer sees the same
text and document boundaries as it would from the original source. Corpus
files and tokenizer models stream directly between remote Modal workers and
Volume APIs, using ephemeral Secrets for source-workspace authentication.
Only configurations, manifests and progress metadata pass through the local
caller. Transfers verify file sizes and SHA-256 hashes; model files are copied
before their completion manifests. The corpus is never published externally.

Future tasks use a six-hour execution timeout, replacing the initial pilot's
one-hour safety timeout. This changes job scheduling, not tokenizer inputs or
pretokenization. CPU and memory reservations remain unspecified.

Volume listing streams are materialized before nested file reads. Manifest
reads run in parallel, transient connection failures are retried, and transfer
temporary filenames are unique. The collector resumes from its saved state.
These changes recovered a connection failure while collecting qingy1337; all
348 initial shard models were subsequently gathered successfully.
