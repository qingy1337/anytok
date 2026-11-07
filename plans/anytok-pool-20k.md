# Uniform AnyTok pool training, 20k

Run `python scripts/run_anytok_pool_20k.py` under tmux. The existing ZeTT run
is preserved. This starts from the same pinned SmolLM2-1.7B initialization and
10k-step identity-warmed hypernetwork; it does not continue the old 20k model.

Each optimizer update draws one of all 1,000 saved tokenizers independently,
with probability 1/1,000 each, with replacement. All 128 examples in that
update use the selected tokenizer for their prefixes. Tokenizer sampling uses
the checkpointed NumPy RNG. The model consumes at most 128 total positions:
canonical BOS + transferred prefix + canonical continuation. Only continuation
positions contribute LM loss. Canonical input/reference embeddings and output
head stay frozen; the LM backbone and input hypernetwork train jointly.

The source recipe, text stream, minimum snippet length, split logic, batch size,
loss and optimizer settings, seed 42, 10k LR ramp, and original 190k cosine
schedule match the previous 20k experiment. This run stops at update 20,000;
it does not compress the LR schedule to 20k. The two ZeTT tokenizers used for periodic sanity evaluation are fixed within
this run. They were regenerated at initialization and differ from the old
run's two sanity tokenizers (including scored continuation byte counts), so
those old/new sanity curves are not paired comparisons. The full common
FineWeb-Edu transfer panel remains a separate evaluation.

Pool: group-cron / anytok-tokenizer-pool / pools/1f5d0be379e32101/.
The pool index and per-model hashes are verified. Native tokenizer IDs, models,
pretokenizers and decoders are preserved. Unigram retains learned vocabulary
scores; BPE and WordPiece use zero prior features, matching the existing BPE
transfer evaluator. Surface forms use the existing 10-piece limit and byte
fallback mapping. Forms are cached on CPU after first use; parsed tokenizer
objects have a bounded 16-entry cache.

Trainer snapshot: runtime/anytok_pool_20k/. Launcher:
scripts/modal_anytok_pool_20k.py. These are separate pinned snapshots so the
previous experiment stays reproducible. GPU: B300 only, no CPU/memory request.
Enabled profile order is scripts/anytok_pool_config.py. Permanently disabled
profiles general and research-everettcc are excluded. The controller checks
credit usage and workspace availability, leaves a $1.50 forecast margin,
uses a $7.50/hour GPU-plus-overhead estimate, and caps segments at 3.5 hours.
This is an estimate; Modal billing can lag.

Checkpoint weights every 500 updates, full restarts every 2k and latest segment
boundary, preserving optimizer, RNG and corpus state. Same v2 Volume name
anytok-pool-20k in each selected profile; input/ contains the pinned old inputs
and the complete saved pool, run/ contains the new experiment. Inputs copy
cloud-to-cloud from projects' original input Volume and group-cron's pool;
only metadata passes through the local caller. The private model repository qingy2024/anytok-pool-smollm2-20k is used for
checkpoint handoffs and archival while HF storage is available. If HF reports
its private-storage quota, the default controller switches to direct, hashed
Modal-to-Modal checkpoint copies. It does not change repository visibility or
route weights through the local machine. Every grid checkpoint remains on its
source Modal Volume. At 20k, the complete 40-checkpoint grid is gathered onto
the final profile's Volume and independently audited. Source checkpoints are compacted only after the next
workspace verifies the HF handoff. The latest pointer is published last. Non-2k handoff restarts use `restarts/`
so they cannot overwrite immutable weights-only saves under `checkpoints/`.

Progress: modal_results/anytok_pool_20k/status.json and controller.log, plus
per-profile logs. Registered as anytok-pool-training in plans/run-dashboard.json.
The dashboard remains read-only. ETA uses median recent update time after at
least 20 timings and excludes checkpoint, evaluation and transfer overhead.
The normal controller command also recovers an interrupted run: it refuses
to overlap active GPU apps, locates the newest full restart in the saved
source/current profiles, audits it, and resumes using an enabled profile with
credit headroom. Source and destination hashes are checked before source
compaction. Invalid checkpoints and non-quota upload errors still stop safely.
No special recovery flag or training restart is required.

On November 7, HF private-storage quota stopped publication after step 8,277.
The complete checkpoint remained on data-pipelines / anytok-pool-20k, and
the new default recovery path resumes through direct Volume transfer.
Recovery tests: python -m unittest discover -s tests -p test_anytok_pool_recovery.py -v.

Verification: python -m unittest discover -s tests -p test_anytok_pool_sampling.py -v.
