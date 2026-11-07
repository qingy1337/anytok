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
it does not compress the LR schedule to 20k. The two saved ZeTT tokenizers used
for periodic sanity evaluation remain unchanged. They are not the full
FineWeb-Edu transfer panel; that separate evaluation follows training.

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
only metadata passes through the local caller. A private model repository,
qingy2024/anytok-pool-smollm2-20k, bridges full restarts and archives all immutable
500-step checkpoints. Source checkpoints are compacted only after the next
workspace verifies the HF handoff. The latest pointer is published last. Non-2k handoff restarts use `restarts/`
so they cannot overwrite immutable weights-only saves under `checkpoints/`.

Progress: modal_results/anytok_pool_20k/status.json and controller.log, plus
per-profile logs. Registered as anytok-pool-training in plans/run-dashboard.json.
The dashboard remains read-only. ETA uses median recent update time after at
least 20 timings and excludes checkpoint, evaluation and transfer overhead.
If the controller fails, reconcile saved remote apps/checkpoints before another
launch; do not restart blindly and duplicate GPU work.

Verification: python -m unittest discover -s tests -p test_anytok_pool_sampling.py -v.
