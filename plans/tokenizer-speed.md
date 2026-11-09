# Tokenizer trainer status

The v4–v8 speed experiments are withdrawn. The requirement is the **same learned
tokenizer from the same 200 MB input**, trained faster: vocabulary entries,
token IDs, Unigram scores, merge rules, runtime settings and segmentation must
remain unchanged. Reversible encode/decode is a separate property and does not
meet this requirement.

Two completed full-size Unigram pairs retained the same 80,000 token strings,
but their token IDs/order and scores differed. The cause has not been isolated
against baseline-vs-baseline nondeterminism. Neither timing result qualifies as
an exact-output speedup. No optimization from these experiments is retained in
the active trainer; even changes not shown to alter outputs were withdrawn
until whole-model equivalence can be established.

## Rollback

- Removed all five experimental native speed patches and their benchmark,
  recovery and automatic observer launchers. Git history retains their source.
- Stopped Modal apps `ap-merQ8USN2jSIWD154iPiZn` and
  `ap-LZRRn3li2xS8ABFE3tzkzW`, and the `anytok-speed-observer` tmux session.
- `scripts/modal_generate_fast_tokenizers.py` now uses the original
  `build/pool-v2/tokenizers-0.22.2-cp39-abi3-manylinux_2_34_x86_64.whl`.
  The historical script name is retained. Its runtime identity changes with the
  wheel/worker/launcher, so old optimized artifacts are not reused.
- Removed the wrapper's rewrite of WordPiece's runtime length cutoff to
  `sys.maxsize`. The original trainer's settings and serialized file remain intact.
- The builder accepts only `--candidate baseline`, applying the three original
  pool patches: all-candidate-noise, unigram-literal-unk and unigram-stack-drop,
  on upstream `f383101a26663708484cac0727792aad74f78234` (tokenizers 0.22.2).
  Those patches predate this speed experiment and define the reference pool.
- Historical report analysis now requires identical serialized tokenizer hashes
  before it can qualify a speedup. This is a conservative gate: harmless JSON
  ordering differences also fail it. No global determinism claim is made.

Completed artifacts, inputs and logs remain in persistent Modal Volumes:
`gildmember/anytok-fast-tokenizers` and `gildmember/anytok-pool-20k`.
They are experimental evidence, not approved replacement tokenizers. Dashboard
entries retain the stopped runs and log locations. No new training is launched
by this rollback.

## Existing trainers investigated

See [tokenizer-trainer-survey.md](tokenizer-trainer-survey.md) for primary-source
links, actual training benchmarks and compatibility limits. Alternative
implementations are candidates, not proven exact-output replacements.
