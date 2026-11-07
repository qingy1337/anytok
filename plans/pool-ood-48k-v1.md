# Existing held-out 48k panel on uniform-pool training

User requested reuse of the eleven saved specialty tokenizers instead of generating
a new matched panel. The fresh-tokenizer Modal app was stopped on 2026-11-07;
completed files remain persistent. Main 20k training is unaffected.

Run `MODAL_PROFILE=trenton modal run --detach scripts/modal_eval_pool_ood.py`.
The immutable inputs and prepared prefixes are in `pool-ood-48k-v1.json`.
Use 2k, 6k and 12k checkpoints from csproject, data-pipelines and deployment,
respectively. Weights were copied directly between Modal workspaces; never relayed
through the local machine. The destination is the existing trenton Volume
`anytok-pool-generalization`; new output lives separately under `ood48-v1/`.
H200, no explicit CPU or memory reservations. Local logs and metadata are in
`modal_results/pool_ood_48k/`; progress appears in the run dashboard.

Base, Code50, Papers25, Math50, QA25, Books25, Medical40, Legal40, Spanish35,
Chinese35 and Japanese35: all 48k byte-level BPE with line256 pretokenization.
All eleven saved tokenizer SHA256s are absent from the completed 1000-model pool.
Their tokenizer-training source corpus is not asserted to be disjoint from that
used to create the pool. This tests unseen tokenizer files, not every algorithm.

Reuse the prior length-controlled FineWeb-Edu panel: 256 fixed document IDs,
32 canonical continuation tokens per document, 8192 tokens and 38534 UTF-8 bytes.
The same raw prefix is used by canonical and all eleven targets, trimmed only
if necessary so BOS + prefix + suffix is at most 128 positions. This is a cap,
not a fixed sequence length; on this panel the maximum is 82 positions. This
panel is distinct from the older full-context/128-token-suffix evaluation.

Only target-prefix embeddings come from the hypernetwork. BOS and suffix input
embeddings, suffix token IDs and the output head remain canonical. The full
checkpoint LM backbone is loaded. Both frozen canonical embedding tables are
verified unchanged. Scores contain per-document suffix NLL, with explicit causal
shift and ignored prefix/padding labels. Compare transfer BPB and its gap from
each checkpoint's own canonical score. The paired document bootstrap describes
uncertainty for this fixed panel; it does not establish generalization to all
tokenizer families. No new tokenizer training is required.
