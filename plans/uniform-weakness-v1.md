# Find useful tokenizers with >=0.1 BPB excess loss

Run `MODAL_PROFILE=trenton modal run --detach scripts/modal_find_uniform_weakness.py`.
Use `--collect-only` after a caller disconnects to reattach to its saved function
call. The existing uniform-pool checkpoint at12k is already staged in trenton's
`anytok-pool-generalization` Volume. New artifacts are under `weakness-v1/`;
only progress, checksums and score metadata return to the local caller.

Immutable selection: `uniform-weakness-v1.json`, 26 native tokenizer files and
nine held-out domains. Include the previous eleven48k BPEs,64k base,150k
regex/line256 and domain-focused BPEs, WordPiece and Unigram, and pinned native
Qwen2.5-Coder, DeepSeek-Coder and StarCoder2 tokenizer files. Existing cloud
artifacts move directly from group-cron/csproject to trenton, and published
tokenizers download directly from Hugging Face to a Modal CPU worker. No new
tokenizer training. H200 eval; no explicit CPU or memory reservations.

For each domain, rank documents by SHA256(domain/id), choose the first48 valid
documents for screening and the next96 (minimum64) for independent confirmation.
Validity is based on reversible text encoding, before any losses. Every target
and canonical control gets the same raw prefix, same32 canonical suffix IDs and
suffix bytes, with BOS+prefix+suffix capped at128 positions. Persist all IDs,
prefixes, rejected encoding cases and per-document confirmation losses remotely.
Keep native vocab IDs, decoder/pretokenizer and Unigram prior scores. Special
tokens cannot appear in scored prefixes. Unsupported adapters or encoding
failures are recorded as such and never counted as model weakness.

Screen every candidate/domain pair, confirm those with gap>=0.08 on unused
documents, and report all tested pairs. A verified candidate needs confirmation
gap>=0.1 against the12k model's canonical control, plus a real use case. The
initial quantitative utility gate is >=10% fewer tokens than canonical on
matched held-out excerpts; report native published-tokenizer interoperability
separately. Token compression alone is not an end-to-end speed benchmark.
Use paired-document bootstrap CIs. A lower CI bound>=0.1 is stronger evidence
than only a point estimate crossing the threshold. Do not claim uniform is weak
on every tokenizer, or that a selected result predicts performance after20k.
Separate150k/published vocabulary sizes outside32k..128k training support.

Published native tokenizer sources:
- https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B
- https://huggingface.co/deepseek-ai/deepseek-coder-1.3b-base
- https://huggingface.co/bigcode/starcoder2-3b

This is a search for practical weaknesses, not evidence yet that RL fixes them.
Final20k replication and a comparison against adaptive sampling remain separate
experiments; report12k results with their checkpoint label.

## Rerun the three reported cases on the newest saved checkpoint

Run `MODAL_PROFILE=trenton modal run --detach scripts/modal_find_uniform_weakness.py --rerun-latest`. The launcher freezes the newest committed checkpoint manifest from the current training/restart profiles and copies only LM/hypernetwork weights directly into the evaluation Volume. It evaluates Chinese35 on Chinese, Japanese35 on Japanese, and Base64k on medical English on their exact original 96-document confirmation contexts. No new documents or cuts are selected. Original 12k files remain unchanged.

Results and checkpoint weights are separate under `weakness-v1/reruns/step-NNNNNN/` on trenton's `anytok-pool-generalization` Volume. Local metadata is under `modal_results/uniform_weakness/reruns/`; the dashboard uses `rerun-status.json`. For interrupted collection, add `--collect-only` to the same rerun command to attach to the saved function call without starting another GPU evaluation.

The report separates changes in canonical BPB, transferred BPB, and their gap. A paired bootstrap across identical document IDs estimates uncertainty in the gap change. Negative gap change means transfer improved relative to that checkpoint's canonical control.
