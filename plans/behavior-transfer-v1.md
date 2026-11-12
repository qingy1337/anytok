# Find a repeatable information-retention weakness

The user wants a concrete, widespread weakness and a reusable general test
setting. Prioritize behavioral failure across varied English contexts over
additional language panels, aggregate audits or isolated high-loss artifacts.

Test the final RL20k LM/hypernetwork against its own trained canonical inputs
on five task families: numerical copying, identifier copying, personal names,
quoted English sentences and simple semantic fact retrieval. Include natural
held-out English continuations as a reference. Keep the128-position limit.

Freeze64 synthetic cases per family per split, crossed by short/long payload
and near/far retrieval. Discovery and confirmation use different random seeds
and different wording templates. Preserve complete facts and queries; reject
only by lossless encoding and the shared128-position cap, before model scores.
Record every rejection. Do not truncate an answer to make it appear correct.

Select18 native pool members by algorithm/pretokenizer cell and proximity to
64k/120k vocabulary sizes, without using losses. Also include the previous
weak cases0003/0635, explicitly recording the representative subset. Predict
embeddings from each full native vocabulary with the saved RL hypernetwork.

Primary metrics: answer-only BPB and exact answer-token accuracy. Exact
target-token accuracy is equivalent to greedy reproduction of that canonical
token sequence. Generate actual greedy answers for selected failures too.
Separately report the usual next32-token BPB so easy canonical trailing tokens
cannot obscure an answer error. Report natural continuation losses separately
from controlled-task accuracy; natural text has no unique predictable32-token
answer. Check token loss against the model's built-in causal loss.

An operationally large failure requires canonical accuracy>=80%, a transferred
accuracy drop>=20 percentage points and a positive paired answer-BPB gap,
on both discovery and confirmation. Show how many representative settings,
algorithms and pretokenizers support it. This is an exploratory stress test,
not a new estimate of average natural-text performance or a guarantee that
such a failure will be found.

Use two B300 workers on blackpantherx86, with no CPU/memory reservations.
Each worker is bounded to1200seconds; verify at least$6 estimated credits
before launch. Reuse all existing cloud inputs and features. Save raw task
contexts and results in `anytok-rl-all-pool-20k/behavior-transfer-v1/`; local
caller receives score metadata and selected concrete examples. No training.

Launcher: `MODAL_PROFILE=blackpantherx86 modal run --detach scripts/modal_behavior_transfer.py`.
Monitor `plans/run-dashboard.json`; logs and metadata in
`modal_results/behavior_transfer/`. Analyze discovery and confirmation
separately and report unsuccessful hypotheses honestly.

## Completed result

943 shared cases passed validation (640 controlled task cases and303 natural
English-reference passages), with zero encoding/length rejections. All20
tokenizer probes finished. The18 settings selected by configuration alone
span all3 algorithms, all3 pretokenizers and both vocabulary scales.

Exact numerical recall is a repeatable weakness. In free-running32-token
responses, allowing extra words, name capitalization and exact numeric
reformatting, canonical value accuracy is100% in both splits. Transferred
accuracy is0–4.7% on discovery and0–1.6% on confirmation across all18
representative settings. Eight-digit payloads also fail: canonical100%,
transferred0–3.1% on confirmation. The median transferred rate is0%.

The confirmation answer-only gap has median+1.98474 BPB. The usual32-token
continuation gap on the numerical task has median+0.34196 BPB. Names also
show a format-tolerant gap: canonical92.2%, transferred median48.4%, with15/18
settings having a≥20-point drop on both wordings. Other task outcomes and
baseline-calibration failures are explicitly retained in the report.

An exact short case records50778117 and asks for it again. Canonical inputs
return50778117; multiple native settings return577117 or5077811. All20
native encodings preserve the full prefix. Independently rebuilt used input
features match the cached features, with zero surface-form truncations in
this case. Causal per-token losses match the model's built-in loss.

Free-generation scoring was recomputed from stored decoded outputs after
fixing handling of terminal punctuation. Original flags are retained beside
the final recomputed flags; no inference rerun was required for that fix.

This establishes a reusable behavioral stress setting, not the natural-text
prevalence of the problem or the learned mechanism responsible. Report,
plots, all per-token scores, generated responses, exact example audit and
checksums are in `reports/behavior-transfer-20k/`. Raw task contexts and
per-tokenizer artifacts remain in the persistent cloud prefix above.
