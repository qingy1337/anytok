# Correct canonical token boundaries in the natural-text transfer map

The initial v1 evaluation completed all38 settings on2,032 natural passages.
An audit of the saved per-token losses showed a large first-token advantage
for transferred prefixes, offsetting positive losses later in the continuation.
A CPU encoding audit found that only579/2,032 cuts preserve ordinary joint
canonical tokenization. At the other cuts, separately encoding the prefix and
suffix creates a different canonical token sequence at the seam. That is an
unfair complication for the intended trained-canonical comparison.

Retain the complete v1 scores and boundary audit as an initial diagnostic.
Correct the protocol before presenting the main natural-text result. Keep
exactly the same documents, discovery/confirmation assignments and disjoint
source windows. Move each cut to the nearest canonical token-end offset within
64 Unicode characters, preferring forward offsets in a tie. Select by encoding
validity only. Require that encoding the complete prefix+continuation normally
gives exactly the saved canonical prefix IDs followed by the32 target IDs.
The raw prefix is again at most64 canonical tokens and jointly fits all38
native encoders within128 total positions, with at least16 canonical tokens.
Do not drop or replace a frozen passage to make its loss different.

Recompute canonical, BOS-only and all38 transferred scores on this corrected
panel. The output head, checkpoint, native tokenizers and full-vocabulary
hypernetwork inference remain identical. The number of scored target tokens
stays65,024. Continuation bytes change because cuts move; compare models
within each version and explicitly distinguish the versions.

Apply the original natural-text analysis: balanced five-domain BPB, tails,
canonical-only context-benefit thresholds learned on discovery, independent
tokenizer-selection advantage on confirmation, clustered paired uncertainty,
and exclusion of image-link markup. The correction is motivated by inspecting
v1 results on these documents; call it a protocol correction, not a fully
blind replication study. No training is performed.

Use at most two B300 workers on blackpantherx86, no explicit CPU/memory. The
v1 inference cost about$1.42; verify credits again. Separate baseline timing
from native-setting timing for the measured conservative cost gate. Cloud
folder `anytok-rl-all-pool-20k/natural-transfer-aligned-v2/`; local metadata
`modal_results/natural_transfer_aligned/`. Dashboard run `natural-transfer-aligned`.

```
ANYTOK_NATURAL_VERSION=aligned-v2 MODAL_PROFILE=blackpantherx86 modal run --detach scripts/modal_natural_transfer.py --action run
```
