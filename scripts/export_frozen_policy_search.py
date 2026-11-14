"""Human-readable setup, complete rankings and provenance for frozen policy search."""
import csv
import gzip
import json
from pathlib import Path

from pool_generalization import file_sha, save_json

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'reports/frozen-policy-search-20k'


def export_entropy_variant(report, summary, cache):
    baseline_dir = ROOT / summary['run_plan']['baseline_report']
    baseline = json.loads((baseline_dir / 'analysis.json').read_text())
    if file_sha(report / 'reward_table.json.gz') != file_sha(baseline_dir / 'reward_table.json.gz'):
        raise ValueError('Entropy ablation changed the complete loss table')
    before = baseline['policy_settings'];after = summary['policy_settings']
    differences = {k: [before[k], after[k]] for k in before if before[k] != after[k]}
    if differences != {'entropy_weight': [0.01, 0.0]}:
        raise ValueError('A setting other than entropy changed')
    comparison = dict(settings_changes=differences, reward_table_sha256=file_sha(report / 'reward_table.json.gz'),
        before=baseline['final'], after=summary['final'], expected_scores={
            split: dict(entropy001_bpb=baseline['splits'][split]['policy_bpb'],
                entropy0_bpb=summary['splits'][split]['policy_bpb'],
                canonical_bpb=summary['splits'][split]['canonical_bpb'],
                uniform_bpb=summary['splits'][split]['uniform_bpb'], best_bpb=summary['splits'][split]['best_bpb'])
            for split in ['screen', 'confirm']})
    save_json(report / 'comparison.json', comparison)
    lines = ['# Frozen-model policy: entropy 0 versus 0.01', '',
        '**Only the entropy coefficient changes, from 0.01 to 0.** Both policies start fresh from exactly the same uniform initialization and seed 43, and run 20,000 updates. The LM and hypernetwork remain frozen at the same final RL20k checkpoint.', '',
        'The complete reward table is byte-for-byte identical: all 1,000 native pool tokenizers, all 288 original reward passages and all 288 original held-out check passages, with corrected token boundaries. No passages or tokenizers are dropped, resampled or replaced. This rerun uses exact cached inference losses locally; no new model inference is needed.', '',
        'The reward remains raw continuation BPB averaged within each of nine domains and then equally across domains. The learner receives only the selected tokenizer’s full-reward-set BPB. Check text and complete rankings are evaluator-only. The shared 18→64→64→1 Tanh network, Adam learning rate 0.001, moving-average baseline decay 0.99, action sampling and update count are unchanged.', '',
        '| Final measurement | Entropy 0.01 | Entropy 0 |', '|---|---:|---:|']
    for label, split, key in [('Probability on hardest ten (reward text)', 'screen', 'top10_mass'),
        ('Probability on exact worst #0635', 'screen', 'top1_mass'),
        ('Probability on hardest ten (check text)', 'confirm', 'top10_mass')]:
        lines.append(f'| {label} | {baseline["final"][split][key]:.2%} | {summary["final"][split][key]:.2%} |')
    for label, split in [('Expected gap on reward text', 'screen'), ('Expected gap on held-out check text', 'confirm')]:
        a = baseline['splits'][split];b = summary['splits'][split]
        lines.append(f'| {label} | {a["policy_bpb"] - a["canonical_bpb"]:+.5f} BPB | {b["policy_bpb"] - b["canonical_bpb"]:+.5f} BPB |')
    final = summary['final'];index = final['most_probable_index']
    lines += ['', f'The entropy-zero policy’s most probable action is **#{index:04d}**, with probability **{final["max_probability"]:.2%}**. Its reward-text rank is **{final["screen"]["most_probable_rank"]}**, and its held-out rank is **{final["confirm"]["most_probable_rank"]}**. It tried {final["unique_tokenizers"]} distinct tokenizers, compared with {baseline["final"]["unique_tokenizers"]} with entropy 0.01.', '',
        '![Entropy comparison](frozen_policy_search.png)', '',
        'Higher expected BPB means the adversary selects harder tokenizers; the frozen LM itself does not change. The exact worst on both fixed text sets is #0635. The first selection of that ID is recorded separately from concentration on it: merely sampling it once is not evidence of a sustained preference.', '',
        'With entropy zero, the unconstrained optimum puts all probability on the highest reward. The policy is not told that optimum, the rankings or other tokenizers’ rewards. With entropy 0.01, the optimum trades reward for spread. This is one matched policy seed on a stationary full-screen reward, not a noisy-minibatch or co-adapting-model experiment; it cannot by itself establish why the earlier joint-training run behaved as it did.', '',
        '## Context and provenance', '',
        'Both sets contain 32 source documents from each of nine domains: English web, code, Chinese, Japanese, Arabic, Hindi, math, medical and legal. Every input is BOS + native prefix + 32 canonical continuation tokens, at most 128 positions. The prefix supplies context; only the 32 continuation targets are scored. All tokenizers share each passage’s raw prefix, targets and bytes. The output head remains canonical.', '',
        'All source IDs, corrected cuts, continuation IDs/bytes, native prefix lengths, tokenizer hashes, trained-model manifests and per-passage loss totals are preserved in reward_table.json.gz. The ablation changes the policy coefficient only; run_plan.json identifies the exact original reward-table hash and parent plan. comparison.json verifies that no other policy setting changed.', '',
        '## Files and reproduction', '',
        '- `analysis.json`, `curve.json`, `actions.json.gz`: complete final ranks/probabilities and all 20,000 sampled actions.',
        '- `comparison.json`: matched baseline comparison and coefficient-only settings check.',
        '- `policy.pt`: final learned policy; resumable checkpoints every 2,000 updates are saved in cloud storage.',
        '- `reward_table.json.gz`, `run_plan.json`: complete unchanged environment and approved ablation plan.',
        '- `frozen_policy_search.png` and `.pdf`: baseline-versus-ablation plots.', '',
        'Reproduce locally, without cloud access: `python scripts/frozen_policy_search.py --cache reports/frozen-policy-search-20k/reward_table.json.gz --run-plan plans/frozen-policy-entropy0-v1.json`. The baseline report is `reports/frozen-policy-search-20k/`. Add `--persist` only when cloud checkpoint/report uploads are desired.', '',
        'The new report and policy checkpoints are persisted separately under aurora / anytok-frozen-policy-search / frozen-policy-entropy0-v1/. Original results and all frozen model, corpus and tokenizer artifacts remain intact in persistent cloud storage.', '']
    (report / 'README.md').write_text('\n'.join(lines))
    names = ['README.md', 'reward_table.json.gz', 'run_plan.json', 'comparison.json', 'analysis.json', 'curve.json', 'actions.json.gz', 'policy.pt', 'frozen_policy_search.png', 'frozen_policy_search.pdf']
    save_json(report / 'manifest.json', dict(files={n: dict(bytes=(report / n).stat().st_size, sha256=file_sha(report / n)) for n in names}))


def main(report=REPORT):
    if report != REPORT:
        summary = json.loads((report / 'analysis.json').read_text())
        cache = json.loads(gzip.decompress((report / 'reward_table.json.gz').read_bytes()))
        if summary.get('reward_objective') == 'canonical_gap':
            from export_frozen_policy_regret import export_regret
            return export_regret(report, summary, cache)
        return export_entropy_variant(report, summary, cache)
    summary = json.loads((REPORT / 'analysis.json').read_text())
    cache = json.loads(gzip.decompress((REPORT / 'reward_table.json.gz').read_bytes()))
    final = summary['final'];rows = cache['rows'];configs = cache['configs']
    lines = ['# Can the adversarial policy find hard tokenizers with the model frozen?', '',
        'We freeze the final RL20k language model and hypernetwork. We train only the original shared-settings tokenizer policy, starting uniformly over all 1,000 existing native tokenizers. Higher BPB is better for this adversary: it wants to find tokenizers that make the frozen model lose more.', '',
        '## The evaluation is complete', '',
        'Every one of the 1,000 tokenizers is scored on all 288 original screen passages and all 288 original confirmation passages. No tokenizer or passage is dropped. These are the original broad nine-domain samples, not the 229 context-dependent stress passages and not a subset of already-valid cuts.', '',
        'The policy receives rewards from the screen text only; we call it **reward text** below. Confirmation text is **held-out check text**: it never enters a policy update, checkpoint choice or hyperparameter choice. Both sets contain 32 documents in each of English web, code, Chinese, Japanese, Arabic, Hindi, math, medical and legal text.', '',
        'The old seam bug is repaired by the nearest reversible canonical-boundary rule. Already-valid original contexts stay unchanged. Invalid cuts move by at most 64 characters, based only on encoding validity. All 1,000 tokenizers share the same raw prefix and canonical continuation within each passage. Prefixes are jointly trimmed only when necessary to fit all tokenizers. A moved cut can change its continuation bytes; this is explicitly a corrected evaluation protocol.', '',
        '| Text | Passages | Changed cuts | Additional joint prefix trims | Scored target tokens | Continuation bytes |',
        '|---|---:|---:|---:|---:|---:|']
    for split, label in [('screen', 'Reward'), ('confirm', 'Held-out check')]:
        sub = [r for r in rows if r['split'] == split]
        lines.append(f'| {label} | {len(sub)} | {sum(r["cut_shift_chars"] != 0 for r in sub)} | {sum(r.get("pool_trim_chars", 0) > 0 for r in sub)} | {32 * len(sub):,} | {sum(r["continuation_bytes"] for r in sub):,} |')
    lines += ['', '## What one policy update does', '',
        '1. The shared settings network assigns a logit to each tokenizer from its 18 generator-setting features, then samples one ID from the softmax.',
        '2. The learner receives only that tokenizer’s BPB on the full 288-passage reward set. BPB is byte-weighted within each domain, then the nine domains are weighted equally.',
        '3. REINFORCE updates only the policy network. The complete ranking and check-text losses are evaluator-only.', '',
        'The fixed model’s per-passage losses are computed once and cached. Reusing them preserves the exact reward; no surrogate reward model or score approximation is used. This experiment deliberately supplies the full-screen mean rather than noisy fresh minibatch rewards. It tests whether the policy can learn a stationary hard-tokenizer ranking. It does not reproduce the reward noise or changing model of the earlier joint RL run.', '',
        'The policy is unchanged: 18→64→64→1 Tanh MLP, seed 43, Adam learning rate 0.001, EMA baseline decay 0.99, entropy bonus 0.01, and exactly uniform initial probabilities. There are 20,000 policy updates. No model or hypernetwork optimizer is instantiated. Scoring is inference-only with gradients disabled, native vocabularies complete, and checkpoint/tokenizer/feature hashes verified.', '',
        '## Results', '',
        '| Text | Canonical BPB | Uniform expected BPB | Learned-policy expected BPB | True highest BPB | True worst ID |',
        '|---|---:|---:|---:|---:|---:|']
    for split, label in [('screen', 'Reward'), ('confirm', 'Held-out check')]:
        r = summary['splits'][split]
        lines.append(f'| {label} | {r["canonical_bpb"]:.5f} | {r["uniform_bpb"]:.5f} | {r["policy_bpb"]:.5f} | {r["best_bpb"]:.5f} | {r["best_index"]:04d} |')
    check = summary['splits']['confirm']
    lines += ['', f'The policy learned to choose harder tokenizers on both text sets. On the held-out check text, its expected transfer gap is **+{check["policy_bpb"] - check["canonical_bpb"]:.5f} BPB**, versus **+{check["uniform_bpb"] - check["canonical_bpb"]:.5f}** under uniform selection. The increase of **+{check["policy_bpb"] - check["uniform_bpb"]:.5f} BPB** persists on text that never supplied a policy reward.', '',
        'Expected BPB averages over the policy’s selection probabilities; it is not the single hardest sampled observation. Higher means harder selections for this frozen model. Uniform is an exact average of all 1,000 measured tokenizers.', '',
        '| Text ranking | Probability on hardest 1 | Hardest 10 | Hardest 50 |', '|---|---:|---:|---:|']
    for split, label in [('screen', 'Reward'), ('confirm', 'Held-out check')]:
        r = final[split];lines.append(f'| {label} | {r["top1_mass"]:.1%} | {r["top10_mass"]:.1%} | {r["top50_mass"]:.1%} |')
    index = final['most_probable_index'];cfg = configs[index]
    lines += ['', f'The most probable final tokenizer is **#{index:04d}**, {cfg["algorithm"]}/{cfg["pretokenizer"]}, vocabulary {cfg["vocab_size"]:,}, with probability **{final["max_probability"]:.1%}**. Its reward-text rank is {final["screen"]["most_probable_rank"]} and held-out rank is {final["confirm"]["most_probable_rank"]}. The policy sampled {final["unique_tokenizers"]} distinct tokenizers.', '',
        '![Frozen policy learning](frozen_policy_search.png)', '',
        f'Tokenizer #0635 remains the highest measured BPB across the complete pool on both text sets. The policy first sampled it at update {summary["splits"]["screen"]["first_best_selection_step"]}; the single uniform baseline first sampled it at update {summary["splits"]["screen"]["uniform_first_best_selection_step"]}. One seed does not establish a general discovery-speed advantage. The final policy favors a hard group rather than making the exact worst tokenizer its most probable action.', '',
        'The entropy bonus favors retaining some probability on alternatives. We therefore also compare with the exact optimum of expected reward plus 0.01 × entropy: softmax(true reward / 0.01). That reference is used only for evaluation and never supplied to the learner. A policy need not put 100% on one tokenizer to optimize this objective.', '',
        'This is one policy seed and one frozen model checkpoint on two fixed text samples. Complete rankings establish the highest BPB on these samples, not a universal worst tokenizer for arbitrary text. The test isolates search under stationary exact rewards; success does not establish that the original policy learned equally well with noisy minibatches and a co-adapting model.', '',
        '## Files and reproduction', '',
        '- `reward_table.json.gz`: complete per-passage NLL totals for all 1,000 × 576, canonical totals, source IDs, corrected cuts, native prefix counts, tokenizer hashes/settings and the frozen plan.',
        '- `analysis.json`, `rankings.csv`: full reward/check rankings, learned probabilities, counts and reference scores.',
        '- `curve.json`, `actions.json.gz`: training progression and every sampled tokenizer ID.',
        '- `policy.pt`: final policy weights, settings, feature table and selection counts; it contains no LM/hypernetwork weights.',
        '- `frozen_policy_search.png` and `.pdf`: reusable result plots.', '',
        'Reproduce the policy run with `python scripts/frozen_policy_search.py --cache reports/frozen-policy-search-20k/reward_table.json.gz`, then `python scripts/export_frozen_policy_search.py`. The policy source hash is frozen in `plans/frozen-policy-search-v1.json`. Original worker: `scripts/modal_frozen_policy_cache.py`; local learner: `scripts/frozen_policy_search.py`.', '',
        'Corpora, native tokenizer files, LM/hypernetwork checkpoints, raw per-token scores and corrected raw prefixes remain in persistent cloud storage. Only prediction-loss metadata crosses to the local learner. Source: blackpantherx86 / anytok-rl-all-pool-20k; scoring copy: aurora / anytok-frozen-policy-search; experiment prefix: frozen-policy-search-v1. No CPU or memory reservations are specified for Modal jobs.', '']
    (REPORT / 'README.md').write_text('\n'.join(lines))
    by_split = {s: {r['index']: r for r in summary['splits'][s]['ranking']} for s in ['screen', 'confirm']}
    with (REPORT / 'rankings.csv').open('w') as stream:
        fields = ['index', 'algorithm', 'pretokenizer', 'vocab_size', 'probability', 'selections', 'screen_rank', 'screen_bpb', 'screen_gap', 'confirm_rank', 'confirm_bpb', 'confirm_gap']
        writer = csv.DictWriter(stream, fieldnames=fields);writer.writeheader()
        for i in range(1000):
            a, b = by_split['screen'][i], by_split['confirm'][i];cfg = configs[i]
            writer.writerow(dict(index=i, algorithm=cfg['algorithm'], pretokenizer=cfg['pretokenizer'], vocab_size=cfg['vocab_size'], probability=a['probability'], selections=a['selections'], screen_rank=a['rank'], screen_bpb=a['bpb'], screen_gap=a['gap_bpb'], confirm_rank=b['rank'], confirm_bpb=b['bpb'], confirm_gap=b['gap_bpb']))
    names = ['README.md', 'reward_table.json.gz', 'analysis.json', 'rankings.csv', 'curve.json', 'actions.json.gz', 'policy.pt', 'frozen_policy_search.png', 'frozen_policy_search.pdf']
    save_json(REPORT / 'manifest.json', dict(files={n: dict(bytes=(REPORT / n).stat().st_size, sha256=file_sha(REPORT / n)) for n in names}))


if __name__ == '__main__':
    main()
