"""Export a human-readable RL report using saved score/config metadata only.

Run from a checkout containing the original local metadata receipts. The exported
folder can be verified and plotted offline, without these receipts or Modal.
"""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'reports/rl-tokenizer-selection-20k'
sys.path.insert(0,str(ROOT))
from monitoring.learning import LearningReader
from monitoring.policy_summary import summarize


def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')


def write_csv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    columns=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns,lineterminator='\n');writer.writeheader();writer.writerows(rows)


def main():
    reader=LearningReader(ROOT)
    records={method:reader.records(f'modal_results/{folder}/*-train.log') for method,folder in
             [('rl','anytok_pool_settings_rl_20k'),('uniform','anytok_pool_20k')]}
    # Export only structured numerical/configuration metadata, never raw logs.
    for method,items in records.items():
        train=[r for (stage,_),r in items.items() if stage=='train'];train.sort(key=lambda r:r['step'])
        if [r['step'] for r in train]!=list(range(1,20001)):raise ValueError('Incomplete training trace')
        columns=['step','tokenizer_config_index','tokenizer_vocab_size','loss_nats_per_token',
                 'train_bpb','scored_tokens','scored_bytes','rejected_examples','truncated_forms',
                 'input_lexical_loss','gradient_norm_before_clip','hn_lr','lm_lr',
                 'policy_reward_bpb','policy_advantage_bpb','policy_baseline_bpb',
                 'policy_action_probability','policy_selected_probability','policy_effective_tokenizers',
                 'policy_max_probability','policy_unique_tokenizers','policy_entropy','policy_updates']
        write_csv(REPORT/f'data/training/{method}.csv',[{k:r.get(k,'') for k in columns} for r in train])
    sanity=[]
    for method,items in records.items():
        for (stage,step),r in sorted(items.items()):
            if stage!='eval':continue
            for i,result in enumerate(r['results']):
                sanity.append({'method':method,'step':step,'tokenizer':f'T{i+1}',
                    'bpb':result['generated_bpb'],'canonical_bpb':result['canonical_bpb'],
                    'gap_bpb':result['generated_bpb']-result['canonical_bpb'],
                    'examples':result['examples'],'continuation_bytes':result['bytes'],
                    'truncated_vocabulary_forms':result['truncated_forms']})
    write_csv(REPORT/'data/training/sanity.csv',sanity)
    copies={
        'plans/tokenizer-pool-1000-v1.json':'data/pool/plan.json',
        'results/specialty_48k_sweep_plan.json':'data/pool/ood-48k-plan.json',
        'modal_results/coverage_evidence/result.json':'data/evaluation/broad-screen.json',
        'modal_results/coverage_evidence/analysis.json':'data/analysis/coverage.json',
        'modal_results/coverage_evidence/paired-screen.csv':'data/evaluation/broad-screen.csv',
        'modal_results/coverage_evidence/policy-history.json':'data/policy/history.json',
        'modal_results/coverage_evidence/component-swaps.json':'data/analysis/component-swaps.json',
        'modal_results/coverage_evidence/feature-audit.json':'data/analysis/feature-audit.json',
        'modal_results/coverage_evidence/medical-token-audit.json':'data/analysis/medical-token-audit.json',
        'modal_results/coverage_evidence/nonlinear-reward.json':'data/analysis/reward-prediction.json',
        'modal_results/coverage_evidence/rng-audit.json':'data/provenance/rl-rng.json',
        'modal_results/coverage_evidence/uniform-rng-audit.json':'data/provenance/uniform-rng.json',
        'modal_results/coverage_evidence/spec.json':'data/provenance/coverage-spec.json',
        'modal_results/coverage_evidence/published.json':'data/provenance/persistent-analysis.json',
        'modal_results/rl_ood_20k/comparison.json':'data/evaluation/original-panel.json',
        'modal_results/rl_ood_20k/evaluate-result.json':'data/evaluation/rl-per-document.json',
        'modal_results/rl_ood_20k/evaluation-spec.json':'data/provenance/rl-evaluation-spec.json',
        'modal_results/uniform_trajectories/trajectory-spec.json':'data/evaluation/panel-spec.json',
        'modal_results/uniform_trajectories/checkpoints/step-020000.json':'data/evaluation/uniform-per-document.json',
        'modal_results/uniform_trajectories/exposures.json':'data/policy/uniform-counts.json',
        'modal_results/anytok_pool_settings_rl_20k/verification.json':'data/provenance/launch-verification.json',
        'modal_results/anytok_pool_settings_rl_20k/neural-net-1411-audit-final-result.json':'data/provenance/checkpoint-audit.json',
        'plans/coverage-evidence-v1.json':'data/provenance/diagnostic-design.json',
    }
    sources=[
        'scripts/modal_anytok_pool_rl_20k.py','scripts/run_anytok_pool_rl_20k.py',
        'scripts/anytok_pool_rl_config.py','scripts/anytok_pool_config.py',
        'scripts/modal_anytok_pool_20k.py','scripts/run_anytok_pool_20k.py',
        'scripts/modal_credits.py','scripts/modal_tokenizer_pool_transfer.py',
        'scripts/modal_pool_generalization.py','scripts/modal_uniform_trajectory.py',
        'scripts/modal_eval_rl_ood.py','scripts/rl_ood_eval.py','scripts/analyze_rl_ood.py',
        'scripts/modal_coverage_evidence.py','scripts/analyze_coverage_evidence.py',
        'scripts/analyze_uniform_trajectory.py','scripts/uniform_trajectory.py',
        'scripts/uniform_weakness.py','scripts/pool_generalization.py','scripts/pool_ood_eval.py',
        'plotting_scripts/plot_coverage_evidence.py','plotting_scripts/plot_rl_ood.py',
        'plotting_scripts/plot_joint_hypernetwork_learning.py','plotting_scripts/plot_uniform_trajectory.py',
        'runtime/anytok_pool_settings_rl_20k/hf_hypernet/settings_policy.py',
        'runtime/anytok_pool_settings_rl_20k/scripts/train_joint_smollm_input.py',
        'runtime/anytok_pool_20k/scripts/train_joint_smollm_input.py',
        'tests/test_pool_settings_policy.py','tests/test_rl_ood_eval.py','tests/test_coverage_evidence.py',
    ]
    sources.extend(str(p.relative_to(ROOT)) for p in sorted((ROOT/'runtime/anytok_pool_20k/hf_hypernet').glob('*.py')))
    sources.extend(str(p.relative_to(ROOT)) for p in sorted((ROOT/'runtime/anytok_pool_20k/rust_utils').rglob('*'))
                   if p.is_file() and 'target' not in p.parts)
    copies.update({name:'code/original/'+name for name in sources})
    provenance=[]
    for source,destination in copies.items():
        src=ROOT/source;dst=REPORT/destination;dst.parent.mkdir(parents=True,exist_ok=True)
        raw=src.read_bytes();exported=raw.replace(b'\r\n',b'\n') if src.suffix=='.csv' else raw;dst.write_bytes(exported)
        provenance.append({'source':source,'report_path':destination,'sha256':hashlib.sha256(raw).hexdigest(),
                           'report_sha256':hashlib.sha256(exported).hexdigest()})
    logs=[p for folder in ['anytok_pool_settings_rl_20k','anytok_pool_20k']
          for p in sorted((ROOT/'modal_results'/folder).glob('*-train.log'))]
    write_json(REPORT/'data/provenance/sources.json',{
        'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'copied_files':provenance,
        'training_log_sources':[{'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in logs],
        'training_export':'Numerical whitelist; deduplicated by (stage, step), matching the run monitor. Source logs remain outside this report.'})
    history=json.loads((REPORT/'data/policy/history.json').read_text())
    configs=json.loads((REPORT/'data/pool/plan.json').read_text())['configs']
    write_json(REPORT/'data/policy/distributions.json',[summarize(configs,h['probabilities'],h['counts'],h['step']) for h in history])
    for name in ['rl_ood_20k','coverage_evidence_overview','coverage_evidence_heatmap','coverage_evidence_policy','coverage_evidence_mechanism']:
        for ext in ['png','pdf']:
            dst=REPORT/f'plots/{name}.{ext}';dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/f'plots/{name}.{ext}',dst)
    subprocess.run([sys.executable,str(REPORT/'code/reproduce.py'),'--write-derived'],check=True)
    print(f'Exported {REPORT.relative_to(ROOT)}')


if __name__=='__main__':main()
