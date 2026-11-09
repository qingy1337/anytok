"""Audit and summarize one complete, matched full-data generation benchmark.

    python scripts/analyze_tokenizer_speed.py --candidate v6 --run-id RUN_ID

Use --partial only for a provisional view; incomplete grids never qualify the
speedup goal. Smoke tests are rejected. Tokenizer/model files stay in Modal.
"""
import argparse
from copy import deepcopy
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re

from pool_generalization import save_json, file_sha

ROOT = Path(__file__).resolve().parents[1]


def verify_reference_texts(rows, references):
    for row in rows:
        if row['manifest']['source_stats'] != references[row['case']]:
            raise ValueError('Training text differs from the original saved pool')


def adopt_capacity_rescue(rows, rescue, parent):
    """Replace an unfinished pair, preserving its actual execution provenance.

    Scheduling may change; native wheels, training code, data plan and settings
    must remain identical. Never replace an already matched original pair.
    """
    if len(rescue)!=2 or {r['variant'] for r in rescue}!={'baseline-200mb','fast-200mb'}:
        raise ValueError('A complete rescue pair is required')
    case=rescue[0]['case'];origin=rescue[0]['provenance']
    if any(r['case']!=case or r['provenance']!=origin for r in rescue):raise ValueError('Mixed rescue measurements')
    if origin.get('parent_id')!=parent['id'] or origin.get('nonpreemptible') is not True:
        raise ValueError('Rescue does not belong to this benchmark')
    if any(origin.get(k)!=v for k,v in parent.items() if k!='id'):
        raise ValueError('Rescue changed native wheels or benchmark protocol')
    if not rescue[0].get('container_id') or rescue[0]['container_id']!=rescue[1].get('container_id'):
        raise ValueError('Rescue timings came from different workers')
    existing=[r for r in rows if r['case']==case]
    if len(existing)==2 and existing[0].get('container_id') and existing[0]['container_id']==existing[1].get('container_id'):
        raise ValueError('Original pair already matched; do not replace it')
    if not existing:raise ValueError('Original configuration reference is missing')
    config=existing[0]['manifest']['config']
    if sum(config['bucket_bytes'].values())!=200_000_000 or any(r['manifest']['config']!=config for r in rescue):
        raise ValueError('Rescue changed the generator settings or 200 MB data quotas')
    selected=[r for r in rows if r['case']!=case]
    for row in rescue:
        normalized=deepcopy(row)
        normalized['execution_provenance']=normalized['provenance']
        normalized['provenance']=deepcopy(parent)
        selected.append(normalized)
    return selected


def analyze(rows, plan, require_complete=True):
    expected = {c['key']: c['config'] for c in plan['cases']}
    seen = set(); paired = {}; provenance = None
    for row in rows:
        key, variant = row['case'], row['variant']
        if key not in expected or variant not in ['baseline-200mb', 'fast-200mb']:
            raise ValueError('Unexpected benchmark case/variant')
        if (key, variant) in seen: raise ValueError('Duplicate benchmark row')
        seen.add((key, variant))
        if provenance is None: provenance = row['provenance']
        if row['provenance'] != provenance: raise ValueError('Mixed benchmark provenance')
        execution=row.get('execution_provenance')
        if execution and (execution.get('parent_id')!=provenance['id'] or execution.get('nonpreemptible') is not True
                          or any(execution.get(k)!=v for k,v in provenance.items() if k!='id')):
            raise ValueError('Rescue execution changed the native benchmark protocol')
        manifest = row['manifest']; audit = row['lossless_audit']; config = expected[key]
        if manifest['config'] != config or manifest['vocab_size_actual'] != config['vocab_size']:
            raise ValueError('Changed generator settings or requested vocabulary')
        quotas = config['bucket_bytes']
        if sum(quotas.values()) != 200_000_000: raise ValueError('Expected original 200 MB data quotas')
        source = manifest['source_stats']
        if set(source) != set(quotas): raise ValueError('Missing corpus buckets')
        for name, amount in quotas.items():
            if source[name]['quota_bytes'] != amount or not 0 <= amount-source[name]['used_bytes'] <= 3:
                raise ValueError('Training text quota changed')
        if (manifest['total_used_bytes'] != sum(r['used_bytes'] for r in source.values())
                or audit['roundtrip_failures'] != 0 or not audit['all_256_byte_symbols_present']
                or audit['normalizer'] is not None or audit['post_processor'] is not None
                or audit['decode_skip_special_tokens'] is not False or audit['fixtures'] < 256):
            raise ValueError('Lossless or full-data audit failed')
        for field in ['ready_seconds', 'subprocess_seconds', 'extended_audit_seconds']:
            if not math.isfinite(row[field]) or row[field] <= 0: raise ValueError('Invalid timing')
        paired.setdefault(key, {})[variant] = row
    required_cells = {(a,p) for a in ['bpe','wordpiece','unigram'] for p in ['regex','line256','no-regex']}
    complete = (len(seen) == len(expected)*2 and len(expected) == 9
                and {(c['algorithm'],c['pretokenizer']) for c in expected.values()} == required_cells)
    if require_complete and not complete: raise ValueError('Full nine-cell grid required for qualification')
    results = []; unmatched = []
    for case in plan['cases']:
        key = case['key']; pair = paired.get(key, {})
        if len(pair) != 2: continue
        b, f = pair['baseline-200mb'], pair['fast-200mb']; config = case['config']
        if b.get('execution_provenance')!=f.get('execution_provenance'):
            raise ValueError('Paired execution policies differ')
        if (b['manifest']['source_stats'] != f['manifest']['source_stats']
                or b['lossless_audit']['fixture_sha256'] != f['lossless_audit']['fixture_sha256']
                or b['cpu_model'] != f['cpu_model']):
            raise ValueError('Paired inputs, audit text or worker CPU changed')
        same_worker = bool(b.get('container_id') and b['container_id'] == f.get('container_id'))
        if not same_worker:
            unmatched.append({'case':key,'baseline_container_id':b.get('container_id'),
                              'fast_container_id':f.get('container_id'),'reason':'Awaiting a timing replay on the same worker'})
            continue
        order='not recorded'
        if b.get('at_utc') and f.get('at_utc'):
            order='fast then baseline' if datetime.fromisoformat(f['at_utc'])<datetime.fromisoformat(b['at_utc']) else 'baseline then fast'
        results.append({'case': key, 'config_index': config['index'], 'algorithm': config['algorithm'],
                        'pretokenizer': config['pretokenizer'], 'vocab_size': config['vocab_size'],
                        'training_bytes': f['manifest']['total_used_bytes'],
                        'baseline_seconds': b['ready_seconds'], 'fast_seconds': f['ready_seconds'],
                        'speedup': b['ready_seconds']/f['ready_seconds'],
                        'baseline_cpu_seconds': b['manifest']['cpu_seconds'], 'fast_cpu_seconds': f['manifest']['cpu_seconds'],
                        'baseline_peak_rss_gib': b['manifest']['peak_rss_gib'], 'fast_peak_rss_gib': f['manifest']['peak_rss_gib'],
                        'baseline_audit_seconds': b['extended_audit_seconds'], 'fast_audit_seconds': f['extended_audit_seconds'],
                        'baseline_subprocess_seconds': b['subprocess_seconds'], 'fast_subprocess_seconds': f['subprocess_seconds'],
                        'identical_serialized_tokenizer': b['manifest']['tokenizer_sha256'] == f['manifest']['tokenizer_sha256'],
                        'baseline_audit_tokens': b['lossless_audit']['fixture_tokens'], 'fast_audit_tokens': f['lossless_audit']['fixture_tokens'],
                        'audit_fixtures': f['lossless_audit']['fixtures'],
                        'zero_roundtrip_failures': True, 'same_container_verified': same_worker,
                        'actual_timing_order':order,
                        'replayed_after_preemption':bool(b.get('replay_after_preemption') or f.get('replay_after_preemption')),
                        'capacity_policy':'non-preemptible' if f.get('execution_provenance',{}).get('nonpreemptible') else 'default',
                        'finite_model_scores': all(r['lossless_audit'].get('finite_model_scores', False) for r in [b,f])})
    def group(part):
        baseline = sum(r['baseline_seconds'] for r in part); fast = sum(r['fast_seconds'] for r in part)
        return {'cases': len(part), 'baseline_seconds': baseline, 'fast_seconds': fast,
                'ready_speedup': baseline/fast if fast else None,
                'cpu_speedup': sum(r['baseline_cpu_seconds'] for r in part)/sum(r['fast_cpu_seconds'] for r in part) if part else None,
                'including_extended_audit_speedup': sum(r['baseline_subprocess_seconds'] for r in part)/sum(r['fast_subprocess_seconds'] for r in part) if part else None}
    total = group(results)
    matched_complete = complete and len(results)==len(expected)
    return {'at_utc': datetime.now(timezone.utc).isoformat(), 'complete': complete,
            'matched_complete':matched_complete,'unmatched_pairs':unmatched,
            'provenance': provenance, 'completed_phases': len(rows), 'expected_phases': len(expected)*2,
            'execution_extensions':{r['case']:r['execution_provenance'] for r in rows if r.get('execution_provenance')},
            'rows': results, 'total': total,
            'algorithms': {a: group([r for r in results if r['algorithm'] == a]) for a in ['bpe','wordpiece','unigram']},
            'qualified_at_least_5x': matched_complete and all(r['finite_model_scores'] for r in results) and total['ready_speedup'] >= 5,
            'qualified_at_least_10x': matched_complete and all(r['finite_model_scores'] for r in results) and total['ready_speedup'] >= 10,
            'measurement': 'Each full-size pair uses identical 200 MB byte quotas, source-document hashes, generator settings, four Rayon threads and the same worker. Ready time includes training, serialization and existing validation; extended Unicode/held-out audit is separate. Equal-frequency workload across nine representative algorithm/pretokenizer cells; ratio of summed times, not a guarantee for every configuration.',
            'timing_order': plan.get('timing_order', 'Alternating order across cells'),
            'limits': 'Nine representative configurations, one paired run per cell. Cloud CPU scheduling varies. Historical pool timings are not the denominator. No smaller corpus, reduced vocabulary, fewer Unigram iterations or restricted generator settings are used.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-id', required=True); p.add_argument('--candidate', choices=['v4','v5','v6','v7','v8'], default='v8')
    p.add_argument('--partial', action='store_true'); a = p.parse_args()
    if not re.fullmatch('[a-f0-9]{16}', a.run_id): raise ValueError('Expected full run ID, never smoke')
    state = ROOT/('modal_results/tokenizer_speed'+('_'+a.candidate if a.candidate != 'v4' else ''))
    plan_name = f'tokenizer-speed-{a.candidate}.json' if a.candidate in ['v6','v7','v8'] else 'tokenizer-speed-v1-implementation-only.json' if a.candidate == 'v5' else 'tokenizer-speed-v1.json'
    plan_path = ROOT/'plans'/plan_name; plan = json.loads(plan_path.read_text())
    path = state/(a.run_id+('-partial.json' if a.partial else '-results.json'))
    rows = json.loads(path.read_text())
    references = {}; reference_files = {}
    for case in plan['cases']:
        original_path = ROOT/case['historical_manifest']
        original = json.loads(original_path.read_text())
        if original['config'] != case['config']: raise ValueError('Original reference configuration differs')
        references[case['key']] = original['source_stats']
        reference_files[str(original_path.relative_to(ROOT))] = file_sha(original_path)
    verify_reference_texts(rows,references)
    report = analyze(rows, plan, not a.partial)
    if report['provenance']['plan_sha256'] != file_sha(plan_path): raise ValueError('Wrong frozen benchmark plan')
    report.update({'source_results_sha256': file_sha(path), 'candidate': a.candidate, 'run_id': a.run_id,
                   'original_pool_text_hashes_verified': True,'reference_manifest_sha256':reference_files})
    save_json(state/f'{a.run_id}-analysis.json', report)
    if report['rows']:
        with (state/f'{a.run_id}-analysis.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(report['rows'][0])); writer.writeheader(); writer.writerows(report['rows'])
    by_case = {r['case']: r for r in report['rows']}
    lines = ['# Tokenizer generation benchmark', '',
             f"Run `{a.run_id}` · candidate `{a.candidate}` · {len(rows)}/18 phases complete.", '',
             ('The complete grid meets the 5× aggregate target.' if report['qualified_at_least_5x'] else
              'The 5× aggregate target is not established by these results.'), '',
             '| Algorithm / pretokenizer | Original (min) | Optimized (min) | Speedup |',
             '| --- | ---: | ---: | ---: |']
    for case in plan['cases']:
        r = by_case.get(case['key'])
        lines.append(f"| {case['key']} | {r['baseline_seconds']/60:.2f} | {r['fast_seconds']/60:.2f} | {r['speedup']:.2f}× |" if r else
                     f"| {case['key']} | Pending pair | Pending pair | — |")
    if report['total']['ready_speedup'] is not None:
        lines += ['', f"Ratio of summed ready times across {len(by_case)}/9 completed pairs: **{report['total']['ready_speedup']:.2f}×**.",
                  f"CPU-time ratio: {report['total']['cpu_speedup']:.2f}×; including extended audits: {report['total']['including_extended_audit_speedup']:.2f}×."]
    lines += ['', report['measurement'], '', report['limits'], '',
              'Each accepted row retains the original 200 MB source quotas and requested vocabulary. Document hashes also match the original saved pool, not only the other timing variant. Paired rows have identical audit fixtures. Lossless audits require the full byte alphabet, reversible runtime settings and exact Unicode round trips with special tokens retained.', '',
              f"Matching physical container IDs: {sum(r['same_container_verified'] for r in by_case.values())}/{len(by_case)} pairs. Finite model-score audits: {sum(r['finite_model_scores'] for r in by_case.values())}/{len(by_case)} pairs.", '',
              'Incomplete grids, smoke tests, unmatched containers and missing score audits cannot qualify the target. Per-cell CPU time, peak RSS, audit duration, model identity and fixture token counts are in the accompanying CSV/JSON.', '',
              f"Cloud artifacts: `gildmember/anytok-fast-tokenizers/{a.run_id}`. Tokenizers and training text remain in persistent cloud storage.", '',
              f"Source result SHA256: `{report['source_results_sha256']}`.", '']
    if report['unmatched_pairs']:
        lines += ['Timings awaiting a same-worker replay are excluded from the aggregate: '+', '.join(r['case'] for r in report['unmatched_pairs'])+'.','']
    replays=[r for r in report['rows'] if r['replayed_after_preemption']]
    if replays:
        lines += ['Preemption replays can reverse the planned fast-first order. Actual replay orders, from completion timestamps: '+
                  '; '.join(r['case']+': '+r['actual_timing_order'] for r in replays)+'. Both variants still use the same worker and data.','']
    rescued=[r['case'] for r in report['rows'] if r['capacity_policy']=='non-preemptible']
    if rescued:
        lines += ['Non-preemptible CPU capacity was used for the following replacement pairs after repeated provider interruptions: '+', '.join(rescued)+'. Native wheels, training code, data and four Rayon threads are unchanged; both variants share the same worker. Execution provenance is retained separately from the logical benchmark identity.','']
    unigram=[r for r in report['rows'] if r['algorithm']=='unigram']
    if unigram:
        lines += ['The Unigram match cache trades higher peak memory for less computation. Peak RSS by completed case (original → optimized): '+
                  '; '.join(f"{r['pretokenizer']}: {r['baseline_peak_rss_gib']:.2f} → {r['fast_peak_rss_gib']:.2f} GiB" for r in unigram)+'.','']
    (state/f'{a.run_id}-findings.md').write_text('\n'.join(lines))
    print(json.dumps({k: report[k] for k in ['complete','completed_phases','total','qualified_at_least_5x']}))


if __name__ == '__main__': main()
