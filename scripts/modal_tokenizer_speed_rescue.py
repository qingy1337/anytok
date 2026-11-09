"""Run a repeatedly preempted timing pair on non-preemptible CPU capacity.

Uses the original benchmark function body, image, four Rayon threads and 200 MB
configuration. No explicit CPU/memory reservations. Results use a separate
prefix and never overwrite the live benchmark. Default action prints the plan.
"""
from datetime import datetime, timezone
from decimal import Decimal
import json
import os
from pathlib import Path

import modal

if os.environ.get('ANYTOK_SPEED_CANDIDATE','v8')!='v8':raise ValueError('Rescue requires the frozen v8 runtime')
os.environ['ANYTOK_SPEED_CANDIDATE']='v8'
import modal_tokenizer_speed as protocol
from pool_generalization import file_sha, save_json
from tokenizer_pool import digest

ROOT=Path(__file__).resolve().parents[1]
STATE=ROOT/'modal_results/tokenizer_speed_rescue'
app=modal.App('anytok-tokenizer-speed-rescue')
stable_case=app.function(image=protocol.image,
    volumes={'/train':protocol.source.with_mount_options(read_only=True),'/out':protocol.outputs},
    max_containers=1,timeout=21600,single_use_containers=True,nonpreemptible=True)(protocol.benchmark_case.get_raw_f())


@app.local_entrypoint()
def main(action: str='plan',case: str='unigram-no-regex',
         benchmark_pointer: str='modal_results/tokenizer_speed_v8/ed3e9f11892aca02-benchmark-call.json',resume_from: str=''):
    if os.environ.get('MODAL_PROFILE')!='gildmember':raise ValueError('Explicit gildmember required')
    if action not in ['plan','smoke','benchmark']:raise ValueError('Invalid action')
    saved=json.loads(Path(benchmark_pointer).read_text());original=saved['provenance']
    plan=json.loads(protocol.PLAN.read_text())
    selected=next(c for c in plan['cases'] if c['key']==case)
    if (file_sha(protocol.PLAN)!=original['plan_sha256'] or file_sha(protocol.FAST)!=original['fast_wheel_sha256']
            or file_sha(protocol.BASE)!=original['base_wheel_sha256']):raise ValueError('Frozen benchmark inputs changed')
    provenance={k:v for k,v in original.items() if k!='id'}
    provenance.update({'parent_id':original['id'],'nonpreemptible':True,'rescue_launcher_sha256':file_sha(__file__),'rescue_case':case})
    provenance['id']=digest(provenance)[:16]
    if resume_from:
        previous=json.loads(Path(resume_from).read_text())
        candidate=previous['provenance']
        if (previous['case']!=case or previous['action']!=action
                or candidate.get('parent_id')!=original['id'] or candidate.get('nonpreemptible') is not True
                or any(candidate.get(k)!=v for k,v in original.items() if k!='id')):
            raise ValueError('Resume pointer uses different settings, runtime or action')
        provenance=candidate
    old=json.loads((ROOT/selected['historical_manifest']).read_text())
    fast=json.loads((ROOT/'modal_results/tokenizer_speed_v8'/f'{original["id"]}-partial.json').read_text())
    fast=next(r for r in fast if r['case']==case and r['variant']=='fast-200mb')
    estimate=3*(old['estimated_process_cost_usd']+fast['manifest']['estimated_process_cost_usd'])
    if action=='plan':
        print(json.dumps({'case':case,'provenance':provenance,'nonpreemptible':True,
                          'estimated_pair_cost_usd':estimate,'estimate_note':'Uses peak-RSS estimates from the original pool and completed fast run; not a billing cap.',
                          'cpu_reservation':None,'memory_reservation':None,'rayon_threads':4},indent=2));return
    STATE.mkdir(parents=True,exist_ok=True);smoke=action=='smoke'
    job=provenance['id']+('-smoke' if smoke else '')
    pointer=Path(resume_from) if resume_from else STATE/f'{job}-{case}-call.json'
    if pointer.exists():call=modal.FunctionCall.from_id(json.loads(pointer.read_text())['call_id'])
    else:
        from modal_credits import billing_row
        balance=billing_row('gildmember','this month',Decimal(30))
        if balance['status']!='ok' or float(balance['estimated_remaining_usd'])<6:raise RuntimeError('Verified $6 credit margin required')
        with protocol.outputs.batch_upload(force=True) as batch:
            batch.put_file(protocol.PLAN,provenance['id']+'/plan.json')
            batch.put_file(Path(__file__),provenance['id']+'/rescue-source.py')
        call=stable_case.spawn(selected,provenance,smoke)
        save_json(pointer,{'call_id':call.object_id,'provenance':provenance,'case':case,'action':action,'parent_pointer':benchmark_pointer})
    while True:
        try:rows=call.get(timeout=10);break
        except (TimeoutError,modal.exception.TimeoutError):pass
        try:entries=list(protocol.outputs.iterdir(job,recursive=True))
        except (FileNotFoundError,modal.exception.NotFoundError):entries=[]
        count=sum(e.path.endswith('/complete.json') for e in entries)
        record={'at_utc':datetime.now(timezone.utc).isoformat(),'stage':f'{action}: non-preemptible pair','completed':count,'total':2,'call_id':call.object_id,'case':case}
        save_json(STATE/'status.json',record)
        with (STATE/'events.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
        print(json.dumps(record),flush=True)
    save_json(STATE/f'{job}-results.json',rows)
    save_json(STATE/'status.json',{'at_utc':datetime.now(timezone.utc).isoformat(),'stage':'complete','completed':len(rows),'total':2})
    print(json.dumps({'job':job,'rows':len(rows),'same_container':rows[0]['container_id']==rows[1]['container_id']}),flush=True)
