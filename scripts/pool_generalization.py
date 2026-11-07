"""Frozen design and paired analysis for seen-versus-fresh tokenizers."""
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random

ALGORITHMS = ('bpe','wordpiece','unigram')
PRETOKENIZERS = ('regex','line256','no-regex')
SEED = 20261107
STEPS = (2000,6000,12000)


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8<<20),b''):h.update(chunk)
    return h.hexdigest()


def save_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
    temporary.replace(path)


def make_plan(pool, manifests, exposures, panel_hashes):
    rng=random.Random(SEED);seeds=random.Random(SEED+1)
    pairs=[]
    for algorithm in ALGORITHMS:
        for pretokenizer in PRETOKENIZERS:
            group=[c for c in pool['configs'] if (c['algorithm'],c['pretokenizer'])==(algorithm,pretokenizer)]
            for seen in rng.sample(group,2):
                index=len(pairs)
                fresh=deepcopy(seen)
                fresh.update(index=index,seed=seeds.randrange(2**63))
                fresh.pop('config_sha256')
                fresh['config_sha256']=digest(fresh)
                pairs.append({'pair':index,'seen':seen,'fresh':fresh,
                              'stratum_weight':len(group)/pool['count'],
                              'seen_sha256':manifests[seen['index']]['tokenizer_sha256'],
                              'seen_exposures':{str(s):exposures[s][seen['index']] for s in STEPS}})
    skip={b['name']:max(c['bucket_bytes'][b['name']] for c in pool['configs']) for b in pool['recipe']['buckets']}
    needed={b['name']:max(p['fresh']['bucket_bytes'][b['name']] for p in pairs) for b in pool['recipe']['buckets']}
    for b in pool['recipe']['buckets']:
        if skip[b['name']]+needed[b['name']]>b['train_available_bytes']:
            raise ValueError('Insufficient disjoint source data; do not resample the panel')
    plan={'seed':SEED,'source_pool_sha256':pool['plan_sha256'],'pairs':pairs,'steps':list(STEPS),
          'position_cap':128,'continuation_tokens':32,'documents':256,
          'continuation_bytes':38534,'total_bytes_per_tokenizer':200000000,
          'recipe':pool['recipe'],'skip_bytes':skip,'required_bytes':needed,'panel_hashes':panel_hashes,
          'design':'Two random training-pool members per algorithm/pretokenizer cell; fresh matched settings, new merge seeds and disjoint source-document ranges. No loss or runtime selection.',
          'primary':'Pool-weighted fresh-minus-seen BPB at 12k and its change from 6k; paired tokenizer/document bootstrap.',
          'limitations':'Matched configurations test unseen tokenizer instances; they do not test new mixture/vocabulary/noise settings. Three in-pool members have zero exposure at 2k; all 18 are seen by 6k.'}
    plan['plan_sha256']=digest(plan)
    return plan


def load_plan(path):
    plan=json.loads(Path(path).read_text())
    if digest({k:v for k,v in plan.items() if k!='plan_sha256'})!=plan['plan_sha256']:
        raise ValueError('Experiment plan hash mismatch')
    return plan


def copy_disjoint_documents(lines, output, skip_bytes, required_bytes):
    """Skip whole records beyond every pool quota, then retain fresh documents."""
    skipped=used=skipped_docs=documents=0
    for line in lines:
        size=len(json.loads(line)['text'].encode('utf-8'))
        if skipped<skip_bytes:
            skipped+=size;skipped_docs+=1
            continue
        if used>=required_bytes:break
        output.write(line);used+=size;documents+=1
    if skipped<skip_bytes or used<required_bytes:
        raise ValueError('Insufficient disjoint documents')
    return {'skipped_bytes':skipped,'skipped_documents':skipped_docs,'available_bytes':used,'documents':documents}


def load_native(path, manifest):
    import numpy as np
    from tokenizers import Tokenizer
    raw=Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=manifest['tokenizer_sha256']:
        raise ValueError('Tokenizer checksum mismatch')
    tokenizer=Tokenizer.from_str(raw.decode())
    vocab=tokenizer.get_vocab();size=manifest['vocab_size_actual']
    if len(vocab)!=size or set(vocab.values())!=set(range(size)):
        raise ValueError('Tokenizer IDs are not contiguous')
    model=json.loads(raw)['model']
    priors=[float(row[1]) for row in model['vocab']] if model['type']=='Unigram' else [0.]*size
    if len(priors)!=size or not np.isfinite(priors).all():raise ValueError('Invalid priors')
    tokens=[None]*size
    for token,index in vocab.items():tokens[index]=token
    if tokenizer.token_to_id('<|endoftext|>') is None:raise ValueError('Missing EOS')
    return tokenizer,tokens,priors


def shared_prefix(raw, continuation, canonical, targets, cap=128):
    """One common raw prefix for every target, with identical fixed suffix IDs."""
    def check(start):
        prefix=raw[start:]
        source=canonical(prefix,add_special_tokens=False).input_ids
        if (not source or 1+len(source)+len(continuation)>cap
                or canonical.decode(source,clean_up_tokenization_spaces=False)!=prefix):return None
        maximum=len(source)
        for tokenizer in targets:
            ids=tokenizer.encode(prefix,add_special_tokens=False).ids
            if (not ids or 1+len(ids)+len(continuation)>cap
                    or tokenizer.decode(ids,skip_special_tokens=False)!=prefix):return None
            maximum=max(maximum,len(ids))
        return prefix,source,maximum
    starts=[0]+[i for i in range(1,len(raw)) if raw[i-1].isspace() and not raw[i].isspace()]
    result=check(0)
    if result:return result
    low,high=1,len(starts)-1
    while low<high:
        middle=(low+high)//2
        if check(starts[middle]) is None:low=middle+1
        else:high=middle
    for index in range(low,len(starts)):
        result=check(starts[index])
        if result:return result
    # A long final word may need a Unicode-character cut; never change documents.
    for start in range(max(0,len(raw)-cap),len(raw)):
        result=check(start)
        if result:return result
    raise ValueError('No shared nonempty context; do not drop this document')


def summarize(plan, results, replicates=5000):
    import numpy as np
    rng=np.random.default_rng(SEED)
    pairs=plan['pairs'];steps=plan['steps'];D=plan['documents']
    byte_counts=np.array(results[str(steps[0])]['bytes'],dtype=float)
    weights=np.array([p['stratum_weight']/2 for p in pairs])
    if len(byte_counts)!=D or not np.isclose(weights.sum(),1):raise ValueError('Invalid cohort or weights')
    denom=byte_counts.sum()*np.log(2)
    diff={};rows=[]
    for step in steps:
        r=results[str(step)]
        if r['bytes']!=byte_counts.tolist():raise ValueError('Scored bytes changed across checkpoints')
        seen=np.array(r['seen_nats']);fresh=np.array(r['fresh_nats'])
        if seen.shape!=(len(pairs),D) or fresh.shape!=seen.shape:raise ValueError('Missing pair scores')
        diff[step]=fresh-seen
        can=float(np.sum(r['canonical_nats'])/denom)
        s=float(np.sum(seen*weights[:,None])/denom);f=float(np.sum(fresh*weights[:,None])/denom)
        rows.append({'step':step,'seen_bpb':s,'fresh_bpb':f,'canonical_bpb':can,
                     'seen_gap_bpb':s-can,'fresh_gap_bpb':f-can,'fresh_minus_seen_bpb':f-s})
    samples={step:[] for step in steps};growth=[]
    for _ in range(replicates):
        chosen=np.concatenate([2*h+rng.integers(2,size=2) for h in range(9)])
        docs=rng.integers(D,size=D);den=byte_counts[docs].sum()*np.log(2)
        vals={s:float(np.sum(diff[s][chosen][:,docs]*weights[:,None])/den) for s in steps}
        for s in steps:samples[s].append(vals[s])
        growth.append(vals[12000]-vals[6000])
    for row in rows:row['fresh_minus_seen_ci95']=np.quantile(samples[row['step']],[.025,.975]).tolist()
    change=rows[-1]['fresh_minus_seen_bpb']-rows[1]['fresh_minus_seen_bpb']
    interval=np.quantile(growth,[.025,.975]).tolist()
    return {'rows':rows,'gap_growth_6k_to_12k':change,'gap_growth_ci95':interval,
            'bootstrap_replicates':replicates,
            'interpretation':'Positive growth means the fresh-tokenizer penalty increased; CI overlap with zero is inconclusive, not proof of equivalence.'}
