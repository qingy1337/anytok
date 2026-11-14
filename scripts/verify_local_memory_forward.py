"""Full-128k forward equivalence check; not a training-memory benchmark."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from local_memory_trial import load_trainer, STATE, ROOT
from local_memory_variants import apply_variant
from pool_generalization import save_json,file_sha


def status(stage,done):
    row=dict(at_utc=datetime.now(timezone.utc).isoformat(),stage=stage,completed=done,total=27,
        platform='Local · RTX PRO 6000 96 GB',workers=[] if stage in ['complete','failed'] else
        [dict(id='memory-gpu',profile='local',platform='Local · RTX PRO 6000 96 GB',state=stage)])
    save_json(STATE/'status.json',row)
    with (STATE/'events.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
    print(json.dumps(row),flush=True)


def main():
    if torch.__version__.split('+')[0] != '2.15.0':
        raise ValueError('Use .venv-memory/bin/python to match the training runtime')
    status('checking full 128k hypernetwork output equivalence',26)
    torch.set_num_threads(1)
    trainer=load_trainer();artifacts=STATE/'artifacts';checkpoint=artifacts/'checkpoint'
    canonical=trainer.AutoTokenizer.from_pretrained(artifacts/'input/model',local_files_only=True)
    weights=torch.load(checkpoint/'lm.pt',map_location='cpu',weights_only=True,mmap=True)
    source=weights['model.embed_tokens.weight'].to('cuda')
    hn=trainer.ZettHypernet.from_pretrained(checkpoint/'hypernet',local_files_only=True).to('cuda')
    fallback=trainer.enable_byte_fallbacks(hn,canonical);trainer.freeze_rescalers(hn);hn.train()
    hn.model.activation_checkpointing=True;hn.activation_checkpoint_input=True
    pool=trainer.TokenizerPool(artifacts/'input/pool');_,vocab=pool.load(989)
    forms_np,truncated=pool.forms(989,canonical,fallback)
    forms=torch.from_numpy(forms_np.astype(np.int64)).to('cuda')
    priors=torch.tensor([score for _,score in vocab],device='cuda',dtype=torch.float32)
    state=torch.load(checkpoint/'state.pt',map_location='cpu',weights_only=True)
    rng=state['cuda_rng'][0];records=[];outputs=[]
    for name in ['original','optimized']:
        if name=='optimized':apply_variant(hn,'split_norm_disk')
        torch.cuda.set_rng_state(rng)
        # Keep training-mode dropout, the complete vocabulary, all ten source
        # pieces and target-prior token. Only autograd storage is disabled.
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            generated,output,_=hn(forms,priors,source,None,deterministic=False)
        if output is not None:raise ValueError('Unexpected output embeddings')
        actual=generated.detach().cpu();del generated
        outputs.append(actual)
        records.append(dict(name=name,output_sha256=hashlib.sha256(actual.numpy().tobytes()).hexdigest(),
            cuda_rng_sha256=hashlib.sha256(torch.cuda.get_rng_state().numpy().tobytes()).hexdigest()))
    result=dict(kind='numerical_forward_validation_only',vocabulary=len(vocab),tokenizer_index=989,
        elements_compared=outputs[0].numel(),shape=list(outputs[0].shape),checkpoint_step=state['step'],
        torch=torch.__version__,variant_sha256=file_sha(ROOT/'scripts/local_memory_variants.py'),
        output_bitwise_equal=torch.equal(*outputs),maximum_absolute_difference=float((outputs[0]-outputs[1]).abs().max()),
        rng_equal=records[0]['cuda_rng_sha256']==records[1]['cuda_rng_sha256'],records=records,
        scope='Complete real 128k hypernetwork output table, with training-mode dropout and identical RNG. Both passes disable autograd. This does not compare 128k gradients or establish full-training memory usage.')
    save_json(STATE/'forward-equivalence-128k.json',result)
    if not result['output_bitwise_equal'] or not result['rng_equal']:raise ValueError('Original and optimized 128k forwards differ')
    print(json.dumps(result),flush=True);status('complete',27)


if __name__=='__main__':
    try:main()
    except Exception:
        status('failed',26)
        raise
