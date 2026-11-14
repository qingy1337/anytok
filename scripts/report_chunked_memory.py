"""Report the exact-artifact chunking study, including rejected candidates."""
import hashlib
import json
from pathlib import Path
import shutil

from pool_generalization import file_sha,save_json

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'reports/chunked-memory'
STATE=ROOT/'modal_results/chunked_memory'


def seconds(r):return sum(p['seconds'] for p in r['phases'] if not p['phase'].startswith('load '))


def main():
    rows=[]
    for p in sorted(REPORT.glob('*.json')):
        if p.name in ['results.json','manifest.json','environment.json','forward-equivalence-128k.json']:continue
        result=json.loads(p.read_text())
        if 'outcome' in result:rows.append(dict(id=p.stem,result=result))
    old=json.loads((ROOT/'reports/local-memory/results.json').read_text())
    data={row['id']:row['result'] for row in old+rows}
    sources=REPORT/'source_versions';sources.mkdir(exist_ok=True)
    for name in ['chunked_hypernetwork','lossless_layer_norm','lossless_mlp','local_memory_trial','local_memory_variants','run_chunked_memory']:
        source=ROOT/'scripts'/f'{name}.py';shutil.copyfile(source,sources/(file_sha(source)+'.py'))
    save_json(REPORT/'results.json',rows)
    candidate43=data.get('norm-source-v2-652-c16384')
    complete128=[r for r in rows if r['result'].get('inputs',{}).get('vocabulary')==128000 and r['result']['outcome']=='complete' and not r['id'].startswith('chunked-')]
    best128=next((r for r in complete128 if r['id']=='validated-staged-idle-edge-989-c16384'),None)
    if best128 is None:best128=min(complete128,key=lambda r:seconds(r['result'])) if complete128 else None
    fast43='validated-gpu-652-c16384' if 'validated-gpu-652-c16384' in data else 'norm-mlp-652-c16384'
    lines=['# Fast, lossless memory diagnostics','',
        'The broad whole-entry chunking prototype failed the losslessness checks and was rejected. A narrower implementation chunks source-embedding preparation and normalization, preserving the original full matrix operations, global vocabulary attention, native dropout and affine-parameter backward reductions. The running Modal trainer is unchanged.','',
        '## Measured results','',
        '| Workload | Implementation | Outcome | Peak allocated GPU GiB | Timed training phases |',
        '|---|---|---|---:|---:|']
    chosen=[('baseline-43k','Original'),(fast43,'Narrow memory changes, GPU only'),('split-disk-fixed-128k-host80-optimizer-expand','Previous CPU/NVMe strategy')]
    if best128:chosen.append((best128['id'],'Narrow memory changes, selected CPU staging'))
    for name,label in chosen:
        if name not in data:continue
        r=data[name];lines.append(f"| {r['inputs']['vocabulary']:,} | {label} | {r['outcome']} | {r['peak_allocated_bytes']/2**30:.2f} | {seconds(r):.2f}s |")
    lines+=['','![Memory, speed and correctness](chunked_memory.png)','','[PDF plot](chunked_memory.pdf) · [Every attempt](results.json)','',
        'Timings are one cold measured step per trial, including canonical scoring, complete forward/backward, clipping, optimizer update and any state transfers. They exclude loading, example preparation and numerical comparisons. They are not steady-state throughput claims. Allocated peaks exclude unused allocator reservations; reserved peaks are retained in raw results. An OOM peak is an incomplete measurement.','',
        '## What “lossless” means here','',
        'The full 43k comparisons use every generated embedding, both BPB losses, lexical loss, final CUDA RNG and all 349 gradient-bearing LM/hypernetwork parameters. Accepted candidates have bit-identical outputs, losses and RNG. Their gradient differences are approximately 1.3–1.4e-8 relative L2, comparable to the unmodified baseline repeated against itself (1.35e-8). Gradients are not claimed bit-identical.','',
        'The complete 128k comparison also checks every embedding value, both losses, RNG and all 349 parameter gradients. The validated candidate matches outputs/losses/RNG exactly and gradients at 8.21e-9 relative L2 (maximum absolute difference 2.85e-9, in the token-type embedding accumulator). Its reference uses the original full-shaped operators with the previously verified checkpoint/CPU/NVMe offloading wrapper. This is not an unmodified single-GPU baseline: the unmodified 128k training graph does not fit locally. Native-reference provenance and every phase are retained in native-reference-989-c16384.json. A separate no-grad full-table check against the unmodified original forward is saved as forward-equivalence-128k.json.','',
        '## How the narrower implementation works','',
        '1. Construct source embeddings in chunks directly in the dtype that the original first matrix operation consumes. All original source scaling and fallback values are preserved. Keep one full native fallback-embedding gradient scatter.','2. Run native FP32 normalization on row chunks, saving the exact original BF16 inputs instead of larger FP32 copies. Reconstruct those FP32 values in backward.','3. Keep the original full-shape normalization weight/bias reduction. Compute the independent input-gradient rows separately, avoiding a full FP32 input-gradient allocation.','4. Combine outer layer and inner operation checkpointing. Matrix multiplications and global vocabulary attention retain their original full shapes. Native dropout is unchanged.','5. The MLP recomputes its original full-shaped operations, preserves the original native parameter-gradient reductions, and reuses its private GELU-gradient buffer after its last use. No matrix reduction is chunked.', '6. The validated 128k variant stages inactive Adam tensors (13.82 GiB), completed LM gradients (6.00 GiB) and three immutable normalization inputs (16.11 GiB peak) on CPU. It restores all gradients before clipping and all Adam state before the unchanged fused update. It performs no activation spill to disk.', '7. After lexical loss and the LM embedding gradient are computed, the final generated-output buffer is no longer needed. Use PyTorch’s GradientEdge to retain its backward connection while releasing its 1 GiB storage. The diagnostic retains a CPU copy solely for numerical comparison, outside the timed training phases.','',
        '**This reduces the coefficient of vocabulary-dependent memory; total peak memory is still not constant in vocabulary size.** Complete layer states and some native full-shape backward operands remain. Fully streaming every operation without changing mixed-precision arithmetic has not been achieved.','',
        '## Rejected broad chunking','',
        'Whole-entry chunking retained all vocabulary entries and the original global latent attention. Packed full-shape native dropout masks reproduced the original masks and RNG advancement. Nevertheless the real 43k outputs differed by up to 0.02812 and gradients by 0.3897 relative L2. Its 128k step completed at 76.36 GiB and 20.93 seconds, but this result is rejected as lossless. These measurements remain in results.json, and its source snapshot is retained. No thresholds were relaxed to accept it.','',
        '## Fidelity and storage','',
        'Use the same authorized local copy as the [earlier memory study](../local-memory/README.md): exact step-2,000 trained LM/hypernetwork, populated optimizer, policy, corpus and RNG state, with the original complete 1,000-tokenizer pool and corpora hash-verified. Memory trials force native tokenizer #652 (43k) or #989 (128k), using the original complete batch 128, microbatch 32, variable-length continuations, paired boundaries, maximum 128 LM positions, canonical output head and lexical weight 0.5. These are memory diagnostics on complete native vocabularies, not a tokenizer-quality sweep. Cloud originals remain authoritative.','',
        '35 CUDA/unit checks and 13 dashboard checks passed; four expected failures retain the rejected broad prototype. Focused checks cover exact native-width normalization and MLP outputs/input/parameter gradients, full-model equivalence with/without dropout and BF16 autocast, and CPU-staged normalization. The broader rejected prototype has explicit expected-failure tests; its failures are retained rather than accepted with wider tolerances.','',
        '## All attempts','', '| Trial | Outcome | Peak allocated GiB | Seconds (partial for OOM) | Failure phase |','|---|---|---:|---:|---|']
    for row in rows:
        r=row['result'];lines.append(f"| {row['id']} | {r['outcome']} | {r['peak_allocated_bytes']/2**30:.2f} | {seconds(r):.2f} | {r.get('phase','')} |")
    lines+=['','## Reproduce','','Use `.venv-memory/bin/python` (PyTorch 2.15.0+cu132). The artifact layout and source hashes are retained in the linked original study and individual trial records. Historical candidate-operator implementations are preserved in source_versions/ by SHA-256. The final commands below use the current harness.','','```bash','.venv-memory/bin/python -m pytest tests/test_lossless_layer_norm.py tests/test_chunked_hypernetwork.py tests/test_lossless_mlp.py tests/test_local_memory_variants.py -q','.venv-memory/bin/python scripts/run_chunked_memory.py --variant chunked_norm_mlp --label validated-gpu --tokenizers 652 --direct-embedding-backward','.venv-memory/bin/python scripts/run_chunked_memory.py --variant split_norm_disk --label native-reference --tokenizers 989 --offload-optimizer --host-budget-gib 80 --reference-128 modal_results/chunked_memory/reference-128 --write-reference-128', '.venv-memory/bin/python scripts/run_chunked_memory.py --variant chunked_norm_staged --label validated-staged-idle-edge --tokenizers 989 --direct-embedding-backward --offload-optimizer --offload-lm-gradients --release-generated-output --reference-128 modal_results/chunked_memory/reference-128', '.venv-memory/bin/python scripts/verify_local_memory_forward.py --candidate chunked_norm_mlp','.venv-memory/bin/python scripts/report_chunked_memory.py','```','','The controller defaults to complete 43k and 128k trials sequentially; the commands above specify each native vocabulary explicitly. They do not modify or resume production training. Large corpus, tokenizer, checkpoint, gradient and activation files are excluded from Git.','']
    (REPORT/'README.md').write_text('\n'.join(lines))
    plot(data,best128,fast43)
    save_json(REPORT/'manifest.json',dict(files={str(p.relative_to(REPORT)):dict(bytes=p.stat().st_size,sha256=file_sha(p))
        for p in sorted(REPORT.rglob('*')) if p.is_file() and p.name!='manifest.json'}))


def plot(data,best128,fast43):
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_manager.fontManager.addfont(str(ROOT/'monitoring/web/Manrope.ttf'))
    plt.rcParams.update({'font.family':'Manrope','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(2,2,figsize=(11,7.3),constrained_layout=True)
    selected=[('baseline-43k','Original'),('norm-nested-652-c16384','Norm only'),(fast43,'Norm + source + MLP')]
    selected=[(n,l) for n,l in selected if n in data]
    for i,(name,label) in enumerate(selected):
        r=data[name]
        axes[0,0].bar(i,r['peak_allocated_bytes']/2**30,color='#36967B')
        axes[0,0].text(i,r['peak_allocated_bytes']/2**30+1,f"{r['peak_allocated_bytes']/2**30:.1f}",ha='center')
        axes[0,1].bar(i,seconds(r),color='#36967B')
        axes[0,1].text(i,seconds(r)+.1,f'{seconds(r):.2f}s',ha='center')
    for ax in axes[0]:ax.set_xticks(range(len(selected)),[l for _,l in selected])
    axes[0,0].set(title='43k complete step · allocated GPU memory',ylabel='GiB',ylim=(0,100))
    axes[0,1].set(title='43k complete step · timed training phases',ylabel='Seconds',ylim=(0,6))
    names=[('split-disk-fixed-128k-host80-optimizer-expand','Previous CPU/NVMe')]
    if best128:names.append((best128['id'],'Selected CPU staging'))
    for i,(name,label) in enumerate(names):
        r=data[name];axes[1,0].barh(i,seconds(r),color='#8657BC')
        axes[1,0].text(seconds(r)+2,i,f"{seconds(r):.1f}s · {r['peak_allocated_bytes']/2**30:.1f} GiB",va='center',fontsize=8)
    axes[1,0].set_yticks(range(len(names)),[l for _,l in names]);axes[1,0].invert_yaxis()
    axes[1,0].set(title='128k complete steps · timed training phases',xlabel='Seconds',xlim=(0,190))
    ax=axes[1,1];ax.axis('off');table=[]
    for name,label in [('repeat-43k','Original repeated'),*selected[1:]]:
        n=data[name]['numeric'];g=n.get('gradients',{})
        table.append([label,str(n.get('output_bitwise_equal')),f"{g.get('relative_l2',float('nan')):.2g}"])
    if best128:
        n=best128['result']['numeric'];g=n.get('gradients',{})
        table.append(['128k validated',str(n.get('output_bitwise_equal')),f"{g.get('relative_l2',float('nan')):.2g}"])
    ax.set_title('Complete outputs and all 349 parameter gradients')
    t=ax.table(cellText=table,colLabels=['Variant','Output exact','Gradient ΔL2'],bbox=[0,.35,1,.45],cellLoc='center',colWidths=[.42,.28,.30])
    t.auto_set_font_size(False);t.set_fontsize(9)
    for c in t.get_celld().values():c.set_edgecolor('#E1E6EB')
    ax.text(.5,.08,'128k reference: original operators with checkpoint/offload wrapper.\nBroad whole-entry chunking rejected for numerical differences.\nTotal memory still scales with vocabulary size.',ha='center',va='bottom',transform=ax.transAxes,fontsize=8)
    for ax in [axes[0,0],axes[0,1],axes[1,0]]:ax.grid(axis='x' if ax is axes[1,0] else 'y',color='#E9EDF1');ax.set_axisbelow(True)
    fig.suptitle('Lossless chunking diagnostics · exact checkpoint · batch 128 · microbatch 32',fontsize=13)
    for folder in [REPORT,ROOT/'plots']:
        for ext in ['png','pdf']:
            destination=folder/f'chunked_memory.{ext}';tmp=destination.with_name(destination.stem+f'.tmp.{ext}')
            fig.savefig(tmp,dpi=170,facecolor='white');tmp.replace(destination)
    plt.close(fig)


if __name__=='__main__':main()
