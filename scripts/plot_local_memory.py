"""Memory and runtime measurements, retaining failed full-vocabulary cases."""
from pathlib import Path


def plot(rows, path):
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    root=Path(__file__).resolve().parents[1];font=root/'monitoring/web/Manrope.ttf'
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family':'Manrope','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    cases=[r for r in rows if r['id']!='repeat-43k']
    fig,axes=plt.subplots(2,2,figsize=(11,7.2),constrained_layout=True)
    capacity=next((r['result']['inputs']['gpu_total_bytes']/2**30 for r in rows if 'inputs' in r['result']),95.59)
    names={'baseline':'Original','offload':'Offload','fine':'Finer checkpoints','fine_offload':'Both'}
    for i,row in enumerate(cases):
        r=row['result'];name=row['id'];large='128k' in name
        variant=r.get('inputs',{}).get('variant',name.split('-128k')[0].replace('-','_'))
        label=names.get(variant,variant)+'\n'+('128k' if large else '43k')
        color='#8657BC' if large else '#36967B';failed=r['outcome']!='complete'
        height=r.get('peak_allocated_bytes',0)/2**30
        axes[0,0].bar(i,height,color='white' if failed else color,edgecolor='#BB675A' if failed else color,hatch='///' if failed else None)
        if failed:axes[0,0].text(i,height+2,'OOM',ha='center',fontsize=8,color='#A85349')
        axes[0,0].set_xticks(range(len(cases)),[names.get(x['result'].get('inputs',{}).get('variant',x['id'].split('-128k')[0].replace('-','_')),x['id'].split('-')[0])+'\n'+('128k' if '128k' in x['id'] else '43k') for x in cases],rotation=25,ha='right',fontsize=7)
        if not failed:
            seconds=sum(p['seconds'] for p in r['phases'] if not p['phase'].startswith('load '))
            axes[0,1].bar(i,seconds,color=color)
        else:axes[0,1].text(i,0,'OOM',ha='center',va='bottom',fontsize=8,color='#A85349')
    axes[0,0].axhline(capacity,color='#89919C',ls=':',label=f'Local GPU: {capacity:.1f} GiB')
    axes[0,0].set(title='Complete native vocabulary · peak allocated memory',ylabel='GiB',ylim=(0,max(capacity+15,110)))
    axes[0,0].legend(frameon=False,fontsize=8)
    axes[0,1].set(title='Measured training phases · complete trials only',ylabel='Seconds · excludes loading and correctness checks')
    axes[0,1].set_xticks(axes[0,0].get_xticks(),[t.get_text() for t in axes[0,0].get_xticklabels()],rotation=25,ha='right',fontsize=7)
    phase_names=['hypernetwork forward','four LM microbatches forward and backward','hypernetwork backward','gradient clipping and complete optimizer update']
    for row in cases:
        if '128k' not in row['id']:continue
        r=row['result'];phases={p['phase']:p['peak_allocated_bytes']/2**30 for p in r.get('phases',[])}
        axes[1,0].plot(range(4),[phases.get(p,float('nan')) for p in phase_names],marker='o',label=row['id'].replace('-128k',''))
    axes[1,0].set(title='128k trials · memory by phase (including partial trials)',ylabel='Peak allocated GiB')
    axes[1,0].set_xticks(range(4),['HN forward','LM + backward','HN backward','Optimizer'],rotation=12,ha='right')
    if any('128k' in r['id'] for r in cases):axes[1,0].legend(frameon=False,fontsize=8)
    axes[1,1].axis('off');table=[]
    for row in rows:
        r=row['result'];numeric=r.get('numeric',{});g=numeric.get('gradients',{})
        if row['id']=='baseline-43k' or '128k' in row['id'] or not numeric:continue
        table.append([row['id'].replace('-43k',''),str(numeric.get('output_bitwise_equal')),
            f"{g.get('relative_l2',float('nan')):.2g}",str(numeric.get('cuda_rng_sha256')==numeric.get('reference_numeric',{}).get('cuda_rng_sha256'))])
    axes[1,1].set_title('43k correctness · full outputs and all parameter gradients',pad=14)
    if table:
        t=axes[1,1].table(cellText=table,colLabels=['Variant','Output exact','Gradient ΔL2','RNG exact'],loc='center',cellLoc='center')
        t.auto_set_font_size(False);t.set_fontsize(8);t.scale(1,1.6)
        for cell in t.get_celld().values():cell.set_edgecolor('#E1E6EB')
    else:axes[1,1].text(.5,.5,'Waiting for complete paired checks',ha='center',va='center')
    for ax in [axes[0,0],axes[0,1],axes[1,0]]:ax.grid(axis='y',color='#E9EDF1',lw=.6);ax.set_axisbelow(True)
    fig.suptitle('Local memory diagnostics · real checkpoint + optimizer · batch 128 · microbatch 32',fontsize=13)
    path.parent.mkdir(parents=True,exist_ok=True)
    for suffix in ['png','pdf']:
        target=path.with_suffix('.'+suffix);temp=target.with_name(target.stem+'.tmp.'+suffix)
        fig.savefig(temp,dpi=170,facecolor='white');temp.replace(target)
    plt.close(fig)
