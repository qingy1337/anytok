"""Plot audited full-data timings, showing missing pairs explicitly.

    python plotting_scripts/plot_tokenizer_speed.py --analysis PATH

Consumes the metadata from scripts/analyze_tokenizer_speed.py. It never reads
training text or tokenizer files. Partial results remain labelled provisional.
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BLUE, GREEN, PURPLE = '#2F8FD0', '#36967B', '#8657BC'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT/'plots/tokenizer_speed')
    args = parser.parse_args()
    report = json.loads(args.analysis.read_text())
    if report.get('run_id', '').endswith('-smoke'):
        raise ValueError('Smoke tests are not performance evidence')
    for font in font_manager.findSystemFonts([str(Path.home()/'.local/share/fonts/manrope')]):
        font_manager.fontManager.addfont(font)
    plt.rcParams.update({'font.family':'Manrope', 'font.size':9,
                         'axes.spines.top':False, 'axes.spines.right':False,
                         'axes.labelcolor':'#333333', 'text.color':'#333333',
                         'pdf.fonttype':42, 'ps.fonttype':42})
    keys = [a+'-'+p for a in ['bpe','wordpiece','unigram'] for p in ['regex','line256','no-regex']]
    rows = {r['case']:r for r in report['rows']}
    replay = {r['case'] for r in report.get('unmatched_pairs',[])}
    labels = [k.replace('wordpiece','WordPiece').replace('unigram','Unigram').replace('bpe','BPE').replace('-line256',' / line 256').replace('-no-regex',' / no regex').replace('-regex',' / regex') for k in keys]
    y = np.arange(len(keys))
    fig, axes = plt.subplots(1,2,figsize=(11,5.8),gridspec_kw={'width_ratios':[1.2,1]})
    fig.subplots_adjust(left=.21,right=.96,top=.79,bottom=.27,wspace=.2)
    time_ax, speed_ax = axes
    for i,key in enumerate(keys):
        if key not in rows:
            time_ax.text(.03,i,'Awaiting replay' if key in replay else 'Pair pending',transform=time_ax.get_yaxis_transform(),va='center',color='#8A929B',fontsize=8)
            continue
        row = rows[key]
        time_ax.plot([row['baseline_seconds']/60,row['fast_seconds']/60],[i,i],color='#C6CED5',lw=2)
        time_ax.scatter(row['baseline_seconds']/60,i,s=35,color=BLUE,label='Original' if key==next(iter(rows)) else None,zorder=3)
        time_ax.scatter(row['fast_seconds']/60,i,s=35,color=GREEN,label='Optimized' if key==next(iter(rows)) else None,zorder=3)
        speed_ax.barh(i,row['speedup'],height=.48,color=PURPLE)
        speed_ax.text(row['speedup'],i,f"  {row['speedup']:.2f}×",va='center',fontsize=8)
    time_ax.set_xscale('log'); time_ax.set_xlabel('Ready-to-use generation time (minutes, log scale)')
    time_ax.set_yticks(y,labels); speed_ax.set_yticks(y,[])
    speed_ax.set_xlabel('Original time / optimized time')
    speed_ax.axvline(1,color='#73808E',lw=.8)
    speed_ax.axvline(5,color=GREEN,lw=1,ls='--',label='5× target')
    speed_ax.set_xlim(0,max(6,max((r['speedup'] for r in rows.values()),default=0)*1.25))
    for ax in axes:
        ax.set_ylim(len(keys)-.4,-.6); ax.grid(axis='x',color='#DCE1E5',lw=.5); ax.set_axisbelow(True)
    time_ax.legend(frameon=False,loc='upper right'); speed_ax.legend(frameon=False,loc='upper right')
    complete = report.get('matched_complete',report['complete'])
    title = 'Tokenizer generation · matched 200 MB benchmark'
    if not complete: title += ' (provisional)'
    total = report['total']['ready_speedup']
    summary = f"Aggregate: {total:.2f}× across {len(rows)}/9 matched pairs" if total is not None else 'No completed timing pairs'
    fig.text(.21,.94,title,fontsize=14,weight='bold')
    fig.text(.21,.885,summary,fontsize=10,color=PURPLE)
    fig.text(.21,.84,'Same source documents, settings and four Rayon threads within each pair',fontsize=9,color='#56616D')
    fig.text(.21,.07,'Time includes training, serialization and existing validation; extended Unicode audit is reported separately.\nAggregate is a ratio of summed times for nine equally weighted cells, not a minimum speedup per tokenizer.\n'+('All timing pairs complete.' if complete else 'Missing pairs are excluded. No full-workload speedup claim is established.'),fontsize=8,color='#56616D')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    for suffix in ['png','pdf']: fig.savefig(args.output.with_suffix('.'+suffix),dpi=190,facecolor='white')
    plt.close(fig)
    print(args.output.with_suffix('.png'))


if __name__ == '__main__': main()
