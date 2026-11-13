"""Explain prefix dependence and tokenizer transfer using one saved real passage."""
import argparse
import gzip
import json
import math
from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]


def example(source):
    data = json.load(gzip.open(source / 'results.json.gz', 'rt'))
    analysis = json.loads((source / 'analysis.json').read_text())
    passage = next(r for r in json.loads((source / 'examples.json').read_text()) if r['domain'] == 'web-en')
    index, meta = next((i, r) for i, r in enumerate(data['preparation']['metadata']) if r['id'] == passage['id'])
    native = next(r for r in data['results'] if r['index'] == 771)
    denominator = meta['continuation_bytes'] * math.log(2)
    canonical = math.fsum(data['canonical'][index]) / denominator
    removed = math.fsum(data['no_prefix'][index]) / denominator
    transferred = math.fsum(native['transferred'][index]) / denominator
    return dict(id=passage['id'], prefix=passage['prefix'], suffix=passage['continuation'],
        canonical_prefix_tokens=len(meta['canonical_prefix_ids']), native_prefix_tokens=meta['native_prefix_lengths']['771'],
        suffix_tokens=32, suffix_bytes=meta['continuation_bytes'], canonical_bpb=canonical,
        no_prefix_bpb=removed, transferred_bpb=transferred, context_benefit=removed - canonical,
        context_threshold=analysis['context_thresholds']['web-en'], transfer_gap=transferred - canonical,
        tokenizer_index=771, native_config=native['config'], position_cap=128,
        selection_uses='Canonical with-prefix and canonical BOS-only losses; no transferred losses.',
        teacher_forcing='For each scored suffix token, all earlier gold suffix tokens are inputs. Future tokens are masked. The prefix and BOS are never scored.')


def draw(data, destination):
    font = ROOT / 'monitoring/web/Manrope.ttf'
    if font.exists():
        font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family': 'Manrope' if font.exists() else 'DejaVu Sans',
                         'font.size': 11, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    fig = plt.figure(figsize=(12, 13.4), facecolor='white')
    ax = fig.add_axes([0, 0, 1, 1]);ax.set(xlim=(0, 1), ylim=(0, 1));ax.axis('off')
    blue, purple, green = '#2F8FD0', '#8657BC', '#267F66'
    ink, muted = '#242B35', '#596574'
    prefix_bg, target_bg = '#EAF3FA', '#FFF2CE'
    def text(x, y, value, size=11, color=ink, weight='normal', **kwargs):
        ax.text(x, y, value, fontsize=size, color=color, weight=weight, va='top', **kwargs)
    def box(x, y, w, h, color, edge='#D8E0E8'):
        ax.add_patch(FancyBboxPatch((x, y - h), w, h, boxstyle='round,pad=0.008,rounding_size=0.008',
            facecolor=color, edgecolor=edge, linewidth=.9))
    def wrap(value, width):
        return '\n'.join('\n'.join(textwrap.wrap(line, width=width, replace_whitespace=False,
            drop_whitespace=False)) if line else '' for line in value.split('\n'))
    def sequence(y, name, prefix_kind, score, total):
        text(.055, y, name, 11, weight='bold')
        by = y - .028
        box(.055, by, .055, .035, '#F0F2F5');text(.0825, by - .009, 'BOS', 9, ha='center')
        if prefix_kind:
            label, color = prefix_kind
            box(.126, by, .35, .035, color);text(.301, by - .009, label, 10, ha='center')
            sx = .492
        else:
            sx = .126
        box(sx, by, .30, .035, target_bg, '#E3BB5C')
        text(sx + .15, by - .009, 'SAME 32 suffix tokens · SCORED', 9.5, ha='center', weight='bold')
        text(.945, by - .003, f'{score:.3f} BPB', 13, weight='bold', ha='right')
        text(.945, by - .022, f'{total} total positions', 8.5, muted, ha='right')
    text(.05, .978, 'What gets scored—and what makes a passage context-dependent?', 18, weight='bold')
    text(.05, .949, 'One actual English-web passage · these are single-passage scores · maximum 128 positions', 10.5, muted)
    text(.05, .931, 'A, B and C use the same trained language model and the same fixed canonical output head.', 10, muted)

    text(.05, .913, 'The text is split once. The scored suffix stays identical in all three runs.', 12, weight='bold')
    box(.05, .883, .54, .148, prefix_bg)
    text(.063, .873, 'PREFIX · visible input · never scored', 10, blue, 'bold')
    text(.063, .851, wrap(data['prefix'], 65), 9.2, family='DejaVu Sans Mono', linespacing=1.5)
    box(.615, .883, .335, .148, target_bg, '#E3BB5C')
    text(.628, .873, 'SUFFIX · all 32 tokens are scored', 10, '#856115', 'bold')
    text(.628, .844, wrap(data['suffix'], 37), 9.2, family='DejaVu Sans Mono', linespacing=1.5)
    text(.628, .751, f'32 canonical tokens = {data["suffix_bytes"]} UTF-8 bytes', 9, muted)

    text(.05, .704, '1   Decide whether the passage depends on its prefix', 15, weight='bold')
    text(.05, .678, 'Use canonical input embeddings in both runs. Compare losses on the yellow suffix only.', 10.5, muted)
    sequence(.650, 'A · With the prefix', ('Canonical prefix · 64 tokens', prefix_bg), data['canonical_bpb'], 97)
    sequence(.560, 'B · Remove the entire prefix', None, data['no_prefix_bpb'], 33)
    box(.05, .473, .90, .061, '#EDF6F1', '#C2DED1')
    text(.065, .462, f'Prefix benefit = B − A = {data["no_prefix_bpb"]:.3f} − {data["canonical_bpb"]:.3f} = {data["context_benefit"]:.3f} BPB', 12, green, 'bold')
    text(.065, .438, f'{data["context_benefit"]:.3f} > {data["context_threshold"]:.3f} threshold → INCLUDE this passage in the context-dependent subset.', 10.5, green)
    text(.055, .397, 'The 0.555 threshold was fixed from discovery English-web passages (75th percentile of prefix benefit).', 9.5, muted)

    text(.05, .359, '2   Separately measure this tokenizer’s transfer penalty', 15, weight='bold')
    text(.05, .333, 'Use the same prefix text, encoded by tokenizer #0771. Keep the suffix canonical.', 10.5, muted)
    sequence(.305, 'C · Transferred input embeddings', ('Same prefix text · 40 native tokens', '#F0EBF7'), data['transferred_bpb'], 73)
    text(.055, .225, f'Transfer gap = C − A = {data["transferred_bpb"]:.3f} − {data["canonical_bpb"]:.3f} = +{data["transfer_gap"]:.3f} BPB', 12, purple, 'bold')
    text(.055, .201, 'C is used to measure transfer loss. It is NOT used to decide whether the text is context-dependent.', 10, purple)

    box(.05, .162, .90, .083, '#F5F6F8')
    text(.065, .152, 'How scoring works inside every yellow box', 11, weight='bold')
    text(.065, .131, 'Predict token 1 → supply the correct token 1 → predict token 2 → … → predict token 32.', 10)
    text(.065, .111, 'Each prediction sees earlier correct suffix tokens, even in B. Future suffix tokens are masked.', 10)
    text(.065, .091, 'Sum all 32 prediction losses in bits, then divide by the same 63 bytes. BOS and prefix add no loss.', 9.5, muted)
    text(.05, .057, 'For the table: all passages → natural-text column; passages passing step 1 → context-dependent column.', 10)
    text(.05, .037, 'Compute each tokenizer’s average gap on that set, then take the median across the 12 tokenizers in its row.', 9.7, muted)
    for suffix in ['png', 'pdf', 'svg']:
        fig.savefig(destination / f'context_eval.{suffix}', dpi=180, facecolor='white')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'reports/natural-transfer-20k')
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/context-eval-explainer')
    args = parser.parse_args();args.output.mkdir(parents=True, exist_ok=True)
    data = example(args.source)
    assert data['context_benefit'] > data['context_threshold']
    assert data['suffix_tokens'] == 32 and data['canonical_prefix_tokens'] == 64 and data['native_prefix_tokens'] == 40
    (args.output / 'example.json').write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
    draw(data, args.output)
    print(args.output / 'context_eval.png')


if __name__ == '__main__':
    main()
