import csv
from pathlib import Path
import json
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from package_training_inputs import sha256
import report_raw_bpb_rl_20k_tillicum as report


def sanity_rows(byte_change=None):
    return [dict(step=step, tokenizer=name, generated_bpb=1 + i / 10 + step / 1e5, canonical_bpb=1 - step / 1e6,
                 examples=92, bytes=1000 + i + (step == byte_change), truncated_forms=0)
            for step in report.SANITY_STEPS for i, name in enumerate(['T1', 'T2'], start=1)]


def test_percent_rounds_exact_halves_up():
    assert [report.percent(605, 1000), report.percent(4895, 20000), report.percent(1, 3),
            report.percent(2, 3), report.percent(0, 7)] == [61, 24, 33, 67, 0]


def test_sanity_means_average_t1_and_t2():
    means = report.sanity_means(sanity_rows())
    assert [row['step'] for row in means] == report.SANITY_STEPS
    assert means[0]['transferred_bpb'] == pytest.approx(1.15) and means[0]['gap_bpb'] == pytest.approx(.15)
    assert means[-1]['transferred_bpb'] == pytest.approx(1.35) and means[-1]['canonical_bpb'] == pytest.approx(.98)


def test_sanity_means_reject_changed_examples():
    with pytest.raises(ValueError, match='Sanity examples'):
        report.sanity_means(sanity_rows(byte_change=500))


def test_most_picked_joins_neighbors_and_drops_weak_blocks():
    blocks = []
    for end in range(500, 20001, 500):
        tokenizer, count = (7, 100) if end <= 1000 else (7, 40) if end == 1500 else (9, 500)
        blocks.append(dict(from_step=end - 499, to_step=end, most_picked_tokenizer=tokenizer, most_picked_count=count,
                           bpe_picks=500, unigram_picks=0, wordpiece_picks=0))
    configs = [dict(index=i, algorithm='bpe', pretokenizer='no-regex', vocab_size=100000) for i in range(1000)]
    ranks = {i: dict(regret_rank=i + 1, raw_bpb_rank=1000 - i) for i in range(1000)}
    rows = report.most_picked(blocks, configs, ranks)
    assert [(r['from_step'], r['to_step'], r['tokenizer']) for r in rows] == [(1, 1000, 7), (1001, 1500, None),
                                                                              (1501, 20000, 9)]
    assert rows[0]['share'] == .2 and rows[1]['share'] is None and rows[2]['share'] == 1
    assert (rows[0]['regret_rank'], rows[0]['raw_bpb_rank']) == (8, 993)


def test_concentration_windows_allow_single_dips_and_skip_short_runs():
    policy = [dict(step=step, policy_max_probability=.99 if (1000 <= step <= 2000 and step != 1500)
                   or 3000 <= step <= 3200 else .5) for step in range(0, 4001, 50)]
    assert report.concentration_windows(policy) == [dict(from_step=1000, to_step=2000, samples=21, samples_at_or_above=20)]


def test_committed_report_matches_its_inputs():
    tables = json.loads((report.REPORT / 'run-tables.json').read_text())['tables']
    status = json.loads((report.REPORT / 'status.json').read_text())
    run_config = json.loads((report.REPORT / 'run_config.json').read_text())
    pool = json.loads(report.POOL.read_text())
    verification = json.loads(report.VERIFICATION.read_text())
    four_way = json.loads(report.FOUR_WAY.read_text())
    with report.RANKING.open() as ranking, report.UNIFORM_SANITY.open() as uniform:
        result = report.analyze(tables, status, run_config, pool, verification, csv.DictReader(ranking),
                                list(csv.DictReader(uniform)), four_way)
    assert json.loads(json.dumps(result)) == json.loads((report.REPORT / 'analysis.json').read_text())
    assert result['sanity']['step0_reproduces_handoff_verification']
    assert result['sanity']['transferred_bpb']['start'] == pytest.approx(1.4113, abs=1e-4)
    assert result['policy']['distinct_tokenizers'] == 669 and result['replay']['first_update_tokenizer_differs'] == 4091


def test_manifest_lists_every_report_file():
    manifest = json.loads((report.REPORT / 'manifest.json').read_text())['files']
    files = {str(p.relative_to(report.REPORT)) for p in report.REPORT.rglob('*') if p.is_file() and p.name != 'manifest.json'}
    assert set(manifest) == files
    for name, entry in manifest.items():
        assert entry == dict(bytes=(report.REPORT / name).stat().st_size, sha256=sha256(report.REPORT / name))
