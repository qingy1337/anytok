"""The regret continuation report: run alignment, picks, complete updates and stable hashes."""
import gzip
import hashlib
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import extract_regret_continuation as extract
from package_training_inputs import atomic_json
from report_regret_continuation import (complete_updates, favorites, manifest, picks, private_files, sanity,
                                        segment_picks, window_shares)


def check(run, step, gap1, gap2, bytes1=10421, bytes2=10575):
    return dict(run=run, step=step, t1_generated_bpb=1 + gap1, t1_canonical_bpb=1.0, t1_examples=92, t1_bytes=bytes1,
                t2_generated_bpb=1 + gap2, t2_canonical_bpb=1.0, t2_examples=92, t2_bytes=bytes2)


def test_sanity_pairs_the_runs_only_at_shared_steps_and_averages_the_two_check_tokenizers():
    evals = [check('regret', 2912, .3, .1), check('regret', 500, .2, .1), check('regret', 0, .4, .2),
             check('raw_bpb', 500, .1, .05), check('raw_bpb', 16409, .1, .1), check('raw_bpb', 0, .4, .2)]
    rows = sanity(evals)
    assert [row['step'] for row in rows] == [0, 500]
    assert rows[1]['regret_gap_1'] == pytest.approx(.2) and rows[1]['regret_gap_2'] == pytest.approx(.1)
    assert rows[1]['regret_mean_gap'] == pytest.approx(.15) and rows[1]['raw_bpb_mean_gap'] == pytest.approx(.075)
    assert rows[1]['regret_minus_raw_bpb'] == pytest.approx(.075) and rows[0]['regret_minus_raw_bpb'] == pytest.approx(0)


def test_sanity_refuses_a_repeated_step_or_different_check_text():
    with pytest.raises(ValueError):
        sanity([check('regret', 500, .2, .1), check('regret', 500, .2, .1), check('raw_bpb', 500, .1, .1)])
    with pytest.raises(ValueError):
        sanity([check('regret', 500, .2, .1), check('raw_bpb', 500, .1, .1, bytes1=10420)])


def test_the_twins_repeated_check_keeps_its_last_row_but_the_regret_log_may_not_repeat_one():
    results = lambda gap: [dict(generated_bpb=1 + gap, canonical_bpb=1.0, examples=92, bytes=10421)] * 2
    events = [dict(stage='initial_eval', results=results(.4)), dict(stage='eval', step=4500, results=results(.2)),
              dict(stage='train', step=4501), dict(stage='eval', step=4500, results=results(.1))]
    rows = extract.eval_rows(events, 'raw_bpb', keep_last=True)
    assert [row['step'] for row in rows] == [0, 4500] and rows[1]['t1_generated_bpb'] == pytest.approx(1.1)
    with pytest.raises(ValueError):
        extract.eval_rows(events, 'regret')


def test_segment_picks_rank_by_count_then_index_and_count_distinct_tokenizers():
    chosen = [7] * 5 + [3] * 3 + [9] * 3 + [1] + [5] * 8
    rows = [dict(step=step, tokenizer_pool_index=tokenizer) for step, tokenizer in enumerate(chosen, 1)]
    first, second = segment_picks(rows, size=12, top=3)
    assert (first['first'], first['last'], first['updates'], first['distinct']) == (1, 12, 12, 4)
    assert [(item['tokenizer'], item['picks']) for item in first['top']] == [(7, 5), (3, 3), (9, 3)]
    assert first['top'][0]['share'] == pytest.approx(5 / 12)
    assert (second['first'], second['last'], second['distinct']) == (13, 20, 1)
    assert second['top'] == [dict(tokenizer=5, picks=8, share=1.0)]
    assert picks(rows, 1, 20, top=2)['top'] == [dict(tokenizer=5, picks=8, share=.4), dict(tokenizer=7, picks=5, share=.25)]


def test_window_shares_count_every_other_tokenizer_as_others():
    rows = [dict(step=step, tokenizer_pool_index=tokenizer) for step, tokenizer in enumerate([1, 1, 2, 3, 4, 1], 1)]
    first, second = window_shares(rows, [1, 2], size=3)
    assert first == dict(first=1, last=3, shares=[2 / 3, 1 / 3], other=0.0)
    assert second == dict(first=4, last=6, shares=[1 / 3, 0.0], other=2 / 3)


def test_the_favorite_is_seen_only_where_the_pick_had_the_highest_probability():
    row = lambda step, tokenizer, selected, top: dict(step=step, tokenizer_pool_index=tokenizer,
                                                     policy_selected_probability=selected, policy_max_probability=top)
    seen, spells = favorites([row(1, 771, .9, .9), row(2, 5, .01, .9), row(3, 771, .91, .91),
                              row(4, 314, .6, .6), row(5, 314, .7, .7)])
    assert seen == 4
    assert spells == [dict(tokenizer=771, first=1, last=3, updates=2, peak_probability=.91),
                      dict(tokenizer=314, first=4, last=5, updates=2, peak_probability=.7)]


def test_the_report_refuses_missing_or_duplicated_updates():
    rows = [dict(step=step) for step in (3, 1, 2)]
    assert [row['step'] for row in complete_updates(rows, total=3)] == [1, 2, 3]
    for bad in [rows[:2], rows + [dict(step=2)], [dict(step=1), dict(step=2), dict(step=2)], rows + [dict(step=4)]]:
        with pytest.raises(ValueError):
            complete_updates(bad, total=3)


def test_the_extractor_refuses_missing_repeated_or_reordered_updates_and_a_changed_reward(monkeypatch):
    monkeypatch.setattr(extract, 'STOP', 3)
    row = lambda step, tokenizer, distinct: dict(
        stage='train', step=step, tokenizer_pool_index=tokenizer, tokenizer_config_index=tokenizer,
        paired_valid_examples=128, paired_maximum_positions=128, policy_reward_bpb=.1, paired_gap_bpb=.1,
        policy_updates=step, policy_unique_tokenizers=distinct)
    good = [row(1, 4, 1), row(2, 4, 1), dict(stage='eval', step=2), row(3, 9, 2)]
    assert [item['step'] for item in extract.train_rows(good)] == [1, 2, 3]
    for bad in [good[:3], good + [row(3, 9, 2)], [good[0], good[3], good[1]]]:
        with pytest.raises(ValueError):
            extract.train_rows(bad)
    changed = [dict(good[0], policy_reward_bpb=.2)] + good[1:]
    with pytest.raises(ValueError):
        extract.train_rows(changed)


def test_compressed_updates_are_byte_stable(tmp_path):
    rows = [dict(step=1, paired_gap_bpb=0.33591441050134324), dict(step=2, paired_gap_bpb=.1)]
    extract.write_csv(tmp_path / 'updates.csv.gz', rows, compress=True)
    first = (tmp_path / 'updates.csv.gz').read_bytes()
    extract.write_csv(tmp_path / 'updates.csv.gz', rows, compress=True)
    assert (tmp_path / 'updates.csv.gz').read_bytes() == first and first[4:8] == b'\0\0\0\0'  # gzip mtime field
    assert gzip.decompress(first).decode() == 'step,paired_gap_bpb\n1,0.33591441050134324\n2,0.1\n'


def test_manifest_hashes_every_report_file_but_itself_and_dot_files_and_is_stable(tmp_path):
    (tmp_path / 'data').mkdir()
    (tmp_path / 'README.md').write_text('# Report\n')
    (tmp_path / 'data' / 'evals.csv').write_bytes(b'run,step\nregret,0\n')
    (tmp_path / 'manifest.json').write_text('{}\n')
    (tmp_path / '.DS_Store').write_bytes(b'\0')
    first = manifest(tmp_path)
    assert list(first['files']) == ['README.md', 'data/evals.csv']
    assert first['files']['data/evals.csv'] == dict(bytes=18, sha256=hashlib.sha256(b'run,step\nregret,0\n').hexdigest())
    atomic_json(tmp_path / 'manifest.json', first)
    written = (tmp_path / 'manifest.json').read_bytes()
    atomic_json(tmp_path / 'manifest.json', manifest(tmp_path))
    assert (tmp_path / 'manifest.json').read_bytes() == written
    (tmp_path / 'README.md').write_text('# Report, changed\n')
    assert manifest(tmp_path)['files']['README.md'] != first['files']['README.md']


def test_the_privacy_guard_flags_unmasked_cluster_paths_but_not_masked_ones(tmp_path):
    (tmp_path / 'masked.json').write_text('{"model": "/gpfs/projects/stf/<user>/anytok-data/input/model"}\n')
    (tmp_path / 'unmasked.md').write_text('Run folder: /gpfs/scrubbed/someone/anytok-runs\n')
    (tmp_path / 'email.csv').write_text('owner\nperson@example.org\n')
    assert private_files(tmp_path) == ['email.csv', 'unmasked.md']
