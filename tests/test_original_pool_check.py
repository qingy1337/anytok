from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
from tokenizers import Tokenizer, models, decoders

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from original_pool_check import correct_panel


def fixture():
    backend = Tokenizer(models.BPE(vocab={'<unk>': 0, 'a': 1, 'b': 2, ' ': 3, 'ab': 4},
                                   merges=[('a', 'b')], unk_token='<unk>'))
    backend.decoder = decoders.Fuse()
    class Canonical:
        _tokenizer = backend
        def __call__(self, text, **kwargs):
            return SimpleNamespace(input_ids=backend.encode(text, add_special_tokens=False).ids)
        def decode(self, ids, **kwargs):
            return backend.decode(ids, skip_special_tokens=False)
    canonical = Canonical();text = 'ab ' * 512
    panel = dict(screen=[], confirm=[])
    for split, name, cut in [('screen', 'bad', 157), ('confirm', 'good', 158)]:
        prefix = text[cut - 90:cut];targets = canonical(text[cut:]).input_ids[:32]
        panel[split].append(dict(id=name, cluster_id=name, domain='web-en', prefix=prefix,
            canonical_prefix_ids=canonical(prefix).input_ids, canonical_continuation_ids=targets,
            continuation_bytes=len(canonical.decode(targets).encode()), cut_chars=cut))
    return panel, {'web-en': {'bad': text, 'good': text}}, canonical, backend


def test_fix_split_merge_and_preserve_valid_context_exactly():
    panel, sources, canonical, backend = fixture()
    rows = correct_panel(panel, sources, canonical, {1: backend}, {1: {0}})
    assert [r['source_document_id'] for r in rows] == ['bad', 'good']
    assert rows[0]['cut_shift_chars'] == 1
    assert not rows[0]['original_joint_encoding_matches']
    assert rows[1]['original_joint_encoding_matches']
    assert rows[1]['prefix'] == panel['confirm'][0]['prefix']
    assert rows[1]['canonical_continuation_ids'] == panel['confirm'][0]['canonical_continuation_ids']
    for r in rows:
        assert len(r['canonical_continuation_ids']) == 32
        assert r['maximum_positions'] <= 128
        assert canonical(r['prefix'] + canonical.decode(r['canonical_continuation_ids'])).input_ids == (
            r['canonical_prefix_ids'] + r['canonical_continuation_ids'])


def test_changed_source_rejected_instead_of_replacing_document():
    panel, sources, canonical, backend = fixture();sources['web-en']['bad'] = ' ' * 1536
    with pytest.raises(ValueError, match='Original prefix'):
        correct_panel(panel, sources, canonical, {1: backend}, {1: {0}})
