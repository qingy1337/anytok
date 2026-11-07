"""Check suffix-only causal scoring and the immutable paired document panel."""
import json
from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from pool_ood_eval import document_nll, examples_for, LABELS, EXPECTED
from pool_generalization import load_plan


class OODEvaluationTest(unittest.TestCase):
    def test_causal_shift_scores_first_suffix_and_ignores_prefix_and_padding(self):
        # Different prefix lengths, exactly two suffix targets each.
        labels = torch.tensor([[-100, -100, 2, 1, -100], [-100, -100, -100, 0, 2]])
        logits = torch.zeros(2, 5, 3)
        nats, counts = document_nll(logits, labels)
        self.assertEqual(counts.tolist(), [2, 2])
        self.assertTrue(torch.allclose(nats, torch.full((2,), 2 * torch.log(torch.tensor(3.)))))
        # Only the position immediately BEFORE the first suffix target changes its loss.
        logits[0, 1, 2] = 20
        logits[1, 0, :] = torch.tensor([20., -20., -20.])  # ignored prefix
        logits[0, 4, :] = torch.tensor([20., -20., -20.])  # unused final position
        changed, _ = document_nll(logits, labels)
        self.assertAlmostEqual(changed[0].item(), torch.log(torch.tensor(3.)).item(), places=5)
        self.assertEqual(changed[1].item(), nats[1].item())

    def test_frozen_panel_and_heldout_hashes(self):
        plan = load_plan(ROOT / 'plans/pool-ood-48k-v1.json')
        panel = plan['panel']
        self.assertEqual((len(panel), sum(len(r['canonical_continuation_ids']) for r in panel),
                          sum(r['continuation_bytes'] for r in panel)), EXPECTED)
        self.assertEqual(len({r['id'] for r in panel}), 256)
        self.assertEqual(set(plan['tokenizer_sha256']), set(LABELS))
        hashes = {json.loads(p.read_text())['tokenizer_sha256'] for p in
                  (ROOT / 'modal_results/tokenizer_pool').glob('[0-9][0-9][0-9][0-9].json')}
        self.assertEqual(len(hashes), 1000)
        self.assertFalse(hashes.intersection(plan['tokenizer_sha256'].values()))
        examples = examples_for(panel)
        self.assertTrue(all(1 + len(p) + len(s) <= 128 and len(s) == 32 for p, s, _ in examples))
        broken = [dict(panel[0], canonical_prefix_ids=[1] * 96)]
        with self.assertRaises(ValueError): examples_for(broken)


if __name__ == '__main__': unittest.main()
