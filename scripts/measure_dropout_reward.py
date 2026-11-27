"""Measure one-pass reward noise from dropout on exact paired training batches."""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch

from online_reward_scoring import BATCH_SIZE, MICROBATCH_SIZE, SharedBatch
from online_tokenizer_training import JointScorer
from package_training_inputs import atomic_json
from run_online_tokenizer_training import CHECKPOINT_SHA, INPUT_SHA, PolicyState, load_trainer, propose, verify_inputs
from run_raw_bpb_training import prepare_code


def emit(output, stage, completed, total, **fields):
    row = dict(at_utc=datetime.now(timezone.utc).isoformat(), stage=stage, completed=completed, total=total,
               platform='Tillicum · H200', profiles=['zlab'], workers=[] if stage in ('complete', 'failed') else [dict(
                   id=str(__import__('os').environ.get('SLURM_JOB_ID', 'dropout-diagnostic')),
                   profile='zlab', platform='Tillicum · H200', state=stage)], **fields)
    atomic_json(output / 'status.json', row)
    with (output / 'events.jsonl').open('a') as stream:
        stream.write(json.dumps(row, allow_nan=False) + '\n')
    print(json.dumps(row, allow_nan=False), flush=True)


def score(scorer, batch, deterministic, seed=None):
    examples, _, forms, priors, metrics = batch
    if deterministic:
        context = scorer.inference()
    else:
        scorer.lm.train()
        scorer.hn.train()
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        context = torch.no_grad()
    with context:
        with torch.autocast('cuda', dtype=torch.bfloat16):
            table, output, _ = scorer.hn(forms, priors, scorer.source, None, deterministic=deterministic)
        if output is not None or table.shape != (len(forms), scorer.source.shape[1]) or not torch.isfinite(table).all():
            raise ValueError('Invalid diagnostic hypernetwork output')
        # The training path converts the generated table to float before the LM.
        table = table.detach().float()
        parts, counts = [], []
        for start in range(0, BATCH_SIZE, MICROBATCH_SIZE):
            micro = examples[start:start + MICROBATCH_SIZE]
            embeds, mask, labels, positions = scorer.trainer.make_lm_microbatch(
                micro, table, scorer.source, 'cuda')
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss = scorer.lm(inputs_embeds=embeds, attention_mask=mask, labels=labels,
                                 position_ids=positions, use_cache=False).loss
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite diagnostic LM loss')
            parts.append(loss.detach())
            counts.append(sum(len(row[1]) for row in micro))
            del embeds, mask, labels, positions, loss
        values = torch.stack(parts).float().cpu().tolist()
        nats = sum(value * count for value, count in zip(values, counts, strict=True))
    return nats / (metrics['scored_bytes'] * math.log(2))


def rank(values):
    order = np.argsort(np.asarray(values), kind='stable')
    result = np.empty(len(order), dtype=float)
    result[order] = np.arange(len(order), dtype=float)
    return result


def summarize(rows, repeats):
    clean = np.asarray([row['dropout_off_bpb'] for row in rows])
    active = np.asarray([row['dropout_on_bpb'] for row in rows])
    difference = active - clean[:, None]
    clean_rank = rank(clean)
    correlations, top_overlaps, inversions = [], [], []
    top_count = max(1, len(rows) // 4)
    clean_top = set(np.argsort(clean, kind='stable')[-top_count:])
    pairs = [(i, j) for i in range(len(rows)) for j in range(i + 1, len(rows)) if clean[i] != clean[j]]
    for repeat in range(repeats):
        scores = active[:, repeat]
        correlations.append(float(np.corrcoef(clean_rank, rank(scores))[0, 1]))
        top = set(np.argsort(scores, kind='stable')[-top_count:])
        top_overlaps.append(len(clean_top & top) / top_count)
        inversions.append(sum((clean[i] > clean[j]) != (scores[i] > scores[j]) for i, j in pairs) / len(pairs))
    per_action_std = active.std(axis=1, ddof=1)
    return dict(
        actions=len(rows), dropout_repeats=repeats, top_count=top_count,
        dropout_off_mean_bpb=float(clean.mean()), dropout_on_mean_bpb=float(active.mean()),
        mean_signed_shift_bpb=float(difference.mean()), mean_absolute_shift_bpb=float(np.abs(difference).mean()),
        rms_shift_bpb=float(np.sqrt(np.square(difference).mean())),
        maximum_absolute_shift_bpb=float(np.abs(difference).max()),
        median_per_action_dropout_std_bpb=float(np.median(per_action_std)),
        maximum_per_action_dropout_std_bpb=float(per_action_std.max()),
        between_action_dropout_off_std_bpb=float(clean.std(ddof=1)),
        spearman_rank_correlation=dict(mean=float(np.mean(correlations)), minimum=float(np.min(correlations)),
                                       values=correlations),
        top_quarter_overlap=dict(mean=float(np.mean(top_overlaps)), minimum=float(np.min(top_overlaps)),
                                 values=top_overlaps),
        pairwise_order_disagreement=dict(mean=float(np.mean(inversions)), maximum=float(np.max(inversions)),
                                         values=inversions))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--actions', type=int, default=32)
    parser.add_argument('--repeats', type=int, default=8)
    args = parser.parse_args()
    if not 2 <= args.actions <= 64 or not 2 <= args.repeats <= 16:
        parser.error('Use 2–64 actions and 2–16 dropout repeats')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Choose an empty output directory')
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        verify_inputs(args.inputs, args.checkpoint)
        if torch.__version__ != '2.15.0+cu132' or 'H200' not in torch.cuda.get_device_name():
            raise ValueError('Expected a Tillicum H200 and PyTorch 2.15.0+cu132')
        identity = dict(checkpoint=CHECKPOINT_SHA, inputs=INPUT_SHA, actions=args.actions, repeats=args.repeats,
                        student='frozen Uniform 20k', policy='fresh direct policy', data_seed=42,
                        action_seed=43, generation_seed=45, dropout_seeds='100000 + action*repeats + repeat')
        atomic_json(args.output / 'run-config.json', identity)
        emit(args.output, 'loading frozen Uniform student', 0, args.actions)
        torch.set_num_threads(1)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        code = prepare_code(args.output.parent)
        trainer = load_trainer(code)
        scorer = JointScorer(trainer, args.inputs, args.checkpoint, json.loads((args.inputs / 'recipe.json').read_text()))
        policy = PolicyState()
        data_rng, action_rng = random.Random(42), np.random.default_rng(43)
        corpus = trainer.BalancedTextStream(args.inputs / 'recipe.json', args.inputs / 'corpus', None)
        rows = []
        for index in range(args.actions):
            began = time.perf_counter()
            shared = SharedBatch(corpus, data_rng)
            with torch.no_grad():
                batch, _, rejected = propose(scorer, policy, action_rng, shared)
            batch = scorer.to_device(batch)
            clean = score(scorer, batch, True)
            active = [score(scorer, batch, False, 100000 + index * args.repeats + repeat)
                      for repeat in range(args.repeats)]
            metrics = batch[-1]
            row = dict(index=index, settings=metrics['settings'], tokenizer_sha256=metrics['tokenizer_sha256'],
                       paired_batch_sha256=metrics['paired_batch_sha256'], scored_tokens=metrics['scored_tokens'],
                       scored_bytes=metrics['scored_bytes'], rejected_settings=rejected, dropout_off_bpb=clean,
                       dropout_on_bpb=active, dropout_on_mean_bpb=float(np.mean(active)),
                       dropout_on_std_bpb=float(np.std(active, ddof=1)),
                       mean_shift_bpb=float(np.mean(active) - clean), seconds=time.perf_counter() - began)
            rows.append(row)
            atomic_json(args.output / 'scores.json', rows)
            emit(args.output, 'scoring frozen paired batches', index + 1, args.actions,
                 latest=dict(index=index, vocab_size=metrics['settings']['vocab_size'],
                             dropout_off_bpb=clean, dropout_on_mean_bpb=row['dropout_on_mean_bpb']))
            del batch
            torch.cuda.empty_cache()
        corpus.close()
        summary = summarize(rows, args.repeats)
        atomic_json(args.output / 'summary.json', summary)
        emit(args.output, 'complete', args.actions, args.actions, summary=summary)
    except Exception as error:
        emit(args.output, 'failed', 0, args.actions, error=str(error))
        raise


if __name__ == '__main__':
    main()
