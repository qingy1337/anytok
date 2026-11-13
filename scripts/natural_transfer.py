"""Loss-independent text selection and paired natural-text transfer statistics."""
import hashlib
import math
import re

from pool_generalization import digest

DOMAINS = ('web-en', 'code', 'math-en', 'medical-en', 'legal-en')
TAGS = ('image_markup', 'structured', 'numeric_dense', 'repeated_number', 'plain_prose')


def choose_tokenizers(spec):
    """One native setting per algorithm/pretokenizer/target-size cell; no losses."""
    chosen, cells = set(), []
    for algorithm in ('bpe', 'wordpiece', 'unigram'):
        for pretokenizer in ('regex', 'line256', 'no-regex'):
            options = [r for r in spec['tokenizers'] if r['config']['algorithm'] == algorithm
                       and r['config']['pretokenizer'] == pretokenizer]
            for target in (32000, 64000, 96000, 128000):
                best = min((r for r in options if r['index'] not in chosen),
                    key=lambda r: (abs(r['vocab_size'] - target), digest(['natural-map-v1', r['index']])))
                chosen.add(best['index'])
                cells.append(dict(index=best['index'], algorithm=algorithm,
                                  pretokenizer=pretokenizer, target_vocab=target))
    return sorted(chosen), cells


def text_tags(prefix, suffix, domain):
    """Overlapping descriptive categories fixed without model/transfer scores."""
    text = prefix + suffix
    markup = bool(re.search(r'!\[[^\]]*\]\(|\{#[^}]+\}|<img\b', text, re.I))
    lines = text.splitlines()
    structured = (sum(bool(re.search(r'\t|\|', line)) for line in lines) >= 2
                  or sum(line.count(',') >= 2 for line in lines) >= 2
                  or sum(bool(re.match(r'\s*[\w .-]{1,40}:\s*\S', line)) for line in lines) >= 2)
    numeric = sum(c.isdigit() for c in text) / max(1, len(text)) >= .08
    repeated = any(m.group() in prefix for m in re.finditer(r'\d{3,}(?:\.\d+)?', suffix))
    prose = (domain != 'code' and not markup and not structured and not numeric
             and sum(c.isalpha() for c in text) / max(1, len(text)) >= .6)
    return dict(image_markup=markup, structured=structured, numeric_dense=numeric,
                repeated_number=repeated, plain_prose=prose)


def build_panel(sources, canonical, tokenizers, forbidden, excluded, counts):
    """Two disjoint source windows/document; discovery/confirmation split by document."""
    from uniform_trajectory import fixed_document
    rows, rejected, corpus = [], [], {}
    used_hashes = set()
    for domain in DOMAINS:
        documents = sources[domain]
        if len({r['id'] for r in documents}) != len(documents):
            raise ValueError('Duplicate source IDs')
        ordered = sorted(documents, key=lambda r: digest(['natural-map-v1', domain, r['id']]))
        accepted, reason_counts = [], {}
        for raw in ordered:
            if raw['id'] in excluded[domain]:
                continue
            content_sha = hashlib.sha256(raw['text'].encode()).hexdigest()
            if content_sha in used_hashes:
                rejected.append(dict(domain=domain, id=raw['id'], reason='duplicate full document text'))
                continue
            try:
                text = raw['text']
                if len(text) < 1024:
                    raise ValueError('less than1024 characters for two disjoint windows')
                prepared = []
                middle = len(text) // 2
                for position, (left, right) in enumerate(((0, middle), (middle, len(text)))):
                    width = min(2048, right - left)
                    maximum = right - width
                    start = left + int(digest([content_sha, position])[:12], 16) % (maximum - left + 1)
                    item = fixed_document(dict(id=raw['id'], text=text[start:start + width]),
                                          canonical, canonical._tokenizer, {0})
                    # Joint validation changes only the prefix, never the continuation.
                    original = item['prefix']
                    for trim in range(len(original)):
                        prefix = original[trim:]
                        source_ids = canonical(prefix, add_special_tokens=False).input_ids
                        if not 16 <= len(source_ids) <= 64 or 0 in source_ids:
                            continue
                        if canonical.decode(source_ids, clean_up_tokenization_spaces=False) != prefix:
                            continue
                        lengths = {}
                        for index, tokenizer in tokenizers.items():
                            ids = tokenizer.encode(prefix, add_special_tokens=False).ids
                            if (not ids or len(ids) > 95 or forbidden[index].intersection(ids)
                                    or tokenizer.decode(ids, skip_special_tokens=False) != prefix):
                                break
                            lengths[str(index)] = len(ids)
                        else:
                            break
                    else:
                        raise ValueError('no shared lossless prefix with16 canonical tokens')
                    suffix = canonical.decode(item['canonical_continuation_ids'], clean_up_tokenization_spaces=False)
                    if (len(item['canonical_continuation_ids']) != 32 or 0 in item['canonical_continuation_ids']
                            or len(suffix.encode()) != item['continuation_bytes']):
                        raise ValueError('invalid continuation')
                    item.update(id=f'{domain}/{raw["id"]}/window-{position}', domain=domain,
                        cluster_id=domain + '/' + raw['id'], source_document_id=raw['id'],
                        document_sha256=content_sha, source_window_start=start, source_window_end=start + width,
                        absolute_cut_chars=start + item['cut_chars'], prefix=prefix,
                        canonical_prefix_ids=source_ids, native_prefix_lengths=lengths,
                        prefix_trim_chars=trim, original_prefix_chars=len(original),
                        tags=text_tags(prefix, suffix, domain),
                        context_sha256=digest([prefix, item['canonical_continuation_ids']]))
                    item.pop('compression_text', None)
                    prepared.append(item)
                if prepared[0]['source_window_end'] > prepared[1]['source_window_start']:
                    raise ValueError('overlapping document windows')
            except ValueError as exc:
                reason = str(exc)
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
                rejected.append(dict(domain=domain, id=raw['id'], reason=reason))
                continue
            used_hashes.add(content_sha)
            accepted.append(prepared)
            if len(accepted) == counts[domain] * 2:
                break
        if len(accepted) != counts[domain] * 2:
            raise ValueError(f'{domain}: only{len(accepted)} valid fresh documents; need{counts[domain] * 2}')
        for j, pair in enumerate(accepted):
            split = 'screen' if j < counts[domain] else 'confirm'
            rows.extend([{**r, 'split': split} for r in pair])
        corpus[domain] = dict(source_documents=len(documents), old_excluded_documents=len(excluded[domain]),
            documents_per_split=counts[domain], passages_per_split=counts[domain] * 2, rejections=reason_counts)
    return rows, dict(corpora=corpus, rejected=rejected)


def validate_scores(rows, nats):
    import numpy as np
    array = np.asarray(nats, dtype=float)
    if array.shape != (len(rows), 32) or not np.isfinite(array).all() or (array < 0).any():
        raise ValueError('Invalid/missing per-token losses')
    return array


def align_panel(parent, sources, canonical, tokenizers, forbidden):
    """Move cuts by encoding rules only; preserve all documents and source windows."""
    documents = {domain: {r['id']: r['text'] for r in sources[domain]} for domain in DOMAINS}
    rows, shifts = [], []
    for row in parent:
        text = documents[row['domain']][row['source_document_id']]
        if hashlib.sha256(text.encode()).hexdigest() != row['document_sha256']:
            raise ValueError('Parent source document changed')
        start, end = row['source_window_start'], row['source_window_end']
        window = text[start:end]
        previous_cut = row['absolute_cut_chars'] - start
        # Candidate offsets come from a normal canonical encoding of the full window.
        encoding = canonical._tokenizer.encode(window, add_special_tokens=False)
        candidates = sorted({offset[1] for offset in encoding.offsets if abs(offset[1] - previous_cut) <= 64},
                            key=lambda cut: (abs(cut - previous_cut), cut < previous_cut, cut))
        accepted = None
        for cut in candidates:
            continuation = canonical(window[cut:], add_special_tokens=False).input_ids[:32]
            if len(continuation) != 32 or 0 in continuation:
                continue
            suffix = canonical.decode(continuation, clean_up_tokenization_spaces=False)
            if (not window[cut:].startswith(suffix)
                    or canonical(suffix, add_special_tokens=False).input_ids != continuation):
                continue
            raw_prefix = window[max(0, cut - 512):cut]
            canonical_budget_trim = None
            for trim in range(len(raw_prefix)):
                prefix = raw_prefix[trim:]
                source_ids = canonical(prefix, add_special_tokens=False).input_ids
                if not 16 <= len(source_ids) <= 64 or 0 in source_ids:
                    continue
                if canonical.decode(source_ids, clean_up_tokenization_spaces=False) != prefix:
                    continue
                if canonical(prefix + suffix, add_special_tokens=False).input_ids != source_ids + continuation:
                    continue
                if canonical_budget_trim is None:
                    canonical_budget_trim = trim
                lengths = {}
                for index, tokenizer in tokenizers.items():
                    ids = tokenizer.encode(prefix, add_special_tokens=False).ids
                    if (not ids or len(ids) > 95 or forbidden[index].intersection(ids)
                            or tokenizer.decode(ids, skip_special_tokens=False) != prefix):
                        break
                    lengths[str(index)] = len(ids)
                else:
                    accepted = {**row, 'prefix': prefix, 'canonical_prefix_ids': source_ids,
                        'canonical_continuation_ids': continuation, 'continuation_bytes': len(suffix.encode()),
                        'native_prefix_lengths': lengths, 'absolute_cut_chars': start + cut,
                        'cut_chars': cut, 'parent_absolute_cut_chars': row['absolute_cut_chars'],
                        'cut_shift_chars': cut - previous_cut, 'prefix_trim_chars': trim - canonical_budget_trim,
                        'original_prefix_chars': len(raw_prefix) - canonical_budget_trim,
                        'maximum_positions': 33 + max(len(source_ids), max(lengths.values())),
                        'tags': text_tags(prefix, suffix, row['domain']),
                        'context_sha256': digest([prefix, continuation]), 'canonical_joint_encoding_matches': True}
                    break
            if accepted is not None:
                break
        if accepted is None:
            raise ValueError('Cannot align the frozen passage without changing its document: ' + row['id'])
        rows.append(accepted)
        shifts.append(accepted['cut_shift_chars'])
    return rows, dict(rows=len(rows), changed_cuts=sum(v != 0 for v in shifts),
        max_absolute_cut_shift_chars=max(abs(v) for v in shifts),
        all_canonical_joint_encodings_match=True, rejected=[],
        correction='Same documents, splits and disjoint source windows as v1. Nearest reversible canonical boundary within64 characters; selection uses encoding validity only.')


def metrics(rows, delta, base, transferred):
    """Pooled BPB plus explicitly document-weighted tails, preserving negative gaps."""
    import numpy as np
    counts = np.array([r['continuation_bytes'] for r in rows], dtype=float) * math.log(2)
    gaps = np.asarray(delta).sum(1) / counts
    n = max(1, math.ceil(len(gaps) * .1))
    return dict(passages=len(rows), documents=len({r['cluster_id'] for r in rows}),
        continuation_tokens=32 * len(rows), continuation_bytes=int(sum(r['continuation_bytes'] for r in rows)),
        canonical_bpb=float(np.asarray(base).sum() / counts.sum()),
        transferred_bpb=float(np.asarray(transferred).sum() / counts.sum()),
        gap_bpb=float(np.asarray(delta).sum() / counts.sum()), mean_passage_gap_bpb=float(gaps.mean()),
        p90_passage_gap_bpb=float(np.quantile(gaps, .9)),
        worst_decile_mean_gap_bpb=float(np.sort(gaps)[-n:].mean()),
        fraction_gap_above_01=float((gaps > .1).mean()),
        fraction_gap_above_025=float((gaps > .25).mean()))
