"""Task-focused stress tests of information retained in transferred prefixes."""
import hashlib
import math
import random
import string

from pool_generalization import digest

FAMILIES = ('numbers', 'identifiers', 'names', 'quoted_text', 'semantic_retrieval')
TAIL = ('\nThe entry was copied from the earlier record. This completes the requested entry and preserves its original wording and formatting. The record is ready for review. '
        'The clerk checked that all required fields were present before saving the completed document. '
        'The next reviewer can now read the record and verify the information against the original note.')
FIRST = ('Clara', 'Nathan', 'Elena', 'Rowan', 'Miriam', 'Owen', 'Vivian', 'Julian', 'Nadia', 'Simon', 'Alice', 'Theo')
LAST = ('Bennett', 'Mercer', 'Morales', 'Whitaker', 'Hollis', 'Fletcher', 'Ramsey', 'Walton', 'Sutton', 'Carver', 'Wells', 'Hartley')


def answer_present(text, row):
    """Allow surrounding prose, name capitalization and exact numeric reformatting."""
    import re
    from decimal import Decimal, InvalidOperation
    answer = ' '.join(row['answer'].split())
    normalized = ' '.join(text.split())
    left, right = (r'(?<![\w.])', r'(?![\w.])') if row['family'] == 'numbers' else (r'(?<!\w)', r'(?![\w-])')
    literal = re.search(left + re.escape(answer.casefold()) + right, normalized.casefold()) is not None
    if literal or row['family'] != 'numbers' or row.get('variant', 0) % 4 == 2:
        return literal
    try:
        expected = Decimal(answer)
        candidates = re.findall(r'(?<![\w.])[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?(?![\w.])', normalized.replace(',', ''))
        return any(Decimal(value) == expected for value in candidates)
    except InvalidOperation:
        return False


def choose_tokenizers(spec, include_known=True):
    chosen = set()
    for algorithm in ['bpe', 'wordpiece', 'unigram']:
        for pretokenizer in ['regex', 'line256', 'no-regex']:
            group = [r for r in spec['tokenizers'] if r['config']['algorithm'] == algorithm
                     and r['config']['pretokenizer'] == pretokenizer]
            for target in [64000, 120000]:
                options = [r for r in group if r['index'] not in chosen]
                best = min(options, key=lambda r: (abs(r['vocab_size'] - target),
                    abs(r['config']['noise_per_million'] - 375000) if algorithm == 'bpe' else 0, r['index']))
                chosen.add(best['index'])
    return sorted(chosen | ({3, 635} if include_known else set()))


def payload(family, size, variant, rng):
    length = 8 if size == 'short' else 20
    if family == 'numbers':
        digits = str(rng.randrange(1, 10)) + ''.join(rng.choices(string.digits, k=length - 1))
        if variant % 4 == 1:
            return digits[:2] + '.' + digits[2:], 'measured value'
        if variant % 4 == 2:
            return digits[:4] + '-' + digits[4:], 'reference number'
        if variant % 4 == 3:
            return digits[:2] + '.' + digits[2:] + 'e-4', 'reported coefficient'
        return digits, 'recorded number'
    if family == 'identifiers':
        alphabets = [string.hexdigits[:16], string.ascii_uppercase + string.digits,
                     string.ascii_lowercase + string.digits, string.ascii_letters]
        value = ''.join(rng.choices(alphabets[variant % 4], k=length))
        return value, ['checksum', 'record identifier', 'file identifier', 'access code'][variant % 4]
    if family == 'names':
        name = rng.choice(FIRST) + ' ' + rng.choice(LAST)
        if size == 'long':
            name = rng.choice(FIRST) + ' ' + name + ' ' + rng.choice(LAST)
        return name, ['contact name', 'author name', 'reviewer name', 'recipient name'][variant % 4]
    if family == 'quoted_text':
        adjective = rng.choice(['blue', 'quiet', 'heavy', 'wooden', 'silver', 'green', 'narrow', 'bright'])
        noun = rng.choice(['gate', 'folder', 'door', 'box', 'window', 'bridge', 'cabin', 'package'])
        if size == 'short':
            return f'The {adjective} {noun} opened', 'quoted sentence'
        return f'The careful engineer placed the {adjective} {noun} beside the desk before noon', 'quoted sentence'
    return rng.choice(['green', 'purple', 'yellow', 'orange', 'blue'] if size == 'short'
                      else ['light blue', 'dark green', 'pale yellow', 'bright orange']), 'folder color'


def synthetic_rows(canonical, split, count=64):
    """Independent seeds and two disjoint wording templates in each split."""
    rng = random.Random(20261114 if split == 'screen' else 20261115)
    result = []
    for family in FAMILIES:
        for index in range(count):
            size = 'short' if index % 2 == 0 else 'long'
            distance = 'near' if index % 4 < 2 else 'far'
            value, field = payload(family, size, index // 4, rng)
            filler = '' if distance == 'near' else 'The clerk checked the date, signed the page, and returned the folder to the shelf.\n'
            if family == 'semantic_retrieval':
                person = rng.choice(FIRST)
                if split == 'screen':
                    prefix = f"{person}'s folder is {value}.\n{filler}Question: What color is {person}'s folder?\nAnswer:"
                else:
                    prefix = f'The folder assigned to {person} has the color {value}.\n{filler}The color of the folder assigned to {person} is'
            elif split == 'screen':
                if index % 8 < 4:
                    prefix = f'Record\nThe {field}: {value}\n{filler}Copy of the record\nThe {field}:'
                else:
                    prefix = f'The {field} was {value}.\n{filler}For the report, the {field} was'
            else:
                if index % 8 < 4:
                    prefix = f'Original entry\n{field.capitalize()}: {value}\n{filler}Repeated entry\n{field.capitalize()}:'
                else:
                    prefix = f'The note gives the {field} as {value}.\n{filler}Question: What was the {field}?\nAnswer:'
            answer = ' ' + value
            answer_ids = canonical(answer, add_special_tokens=False).input_ids
            full = answer + TAIL
            continuation = canonical(full, add_special_tokens=False).input_ids[:32]
            if len(continuation) != 32 or len(answer_ids) > 32 or continuation[:len(answer_ids)] != answer_ids:
                raise ValueError(f'Invalid answer span: {split}/{family}/{index}, '
                    f'answer_tokens={len(answer_ids)}, continuation_tokens={len(continuation)}, '
                    f'prefix_match={continuation[:len(answer_ids)] == answer_ids}')
            suffix = canonical.decode(continuation, clean_up_tokenization_spaces=False)
            if not full.startswith(suffix) or canonical(suffix, add_special_tokens=False).input_ids != continuation:
                raise ValueError('Continuation does not round trip')
            result.append({'id': f'{split}/{family}/{index:03d}', 'family': family, 'split': split,
                'size': size, 'distance': distance, 'variant': index // 4, 'prefix': prefix,
                'canonical_prefix_ids': canonical(prefix, add_special_tokens=False).input_ids,
                'canonical_continuation_ids': continuation, 'continuation_bytes': len(suffix.encode()),
                'answer_ids': answer_ids, 'answer_bytes': len(answer.encode()), 'answer': answer})
    return result


def validate_examples(rows, canonical, tokenizers):
    """Keep entire task facts and query; reject by encoding validity only."""
    accepted, rejected = [], []
    for row in rows:
        failed = []
        if 1 + len(row['canonical_prefix_ids']) + 32 > 128:
            failed.append('canonical_length')
        for index, tokenizer in tokenizers.items():
            ids = tokenizer.encode(row['prefix'], add_special_tokens=False).ids
            if (not ids or 1 + len(ids) + 32 > 128
                    or tokenizer.decode(ids, skip_special_tokens=False) != row['prefix']
                    or 0 in ids):
                failed.append(str(index))
        (rejected if failed else accepted).append({'id': row['id'], 'failed': failed} if failed else row)
    return accepted, rejected


def summarize(rows, scored):
    """Answer-only metrics preserve prefix dependence; 32-token BPB is separate."""
    if len(rows) != len(scored):
        raise ValueError('Missing behavior scores')
    groups = {}
    for row, score in zip(rows, scored):
        if (len(score['nats']) != 32 or len(score['correct']) != 32
                or not all(math.isfinite(v) and v >= 0 for v in score['nats'])
                or not 1 <= len(row['answer_ids']) <= 32 or row['answer_bytes'] <= 0):
            raise ValueError('Invalid behavior scores')
        groups.setdefault((row['split'], row['family'], row.get('size', 'natural'), row.get('distance', 'natural')), []).append((row, score))
    result = []
    for (split, family, size, distance), examples in sorted(groups.items()):
        bytes32 = sum(r['continuation_bytes'] for r, _ in examples)
        answer_bytes = sum(r['answer_bytes'] for r, _ in examples)
        answer_nats = sum(sum(s['nats'][:len(r['answer_ids'])]) for r, s in examples)
        result.append({'split': split, 'family': family, 'size': size, 'distance': distance,
            'examples': len(examples), 'bpb_32': sum(sum(s['nats']) for _, s in examples) / (bytes32 * math.log(2)),
            'answer_bpb': answer_nats / (answer_bytes * math.log(2)),
            'answer_exact_match': sum(all(s['correct'][:len(r['answer_ids'])]) for r, s in examples) / len(examples),
            'answer_tokens': sum(len(r['answer_ids']) for r, _ in examples),
            'continuation_bytes': bytes32, 'answer_bytes': answer_bytes})
    return result
