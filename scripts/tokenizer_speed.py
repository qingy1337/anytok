"""Matched tokenizer generation benchmark and exact Unicode round-trip audit.

This worker never downloads corpora or tokenizers to the caller. Set PYTHONPATH
to the intended isolated tokenizer wheel; use scripts/modal_tokenizer_speed.py
for persistent cloud experiments. Training settings and corpus quotas are
unchanged unless a separately named smaller-corpus variant is requested.
"""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import random
from pathlib import Path
import sys
import time

from pool_generalization import save_json, file_sha


def unicode_fixtures(count=256):
    """Exercise all Unicode planes, byte values, combining marks and controls."""
    rows = ['', ' ', '\n\r\t\x00', 'é e\u0301 中文 العربية हिन्दी 日本語 🙂',
            '<|endoftext|><|unk|><UNK>', 'x <|endoftext|> \n<|unk|> y', '\ufeff\u200b\u200d\u202e\ufffd',
            ''.join(chr(i) for i in range(256))]
    rng = random.Random(20261108)
    for _ in range(count):
        chars = []
        for _ in range(rng.randrange(1, 129)):
            code = rng.randrange(0x110000)
            if 0xD800 <= code <= 0xDFFF: code = 0x20
            chars.append(chr(code))
        rows.append(''.join(chars))
    return rows


def lossless_audit(path, heldout_root=None):
    from tokenizers import Tokenizer, pre_tokenizers
    tokenizer = Tokenizer.from_file(str(path)); config = json.loads(Path(path).read_text())
    if config['normalizer'] is not None or config['post_processor'] is not None:
        raise ValueError('Lossless path must not normalize or insert/drop text')
    if config.get('truncation') is not None or config.get('padding') is not None:
        raise ValueError('Lossless path must not truncate or pad text')
    if any(t.get('lstrip') or t.get('rstrip') for t in config['added_tokens']):
        raise ValueError('Special-token matching must not strip surrounding whitespace')
    if config['decoder']['type'] != 'ByteLevel': raise ValueError('Expected reversible byte-level decoder')
    if config['model']['type'] == 'Unigram' and any(score is None or not math.isfinite(score) for _,score in config['model']['vocab']):
        raise ValueError('Unigram byte coverage requires finite model scores')
    pre = config['pre_tokenizer']
    if pre['type'] == 'Sequence':
        steps = pre['pretokenizers']
        if (len(steps) != 2 or steps[0]['type'] != 'Split' or steps[0]['behavior'] != 'Isolated'
                or steps[0]['invert'] or steps[0]['pattern'] != {'Regex': r'[^\n]{1,256}\n?|\n'}):
            raise ValueError('Unsupported potentially lossy pretokenizer')
        pre = steps[1]
    if pre['type'] != 'ByteLevel' or pre['add_prefix_space']:
        raise ValueError('Expected byte-level pretokenization without added spaces')
    if config['model']['type'] == 'WordPiece':
        if config['model']['continuing_subword_prefix'] != '':
            raise ValueError('Byte-complete WordPiece requires an empty continuation prefix')
        if config['model']['max_input_chars_per_word'] < sys.maxsize:
            raise ValueError('Finite WordPiece input cutoff can replace long text with UNK')
    vocabulary = tokenizer.get_vocab()
    missing = set(pre_tokenizers.ByteLevel.alphabet()) - vocabulary.keys()
    if missing: raise ValueError('Missing byte alphabet entries')
    fixtures = unicode_fixtures()
    domains = {}; compressed = {}; raw_bytes = 0; tokens = 0
    if heldout_root is not None:
        for source in sorted(Path(heldout_root).glob('*.jsonl.gz')):
            texts = []
            with gzip.open(source, 'rt', encoding='utf-8') as stream:
                for line in stream:
                    text = json.loads(line)['text'][:512]
                    texts.append(text)
                    if len(texts) == 8: break
            domains[source.stem.removesuffix('.jsonl')] = len(texts)
            fixtures.extend(texts)
    h = hashlib.sha256()
    for text in fixtures:
        encoding = tokenizer.encode(text, add_special_tokens=False)
        restored = tokenizer.decode(encoding.ids, skip_special_tokens=False)
        if restored != text: raise ValueError('Exact Unicode round-trip failed')
        raw = text.encode('utf-8'); h.update(len(raw).to_bytes(8, 'little')); h.update(raw)
        raw_bytes += len(raw); tokens += len(encoding.ids)
    return {'roundtrip_failures': 0, 'fixtures': len(fixtures), 'heldout_domains': domains,
            'fixture_sha256': h.hexdigest(), 'fixture_bytes': raw_bytes, 'fixture_tokens': tokens,
            'all_256_byte_symbols_present': True, 'normalizer': None, 'post_processor': None,
            'finite_model_scores': True,
            'decode_skip_special_tokens': False,
            'scope': 'Valid Unicode strings; excludes unpaired surrogate codepoints. Standard decoder with skip_special_tokens=True intentionally discards special strings and is not the lossless API.'}


def run(config_path, recipe_path, destination, heldout_root=None):
    import tokenizer_pool
    config = json.loads(Path(config_path).read_text()); recipe = json.loads(Path(recipe_path).read_text())
    started = time.perf_counter()
    manifest = tokenizer_pool.train(config, recipe, destination)
    if config['algorithm'] == 'wordpiece':
        # Cover every representable input string rather than silently replacing
        # a very long no-regex segment with UNK. Learned vocabulary is unchanged.
        from tokenizers import Tokenizer
        path = Path(destination)/'tokenizer.json'
        tokenizer = Tokenizer.from_file(str(path))
        tokenizer.model.max_input_chars_per_word = sys.maxsize
        tokenizer.save(str(path))
        manifest.update({'tokenizer_sha256': file_sha(path), 'tokenizer_bytes': path.stat().st_size,
                         'lossless_runtime': {'wordpiece_max_input_chars_per_word': sys.maxsize}})
        save_json(Path(destination)/'manifest.json', manifest)
    ready_seconds = time.perf_counter() - started
    audit_started = time.perf_counter()
    audit = lossless_audit(Path(destination) / 'tokenizer.json', heldout_root)
    result = {'at_utc': datetime.now(timezone.utc).isoformat(), 'manifest': manifest,
              'ready_seconds': ready_seconds, 'extended_audit_seconds': time.perf_counter() - audit_started,
              'lossless_audit': audit, 'worker_sha256': file_sha(__file__)}
    save_json(Path(destination) / 'benchmark.json', result)
    print(json.dumps({'stage': 'audited', 'ready_seconds': ready_seconds, 'lossless_audit': audit}), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True); parser.add_argument('--recipe', required=True)
    parser.add_argument('--destination', required=True); parser.add_argument('--heldout-root')
    args = parser.parse_args()
    run(args.config, args.recipe, args.destination, args.heldout_root)
