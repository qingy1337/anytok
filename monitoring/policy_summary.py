"""Small JSON summaries of policy weights, computed where checkpoints live."""
import math


def summarize(configs, probabilities, counts, step):
    if (len(configs) != len(probabilities) or len(counts) != len(configs)
            or not configs or sum(counts) != step
            or any(not math.isfinite(p) or p < 0 for p in probabilities)
            or not math.isclose(sum(probabilities), 1., abs_tol=1e-10)):
        raise ValueError('Invalid policy distribution/counters')
    n = len(configs)
    groups = {}

    def group(name, labels, eligible=None):
        eligible = list(range(n)) if eligible is None else eligible
        mass = sum(probabilities[i] for i in eligible)
        visits = sum(counts[i] for i in eligible)
        groups[name] = {label: {
            'policy': sum(probabilities[i] for i in eligible if labels[i] == label) / mass if mass else 0.,
            'uniform': sum(labels[i] == label for i in eligible) / len(eligible),
            'observed': sum(counts[i] for i in eligible if labels[i] == label) / visits if visits else None,
        } for label in sorted({labels[i] for i in eligible})}

    for field in ('algorithm', 'pretokenizer'):
        group(field, [c[field] for c in configs])
    group('vocab', [f'{min(3, int((c["vocab_size"] - 32000) / 24000))}' for c in configs])
    bpe = [i for i, c in enumerate(configs) if c['algorithm'] == 'bpe']
    if bpe:
        group('merge_noise', [str(min(2, int((c['noise_per_million'] or 0) / 250000))) for c in configs], bpe)
    groups['domain'] = {}
    for domain in configs[0]['domain_bytes']:
        fractions = [c['domain_bytes'][domain] / sum(c['domain_bytes'].values()) for c in configs]
        groups['domain'][domain] = {
            'policy': sum(p * f for p, f in zip(probabilities, fractions)),
            'uniform': sum(fractions) / n,
            'observed': sum(c * f for c, f in zip(counts, fractions)) / step if step else None}
    return {'step': step, 'groups': groups,
            'entropy': -sum(p * math.log(p) for p in probabilities if p),
            'top_100_probability': sum(sorted(probabilities, reverse=True)[:100]),
            'max_probability': max(probabilities),
            'mean_vocab': sum(p * c['vocab_size'] for p, c in zip(probabilities, configs)),
            'top_tokenizers': [
                {'index': configs[i]['index'], 'probability': probabilities[i],
                 'algorithm': configs[i]['algorithm'], 'pretokenizer': configs[i]['pretokenizer'],
                 'vocab_size': configs[i]['vocab_size'], 'selections': counts[i]}
                for i in sorted(range(n), key=lambda i: probabilities[i], reverse=True)[:10]]}
