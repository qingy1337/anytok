"""Repair original evaluation seams while preserving already valid contexts."""
import hashlib

from pool_generalization import digest


def correct_panel(panel, sources, canonical, tokenizers, forbidden):
    rows = []
    for split in ['screen', 'confirm']:
        for original in panel[split]:
            domain = original['domain'];text = sources[domain][original['cluster_id']]
            window = text[:4096];old_cut = original['cut_chars']
            left = old_cut - len(original['prefix'])
            if window[left:old_cut] != original['prefix']:
                raise ValueError('Original prefix does not match source document')
            old_targets = original['canonical_continuation_ids']
            old_suffix = canonical.decode(old_targets, clean_up_tokenization_spaces=False)
            if not window[old_cut:].startswith(old_suffix):
                raise ValueError('Original continuation does not match source document')
            old_valid = (canonical(original['prefix'] + old_suffix, add_special_tokens=False).input_ids
                         == original['canonical_prefix_ids'] + old_targets)
            candidates = sorted({end for _, end in canonical._tokenizer.encode(window, add_special_tokens=False).offsets
                                 if abs(end - old_cut) <= 64},
                                key=lambda cut: (abs(cut - old_cut), cut < old_cut, cut))
            # Already valid contexts must remain exactly unchanged.
            if old_valid:
                candidates = [old_cut]
            accepted = None
            for cut in candidates:
                if cut <= left:
                    continue
                targets = old_targets if old_valid else canonical(window[cut:], add_special_tokens=False).input_ids[:32]
                if len(targets) != 32 or 0 in targets:
                    continue
                suffix = canonical.decode(targets, clean_up_tokenization_spaces=False)
                if (not window[cut:].startswith(suffix)
                        or canonical(suffix, add_special_tokens=False).input_ids != targets):
                    continue
                raw_prefix = window[left:cut]
                for trim in range(len(raw_prefix)):
                    prefix = raw_prefix[trim:]
                    ids = canonical(prefix, add_special_tokens=False).input_ids
                    if not 16 <= len(ids) <= 64 or 0 in ids:
                        continue
                    if (canonical.decode(ids, clean_up_tokenization_spaces=False) != prefix
                            or canonical(prefix + suffix, add_special_tokens=False).input_ids != ids + targets):
                        continue
                    lengths = {}
                    for index, tokenizer in tokenizers.items():
                        native = tokenizer.encode(prefix, add_special_tokens=False).ids
                        if (not native or len(native) > 95 or forbidden[index].intersection(native)
                                or tokenizer.decode(native, skip_special_tokens=False) != prefix):
                            break
                        lengths[str(index)] = len(native)
                    else:
                        accepted = dict(original_id=original['id'], id=domain + '/' + original['id'],
                            source_document_id=original['cluster_id'], cluster_id=domain + '/' + original['cluster_id'],
                            domain=domain, split=split, document_sha256=hashlib.sha256(text.encode()).hexdigest(),
                            source_window_start=0, source_window_end=len(window), prefix=prefix,
                            canonical_prefix_ids=ids, canonical_continuation_ids=targets,
                            continuation_bytes=len(suffix.encode()), native_prefix_lengths=lengths,
                            original_cut_chars=old_cut, cut_chars=cut, cut_shift_chars=cut - old_cut,
                            original_prefix_chars=len(original['prefix']), additional_prefix_trim_chars=trim,
                            original_joint_encoding_matches=old_valid, canonical_joint_encoding_matches=True,
                            maximum_positions=33 + max(len(ids), max(lengths.values())),
                            context_sha256=digest([prefix, targets]))
                        break
                if accepted is not None:
                    break
            if accepted is None:
                raise ValueError('Cannot correct the frozen passage: ' + domain + '/' + original['id'])
            if old_valid and (accepted['prefix'] != original['prefix']
                              or accepted['canonical_continuation_ids'] != old_targets):
                raise ValueError('An already valid context changed')
            rows.append(accepted)
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate source IDs across splits')
    return rows
