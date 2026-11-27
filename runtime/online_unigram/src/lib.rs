//! A sampled suffix trie represented by sorted, length-limited slices of the actual mixture.
use onig::Regex;
use pyo3::{exceptions::PyValueError, prelude::*};
use rayon::prelude::*;
use std::sync::LazyLock;

static BYTELEVEL: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+")
        .unwrap()
});
static LINE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[^\n]{1,256}\n?|\n").unwrap());

struct Bucket {
    data: Vec<u8>,
    ends: Vec<u32>,
    regex: Vec<u32>,
    lines: Vec<u32>,
}

fn ends_for(text: &str, base: usize, re: &Regex) -> Vec<u32> {
    re.find_iter(text)
        .map(|(_, end)| (base + end) as u32)
        .collect()
}

#[pyclass]
struct Corpus {
    buckets: Vec<Bucket>,
}

#[derive(Clone, Copy)]
struct Span {
    start: u32,
    len: u32,
}
impl Span {
    fn bytes(self, text: &[u8]) -> &[u8] {
        &text[self.start as usize..(self.start + self.len) as usize]
    }
}
fn mix(mut x: u64) -> u64 {
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}
fn alphabet() -> [char; 256] {
    let mut extra = 256;
    std::array::from_fn(|i| {
        let code =
            if (33..=126).contains(&i) || (161..=172).contains(&i) || (174..=255).contains(&i) {
                i as u32
            } else {
                extra += 1;
                extra - 1
            };
        char::from_u32(code).unwrap()
    })
}

#[derive(Clone, Copy)]
struct Candidate {
    span: Span,
    score: f64,
}
struct Candidates<'a> {
    text: &'a [u8],
    list: Vec<Candidate>,
    bias: f64,
    noise: f64,
    seed: u64,
}
impl Candidates<'_> {
    fn push(&mut self, span: Span, count: usize) {
        if span.len < 2 {
            return;
        }
        let piece = span.bytes(self.text);
        if piece == b"<|endoftext|>" || piece == b"<|unk|>" {
            return;
        }
        let mut score = (count as f64).ln() + self.bias * (span.len as f64).ln();
        if self.noise != 0.0 {
            // The sorted traversal gives every distinct candidate a reproducible draw.
            let bits = mix(self.seed.wrapping_add(self.list.len() as u64)) >> 12;
            let u = (bits as f64 + 0.5) / (1u64 << 52) as f64;
            score -= self.noise * (-u.ln()).ln();
        }
        self.list.push(Candidate { span, score });
    }
}

#[pymethods]
impl Corpus {
    #[new]
    fn new() -> Self {
        Self {
            buckets: Vec::new(),
        }
    }

    fn add_bucket(&mut self, py: Python<'_>, data: &[u8], ends: Vec<u32>) -> PyResult<()> {
        if data.len() > u32::MAX as usize
            || ends.last().copied().unwrap_or(0) as usize != data.len()
            || ends.windows(2).any(|x| x[0] >= x[1])
        {
            return Err(PyValueError::new_err("Invalid document ends"));
        }
        let text = std::str::from_utf8(data).map_err(|_| PyValueError::new_err("Invalid UTF-8"))?;
        if ends.iter().any(|e| !text.is_char_boundary(*e as usize)) {
            return Err(PyValueError::new_err("Invalid document boundary"));
        }
        let (regex, lines) = py.detach(|| {
            let rows: Vec<_> = ends
                .par_iter()
                .enumerate()
                .map(|(i, &end)| {
                    let start = if i == 0 { 0 } else { ends[i - 1] as usize };
                    let doc = &text[start..end as usize];
                    (
                        ends_for(doc, start, &BYTELEVEL),
                        ends_for(doc, start, &LINE),
                    )
                })
                .collect();
            let mut regex = Vec::new();
            let mut lines = Vec::new();
            for (r, l) in rows {
                regex.extend(r);
                lines.extend(l)
            }
            (regex, lines)
        });
        self.buckets.push(Bucket {
            data: data.to_vec(),
            ends,
            regex,
            lines,
        });
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn sample(
        &self,
        py: Python<'_>,
        quotas: Vec<usize>,
        mode: &str,
        vocab: usize,
        max_length: usize,
        stride: usize,
        bias: f64,
        noise: f64,
        seed: u64,
    ) -> PyResult<(Vec<(String, f64)>, usize, usize, usize)> {
        if quotas.len() != self.buckets.len()
            || !matches!(mode, "regex" | "line256" | "no-regex")
            || vocab < 258
            || max_length < 1
            || stride < 1
            || !bias.is_finite()
            || !noise.is_finite()
            || noise < 0.
            || quotas.iter().map(|&x| x as u128).sum::<u128>() >= u32::MAX as u128
            || quotas
                .iter()
                .zip(&self.buckets)
                .any(|(q, b)| *q > b.data.len() + 3)
        {
            return Err(PyValueError::new_err(
                "Invalid generator settings or unavailable corpus quota",
            ));
        }
        py.detach(|| {
            let mut raw = Vec::with_capacity(quotas.iter().sum());
            let mut spans = Vec::new();
            let mut singles = [0usize; 256];
            for (bi, (b, quota)) in self.buckets.iter().zip(quotas).enumerate() {
                // Match pool-v2: stop if <=3 bytes remain after a whole document,
                // otherwise take the final UTF-8-safe prefix of the next document.
                let whole = b.ends.partition_point(|&end| (end as usize) < quota.saturating_sub(3));
                let start = if whole == 0 {
                    0
                } else {
                    b.ends[whole - 1] as usize
                };
                let mut used = if quota - start <= 3 {
                    start
                } else {
                    quota.min(b.ends.get(whole).map_or(b.data.len(), |&end| end as usize))
                };
                while used < b.data.len() && b.data[used] & 0xc0 == 0x80 {
                    used -= 1
                }
                let offset = raw.len();
                raw.extend_from_slice(&b.data[..used]);
                let boundaries = match mode {
                    "regex" => &b.regex,
                    "line256" => &b.lines,
                    _ => &b.ends,
                };
                let complete = boundaries.partition_point(|&end| end as usize <= start);
                let tail = std::str::from_utf8(&b.data[start..used]).unwrap();
                let last = if tail.is_empty() {
                    vec![]
                } else {
                    match mode {
                        "regex" => ends_for(tail, start, &BYTELEVEL),
                        "line256" => ends_for(tail, start, &LINE),
                        _ => vec![used as u32],
                    }
                };
                let mut pos = (mix(seed ^ bi as u64) as usize) % stride;
                for &end in boundaries[..complete].iter().chain(&last) {
                    let end = end as usize;
                    while pos < end {
                        singles[b.data[pos] as usize] += 1;
                        let len = (end - pos).min(max_length);
                        if len >= 2 {
                            spans.push(Span {
                                start: (offset + pos) as u32,
                                len: len as u32,
                            })
                        }
                        pos = pos.saturating_add(stride);
                    }
                }
            }
            let n = spans.len();
            spans.par_sort_unstable_by(|a, b| a.bytes(&raw).cmp(b.bytes(&raw)));
            let mut c = Candidates {
                text: &raw,
                list: Vec::new(),
                bias,
                noise,
                seed,
            };
            // Each internal compact trie node is a repeated prefix with a distinct
            // occurrence interval. Include full leaves too, even if observed once.
            let mut stack: Vec<(usize, usize)> = Vec::new(); // prefix length, leftmost suffix
            for i in 0..=n {
                let next = if i == n {
                    &[][..]
                } else {
                    spans[i].bytes(&raw)
                };
                let prev = if i == 0 {
                    &[][..]
                } else {
                    spans[i - 1].bytes(&raw)
                };
                let lcp = prev.iter().zip(next).take_while(|(a, b)| a == b).count();
                let mut left = i.saturating_sub(1);
                while stack.last().is_some_and(|&(len, _)| len > lcp) {
                    let (len, first) = stack.pop().unwrap();
                    c.push(
                        Span {
                            start: spans[first].start,
                            len: len as u32,
                        },
                        i - first,
                    );
                    left = first;
                }
                if lcp >= 2 && stack.last().is_none_or(|&(len, _)| len < lcp) {
                    stack.push((lcp, left));
                }
                if i < n && next.len() > lcp {
                    stack.push((next.len(), i));
                }
            }
            let wanted = vocab - 258;
            let available = c.list.len();
            if wanted > available {
                return Err(PyValueError::new_err(format!(
                    "Only {} distinct eligible pieces for this request",
                    available + 258
                )));
            }
            let rank = |a: &Candidate, b: &Candidate| {
                b.score
                    .total_cmp(&a.score)
                    .then_with(|| a.span.bytes(&raw).cmp(b.span.bytes(&raw)))
            };
            if wanted < available {
                c.list.select_nth_unstable_by(wanted, rank);
                c.list.truncate(wanted);
            }
            c.list.par_sort_unstable_by(rank);
            let alphabet = alphabet();
            let mut pieces = vec![("<|unk|>".to_owned(), 0.), ("<|endoftext|>".to_owned(), 0.)];
            pieces.extend(
                alphabet
                    .iter()
                    .zip(singles)
                    .map(|(ch, count)| (ch.to_string(), (count.max(1) as f64).ln())),
            );
            for item in c.list {
                pieces.push((
                    item.span
                        .bytes(&raw)
                        .iter()
                        .map(|&b| alphabet[b as usize])
                        .collect(),
                    item.score,
                ));
            }
            let max = pieces[2..]
                .iter()
                .map(|r| r.1)
                .fold(f64::NEG_INFINITY, f64::max);
            let norm = pieces[2..]
                .iter()
                .map(|r| (r.1 - max).exp())
                .sum::<f64>()
                .ln();
            for (_, s) in &mut pieces[2..] {
                *s = (*s - max) - norm;
                if !(*s as f32).is_finite() {
                    return Err(PyValueError::new_err(
                        "Requested bias or noise exceeds the float32 range of the hypernetwork priors",
                    ));
                }
                // Exact binary fractions keep tied Viterbi paths tied through
                // tokenizer.json reloads. Practical log priors need <=15
                // decimal digits at this resolution.
                *s = (*s * 4096.0).round() / 4096.0;
            }
            Ok((pieces, raw.len(), n, available + 258))
        })
    }
}

#[pymodule]
fn anytok_online_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Corpus>()
}
