'use strict';
(() => {
  const get = id => document.getElementById(id);
  const NS = 'http://www.w3.org/2000/svg';
  const colors = {green: '#2f8058', blue: '#397db5', orange: '#b57a3b', gray: '#9a9a9a', purple: '#8269a7'};
  let activeRun = '', requested = '', lastFetch = 0, data = null, busy = false;
  const node = (tag, text, className) => {
    const el = document.createElement(tag);
    if (text !== undefined) el.textContent = text;
    if (className) el.className = className;
    return el;
  };
  function svgNode(tag, attrs = {}, text) {
    const el = document.createElementNS(NS, tag);
    for (const [key, value] of Object.entries(attrs)) el.setAttribute(key, value);
    if (text !== undefined) el.textContent = text;
    return el;
  }
  const fmtStep = value => value >= 1000 ? `${+(value / 1000).toFixed(1)}k` : String(Math.round(value));
  const pct = value => `${(value * 100).toFixed(1)}%`;
  function legend(parent, series) {
    const el = node('div', undefined, 'chart-legend');
    for (const s of series) {
      const item = node('span', undefined, 'legend-item');
      const swatch = svgNode('svg', {width: 20, height: 10, 'aria-hidden': true});
      swatch.append(svgNode('line', {x1: 0, x2: 20, y1: 5, y2: 5, stroke: s.color,
        'stroke-width': 2, 'stroke-dasharray': s.dash || ''}));
      item.append(swatch, node('span', s.name)); el.append(item);
    }
    parent.append(el);
  }
  function chart(id, series, options = {}) {
    const host = get(id);
    const signature = JSON.stringify([series, options]);
    if (host.dataset.signature === signature) return;
    host.dataset.signature = signature;
    host.replaceChildren();
    const available = series.flatMap(s => s.points).filter(p => Number.isFinite(p[1]));
    if (!available.length) { host.append(node('p', 'Waiting for the next logged measurement…', 'intro chart-empty')); return; }
    const w = 600, h = 280, left = 55, right = 16, top = 15, bottom = 42;
    let xmin = Math.min(...available.map(p => p[0])), xmax = Math.max(...available.map(p => p[0]));
    if (xmin === xmax) { xmin = Math.max(0, xmin - 250); xmax += 250; }
    let ymin = Math.min(...available.map(p => p[1])), ymax = Math.max(...available.map(p => p[1]));
    const pad = Math.max((ymax - ymin) * .09, options.minPad || .003);
    ymin = options.zero ? 0 : ymin - pad; ymax += pad;
    const x = value => left + (value - xmin) / (xmax - xmin) * (w - left - right);
    const y = value => h - bottom - (value - ymin) / (ymax - ymin) * (h - top - bottom);
    const format = options.format === 'count' ? value => Math.round(value).toLocaleString('en-US')
      : options.format === 'percent' ? value => `${value.toFixed(2)}%` : value => value.toFixed(3);
    const svg = svgNode('svg', {viewBox: `0 0 ${w} ${h}`, role: 'img', tabindex: 0,
      'aria-label': options.label + '. Use left and right arrow keys to inspect steps.'});
    svg.append(svgNode('title', {}, options.label));
    for (let i = 0; i <= 4; i++) {
      const value = ymin + (ymax - ymin) * i / 4;
      svg.append(svgNode('line', {x1: left, x2: w - right, y1: y(value), y2: y(value), class: 'chart-gridline'}));
      svg.append(svgNode('text', {x: left - 8, y: y(value) + 4, 'text-anchor': 'end', class: 'chart-axis'}, format(value)));
      const step = xmin + (xmax - xmin) * i / 4;
      svg.append(svgNode('text', {x: x(step), y: h - 20, 'text-anchor': 'middle', class: 'chart-axis'}, fmtStep(step)));
    }
    svg.append(svgNode('text', {x: (left + w - right) / 2, y: h - 2, 'text-anchor': 'middle', class: 'chart-axis'}, 'Training step'));
    for (const s of series) {
      const points = s.points.filter(p => Number.isFinite(p[1]));
      if (!points.length) continue;
      svg.append(svgNode('path', {d: points.map((p, i) => `${i ? 'L' : 'M'}${x(p[0])},${y(p[1])}`).join(' '),
        fill: 'none', stroke: s.color, 'stroke-width': 2.2, 'stroke-dasharray': s.dash || '',
        'stroke-linecap': 'round', 'stroke-linejoin': 'round'}));
      if (points.length < 50) for (const p of points) {
        svg.append(svgNode('circle', {cx: x(p[0]), cy: y(p[1]), r: 3, fill: s.color}));
      }
    }
    const cursor = svgNode('line', {x1: left, x2: left, y1: top, y2: h - bottom, class: 'chart-cursor', visibility: 'hidden'});
    svg.append(cursor);
    const tip = node('div', undefined, 'chart-tooltip'); tip.hidden = true;
    const steps = [...new Set(available.map(p => p[0]))].sort((a, b) => a - b);
    let selected = steps.length - 1;
    function inspect(step) {
      selected = steps.reduce((best, v, i) => Math.abs(v - step) < Math.abs(steps[best] - step) ? i : best, 0);
      const chosen = steps[selected];
      cursor.setAttribute('x1', x(chosen)); cursor.setAttribute('x2', x(chosen)); cursor.setAttribute('visibility', 'visible');
      tip.replaceChildren(node('strong', `Step ${chosen.toLocaleString('en-US')}`));
      for (const s of series) {
        if (!s.points.length) continue;
        const point = s.points.reduce((a, b) => Math.abs(a[0] - chosen) <= Math.abs(b[0] - chosen) ? a : b);
        if (!Number.isFinite(point[1])) continue;
        tip.append(node('div', `${s.name}: ${format(point[1])}${point[0] === chosen ? '' : ` (step ${point[0]})`}`));
      }
      tip.style.left = chosen > (xmin + xmax) / 2 ? 'auto' : `${Math.max(2, x(chosen) / w * 100)}%`;
      tip.style.right = chosen > (xmin + xmax) / 2 ? `${Math.max(2, (w - x(chosen)) / w * 100)}%` : 'auto';
      tip.hidden = false;
    }
    svg.addEventListener('pointermove', e => {
      const px = (e.clientX - svg.getBoundingClientRect().left) / svg.getBoundingClientRect().width * w;
      inspect(xmin + (px - left) / (w - left - right) * (xmax - xmin));
    });
    svg.addEventListener('pointerleave', () => { if (document.activeElement !== svg) { tip.hidden = true; cursor.setAttribute('visibility', 'hidden'); } });
    svg.addEventListener('focus', () => inspect(steps[selected]));
    svg.addEventListener('blur', () => { tip.hidden = true; cursor.setAttribute('visibility', 'hidden'); });
    svg.addEventListener('keydown', e => {
      if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(e.key)) return;
      e.preventDefault();
      selected = e.key === 'Home' ? 0 : e.key === 'End' ? steps.length - 1 : Math.max(0, Math.min(steps.length - 1, selected + (e.key === 'ArrowRight' ? 1 : -1)));
      inspect(steps[selected]);
    });
    host.append(svg, tip); legend(host, series);
  }
  function render() {
    if (!data) return;
    const includeInitial = get('show-initial').checked;
    const scoreSeries = metric => ['rl', 'uniform'].flatMap(run => [0, 1].map(i => ({
      name: `${run === 'rl' ? 'RL' : 'Uniform'} · T${i + 1}`, color: i ? colors.blue : colors.green,
      dash: run === 'uniform' ? '6 4' : '',
      points: data.curves[run].filter(p => includeInitial || p.step > 0).map(p => [p.step, p.tokenizers[i][metric]])
    })));
    const scores = scoreSeries('bpb');
    if (get('show-canonical').checked) for (const run of ['rl', 'uniform']) for (let i = 0; i < 2; i++) {
      scores.push({name: `${run === 'rl' ? 'RL' : 'Uniform'} canonical · T${i + 1}`, color: i ? '#b59e7a' : colors.gray,
        dash: run === 'uniform' ? '6 4' : '', points: data.curves[run].filter(p => includeInitial || p.step > 0).map(p => [p.step, p.tokenizers[i].canonical])});
    }
    chart('sanity-bpb', scores, {label: 'Sanity continuation BPB'});
    chart('sanity-gap', scoreSeries('gap'), {label: 'Sanity transfer gap in BPB'});
    const latest = data.latest_step;
    chart('policy-spread', [
      {name: 'Effective tokenizer count', color: colors.green, points: [[0, data.pool_size], ...data.policy.map(p => [p.step, p.policy_effective_tokenizers])]},
      {name: 'Unique tokenizers sampled', color: colors.blue, points: [[0, 0], ...data.policy.map(p => [p.step, p.policy_unique_tokenizers])]},
      {name: 'Pool size', color: colors.gray, dash: '5 4', points: [[0, data.pool_size], [latest, data.pool_size]]}
    ], {label: 'Policy effective count and observed coverage', zero: true, minPad: 10, format: 'count'});
    chart('policy-peak', [
      {name: 'Highest probability', color: colors.orange, points: [[0, 100 / data.pool_size], ...data.policy.map(p => [p.step, p.policy_max_probability * 100])]},
      {name: 'Uniform', color: colors.gray, dash: '5 4', points: [[0, 100 / data.pool_size], [latest, 100 / data.pool_size]]}
    ], {label: 'Maximum single tokenizer probability', zero: true, format: 'percent'});
    const last = data.latest_policy;
    const metrics = get('policy-metrics'); metrics.replaceChildren();
    if (last) for (const [name, value] of [
      ['EFFECTIVE TOKENIZERS', Math.round(last.policy_effective_tokenizers).toLocaleString('en-US')],
      ['UNIQUE SAMPLED', `${last.policy_unique_tokenizers} / ${data.pool_size}`],
      ['HIGHEST PROBABILITY', `${(last.policy_max_probability * 100).toFixed(3)}%`],
      ['ENTROPY (NATS)', last.policy_entropy.toFixed(3)]]) {
      const item = node('div'); item.append(node('dt', name), node('dd', value)); metrics.append(item);
    }
    get('learning-freshness').textContent = `Through step ${latest.toLocaleString('en-US')} · updates every 10s`;
    const recent = Object.entries(data.recent_exposure.algorithm || {}).sort((a, b) => b[1] - a[1]);
    get('recent-selection').textContent = `Actual selections in the latest ${data.recent_steps} updates: ${recent.map(([name, value]) => `${names[name]} ${pct(value)}`).join(' · ')}. These are sampled actions, not policy probabilities.`;
    const warning = [...data.warnings, data.snapshot_error ? `Distribution refresh: ${data.snapshot_error}` : ''].filter(Boolean).join(' · ');
    get('learning-error').textContent = warning; get('learning-error').hidden = !warning;
    const picker = get('policy-snapshot'), previous = picker.value || 'latest';
    picker.replaceChildren();
    const option = node('option', 'Latest available'); option.value = 'latest'; picker.append(option);
    for (const snapshot of data.snapshots) {
      const option = node('option', `Step ${snapshot.step.toLocaleString('en-US')}`); option.value = String(snapshot.step); picker.append(option);
    }
    picker.value = [...picker.options].some(o => o.value === previous) ? previous : 'latest';
    renderDistribution();
  }
  const names = {bpe: 'BPE', wordpiece: 'WordPiece', unigram: 'Unigram', regex: 'Regex', line256: 'Line 256', 'no-regex': 'No regex'};
  function renderDistribution() {
    if (!data) return;
    const chosen = get('policy-snapshot').value;
    const snapshot = chosen === 'latest' ? data.snapshots.at(-1) : data.snapshots.find(s => String(s.step) === chosen);
    const legendHost = get('distribution-legend'); legendHost.replaceChildren();
    legend(legendHost, [{name: 'Policy probability', color: colors.green}, {name: 'Uniform pool', color: '#b7b7b7'}, {name: 'Observed selections so far', color: colors.orange}]);
    for (const field of ['algorithm', 'pretokenizer', 'vocab', 'merge_noise', 'domain']) {
      const host = get(`dist-${field}`); host.replaceChildren();
      if (!snapshot) { host.append(node('p', 'Waiting for the first cloud snapshot…', 'intro chart-empty')); continue; }
      const groups = snapshot.groups[field] || {};
      const maximum = Math.max(.01, ...Object.values(groups).flatMap(r => [r.policy, r.uniform, r.observed || 0]));
      for (const [key, values] of Object.entries(groups)) {
        const row = node('div', undefined, 'bar-row');
        const title = field === 'vocab' ? ['32–56k', '56–80k', '80–104k', '104–128k'][Number(key)] : field === 'merge_noise' ? ['0–25%', '25–50%', '50–75%'][Number(key)] : names[key] || key;
        row.append(node('span', title, 'bar-label'));
        const stack = node('div', undefined, 'bar-stack');
        for (const [metric, label, color] of [['policy', 'Policy', colors.green], ['uniform', 'Uniform', '#b7b7b7'], ['observed', 'Observed', colors.orange]]) {
          if (values[metric] === null) continue;
          const track = node('div', undefined, 'bar-track');
          const bar = node('span', undefined, 'bar-fill'); bar.style.width = `${values[metric] / maximum * 100}%`; bar.style.backgroundColor = color;
          track.title = `${title} · ${label}: ${pct(values[metric])}`;
          track.setAttribute('aria-label', track.title); track.append(bar); stack.append(track);
        }
        row.append(stack, node('span', pct(values.policy), 'bar-value')); host.append(row);
      }
    }
    get('distribution-note').textContent = snapshot ? `Policy snapshot at step ${snapshot.step.toLocaleString('en-US')} · Top 100 tokenizers hold ${pct(snapshot.top_100_probability)} probability · Mean vocabulary ${Math.round(snapshot.mean_vocab).toLocaleString('en-US')}. Observed bars count all selections through this checkpoint. Hover bars for exact values.` : 'Waiting for checkpoint metadata; training continues independently.';
    get('policy-top').replaceChildren(...(snapshot?.top_tokenizers || []).map(t => {
      const tr = node('tr');
      for (const value of [`#${t.index}`, `${names[t.algorithm]} · ${names[t.pretokenizer]} · ${Math.round(t.vocab_size / 1000)}k`, `${(t.probability * 100).toFixed(3)}%`, t.selections]) tr.append(node('td', value));
      return tr;
    }));
  }
  window.refreshLearning = async runId => {
    get('learning').hidden = runId !== 'anytok-pool-settings-rl-training';
    if (runId !== activeRun) { activeRun = runId; lastFetch = 0; }
    if (get('learning').hidden || busy || Date.now() - lastFetch < 10000) return;
    busy = true; requested = runId;
    try {
      const response = await fetch(`/api/learning?run=${encodeURIComponent(runId)}`, {signal: AbortSignal.timeout(10000)});
      if (!response.ok) throw new Error(`Server returned ${response.status}`);
      const result = await response.json();
      if (requested !== activeRun) return;
      data = result; render(); lastFetch = Date.now();
    } catch (error) {
      if (requested === activeRun) { get('learning-error').hidden = false; get('learning-error').textContent = `Learning refresh failed; showing previous measurements. ${error.message}`; }
    } finally { busy = false; }
  };
  get('show-initial').addEventListener('change', render);
  get('show-canonical').addEventListener('change', render);
  get('policy-snapshot').addEventListener('change', renderDistribution);
})();
