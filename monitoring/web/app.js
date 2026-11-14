'use strict';
const $ = id => document.getElementById(id);
const number = value => Number(value || 0).toLocaleString('en-US');
function duration(seconds) {
  if (seconds === null || seconds === undefined) return '—';
  const n = Math.max(0, Math.floor(seconds));
  if (n >= 86400) return `${Math.floor(n / 86400)}d ${Math.floor(n % 86400 / 3600)}h`;
  if (n >= 3600) return `${Math.floor(n / 3600)}h ${Math.floor(n % 3600 / 60)}m`;
  if (n >= 60) return `${Math.floor(n / 60)}m ${n % 60}s`;
  return `${n}s`;
}
let run = new URLSearchParams(location.search).get('run') || '';
let profile = '', worker = '', following = true, busy = false, generation = 0, current = null;
function cell(row, value, className = '') {
  const td = document.createElement('td'); td.textContent = value; td.className = className;
  row.append(td); return td;
}
function label(parent, text, className = '') {
  const span = document.createElement('span'); span.textContent = text; span.className = className; parent.append(span);
}
function button(parent, text, action, className = 'row-button') {
  const el = document.createElement('button'); el.textContent = text; el.className = className;
  el.addEventListener('click', action); parent.append(el); return el;
}
function select(p, w = '') { profile = p; worker = w; generation++; render(current); update(); }
function render(data) {
  if (!data) return;
  const percent = data.total ? Math.min(100, data.completed / data.total * 100) : 0;
  $('name').textContent = data.name;
  $('platform').textContent = data.platform;
  $('stage').textContent = data.stage.replaceAll('_', ' ');
  $('stage').classList.toggle('complete', data.stage === 'complete');
  $('completed').textContent = number(data.completed);
  $('total').textContent = `/ ${number(data.total)} ${data.unit}`;
  $('percent').textContent = `${percent.toFixed(1)}%`;
  $('progress').value = percent;
  $('breakdown').textContent = data.breakdown || `${number(data.completed)} ${data.unit} completed`;
  $('collected').textContent = data.collected === null ? '' : `${number(data.collected)} collected`;
  $('active').textContent = number(data.workers.length);
  $('elapsed').textContent = duration(data.elapsed);
  $('storage').textContent = data.output_bytes ? `${(data.output_bytes / 1e9).toFixed(2)} GB` : '—';
  $('eta').textContent = data.eta;
  $('training-plots').hidden = data.id !== 'anytok-pool-training';
  if (window.refreshLearning) window.refreshLearning(data.id);
  $('trajectory-plots').hidden = data.id !== 'uniform-trajectories';
  const plotConfigs = {
    'local-memory': ['local_memory', 'local_memory_plot_updated_at', 'Local hypernetwork memory diagnostics', 'Real checkpoint and optimizer · complete native vocabularies · batch 128 · microbatch 32 · out-of-memory cases retained.', 'local-memory'],
    'chunked-memory': ['chunked_memory', 'chunked_memory_plot_updated_at', 'Lossless chunking: memory and speed', 'Full native vocabularies · original outputs and all parameter gradients · rejected candidates retained.', 'chunked-memory'],
    'vocab-scaling': ['vocab_scaling', 'vocab_scaling_plot_updated_at', 'Vocabulary scaling and slowdown', 'Measured 32k–43k · dashed extrapolations to 128k · complete native vocabularies · paired warmup and checkpoint reload.', 'vocab-scaling'],
    'frozen-policy-regret': ['frozen_policy_regret', 'frozen_regret_plot_updated_at', 'Canonical-gap versus raw BPB reward · entropy 0.01', 'Only the policy reward changes · same frozen RL20k model, initialization, 1,000 tokenizers and all 288 + 288 passages.', 'frozen-policy-regret-20k'],
    'frozen-policy-regret-entropy0': ['frozen_policy_regret_entropy0', 'frozen_regret_entropy0_plot_updated_at', 'Canonical-gap versus raw BPB reward · entropy 0', 'Only the policy reward changes · same frozen RL20k model, initialization, 1,000 tokenizers and all 288 + 288 passages.', 'frozen-policy-regret-entropy0-20k'],
    'frozen-policy-entropy0': ['frozen_policy_entropy0', 'frozen_entropy_plot_updated_at', 'Entropy 0 versus 0.01', 'Only the entropy coefficient changes · same seed, initialization, 20,000 updates and complete 1,000-tokenizer / 288 + 288 passage reward table.', 'frozen-policy-entropy0-20k'],
    'frozen-policy-search': ['frozen_policy_search', 'frozen_policy_plot_updated_at', 'Can the policy find hard tokenizers?', 'Frozen final RL20k LM and hypernetwork · all 1,000 tokenizers · all 288 original reward passages and 288 held-out check passages · corrected token boundaries.', 'frozen-policy-search-20k'],
    'corrected-pool-check': ['corrected_pool_check', 'corrected_pool_plot_updated_at', 'Corrected broad-text check', 'Final RL 20k · original broad screen and confirmation documents · frozen random 50 including old worst #0635 · no context-dependent passage selection.', 'corrected-pool-check'],
    'natural-transfer-aligned': ['natural_transfer', 'natural_plot_updated_at', 'Natural-text transfer map', 'Final RL20k · corrected canonical boundaries · 36 representative settings plus two controls.', 'natural-transfer-20k'],
    'natural-model-comparison': ['model_comparison', 'model_comparison_plot_updated_at', 'Final trained models · fixed natural-text regime', 'ZeTT, uniform-pool and RL at 20k · identical passages · 229 fixed confirmation stress passages.', 'natural-model-comparison-20k'],
    'pool-regime-comparison': ['random_pool_50', 'pool_reference_plot_updated_at', 'Selected 24 versus a uniform random 50', 'Measured scores · 50 pool tokenizers drawn uniformly without replacement · the same 229 stress passages and three trained models.', 'random-pool-50'],
  };
  const plotConfig = plotConfigs[data.id];
  $('natural-plots').hidden = !plotConfig || !data[plotConfig[1]];
  if (plotConfig && data[plotConfig[1]]) {
    const [filename, timestamp, heading, intro, report] = plotConfig;
    const plot = $('natural-plot-image');
    const version = `${filename}-${data[timestamp]}`;
    $('natural-heading').textContent = heading;
    plot.alt = heading;
    $('natural-intro').textContent = intro;
    $('natural-png-link').href = `/plots/${filename}.png`;
    $('natural-pdf-link').href = `/plots/${filename}.pdf`;
    $('natural-report-link').href = `https://github.com/qingy1337/anytok/tree/zett-modal-20k/reports/${report}`;
    if (plot.dataset.version !== version) {
      plot.src = `/plots/${filename}.png?v=${version}`;
      plot.dataset.version = version;
    }
  }
  $('pool-reference-table').hidden = data.id !== 'pool-regime-comparison' || !data.pool_reference?.models;
  if (data.id === 'pool-regime-comparison' && data.pool_reference?.models) {
    const labels = {zett20k: 'ZeTT 20k', uniform20k: 'Uniform pool 20k', rl20k: 'RL 20k'};
    $('pool-reference-values').replaceChildren(...data.pool_reference.models.map(value => {
      const row = document.createElement('tr');
      cell(row, labels[value.model]);
      for (const key of ['random50_mean_gap', 'hard24_mean_gap', 'hard_minus_random']) {
        cell(row, `${value[key] >= 0 ? '+' : ''}${value[key].toFixed(3)}`);
      }
      return row;
    }));
  }
  $('coverage-plots').hidden = !data.id.startsWith('coverage-') || !data.coverage_plot_updated_at;
  if (data.id.startsWith('coverage-') && data.coverage_plot_updated_at) {
    for (const image of document.querySelectorAll('[data-coverage-figure]')) {
      const version = String(data.coverage_plot_updated_at);
      if (image.dataset.version !== version) {
        image.src = `/plots/coverage_evidence_${image.dataset.coverageFigure}.png?v=${version}`;
        image.dataset.version = version;
      }
    }
  }
  $('rl-ood-plots').hidden = data.id !== 'rl-ood-20k' || !data.rl_ood_plot_updated_at;
  if (data.id === 'rl-ood-20k' && data.rl_ood_plot_updated_at) {
    const plot = $('rl-ood-plot-image');
    const version = String(data.rl_ood_plot_updated_at);
    if (plot.dataset.version !== version) {
      plot.src = `/plots/rl_ood_20k.png?v=${version}`;
      plot.dataset.version = version;
    }
  }
  if (data.id === 'anytok-pool-training' && data.plot_updated_at) {
    const plot = $('training-plot-image');
    const version = String(data.plot_updated_at);
    if (plot.dataset.version !== version) {
      plot.src = `/plots/anytok_pool_learning.png?v=${version}`;
      plot.dataset.version = version;
    }
    plot.hidden = false;
    $('plots-updated').textContent = data.plot_error || `Updated ${new Date(data.plot_updated_at * 1000).toLocaleTimeString()} · Refreshes every minute`;
  }
  $('warnings').textContent = data.warnings.join(' · ');
  $('warnings').hidden = !data.warnings.length;
  $('error').textContent = data.error || ''; $('error').hidden = !data.error;
  $('connection').textContent = data.offline ? 'OFFLINE · CACHED DATA' : 'LIVE · UPDATED ' + new Date(data.at * 1000).toLocaleTimeString([], {hour: '2-digit', minute: '2-digit', second: '2-digit'});
  $('connection').classList.toggle('live', !data.offline && !data.error);
  const age = data.remote_age === null ? 'Awaiting Modal metadata' : `Modal data ${duration(data.remote_age)} old`;
  $('freshness').textContent = `${age}${data.polling ? ' · Refreshing' : ''} · Local artifacts every 2s · Modal every 30s · Credits every 5m`;
  if ($('run').dataset.signature !== JSON.stringify(data.runs)) {
    $('run').replaceChildren(...data.runs.map(r => {
      const option = document.createElement('option'); option.value = r.id;
      option.textContent = `${r.name} (${number(r.completed)}/${number(r.total)})`; return option;
    }));
    $('run').dataset.signature = JSON.stringify(data.runs);
  }
  $('run').value = data.id;
  $('profiles').replaceChildren(...data.profiles.map(p => {
    const row = document.createElement('tr'); row.classList.toggle('selected', profile === p.name);
    const name = cell(row, ''); button(name, p.name, () => select(profile === p.name && !worker ? '' : p.name));
    cell(row, p.platform);
    cell(row, p.done === null ? '—' : `${number(p.done)} / ${number(p.total)}`, 'mono');
    cell(row, number(p.active), 'mono');
    cell(row, p.remaining === null || p.remaining === undefined ? '—' : `≈ $${p.remaining.toFixed(2)}`, 'mono');
    const stale = p.inferred || p.error;
    const state = cell(row, p.state + (stale ? ' · stale / inferred' : ''), `state ${p.active && !stale ? 'running' : ''}`);
    if (p.error) state.title = p.error;
    return row;
  }));
  const workers = data.workers.filter(w => !profile || w.profile === profile);
  $('worker-count').textContent = workers.length;
  $('workers').replaceChildren(...workers.map(w => {
    const row = document.createElement('tr'); row.classList.toggle('selected', worker === w.id);
    const name = cell(row, '');
    button(name, w.id, () => select(w.profile, worker === w.id ? '' : w.id), 'row-button worker-id');
    label(name, `${w.profile} · ${w.platform}`, 'sub');
    const task = cell(row, w.index === undefined ? w.task || 'Starting task' : `#${String(w.index).padStart(4, '0')} · ${w.algorithm || '—'}`);
    if (w.pretokenizer) label(task, w.pretokenizer, 'sub');
    cell(row, w.vocab_size ? number(w.vocab_size) : '—', 'mono');
    cell(row, `${w.inferred ? '≈ ' : ''}${duration(w.elapsed)}`, 'mono');
    cell(row, w.eta || 'Awaiting measurements');
    cell(row, w.state, 'state running');
    return row;
  }));
  if (!workers.length) {
    const row = document.createElement('tr'); const td = cell(row, data.stage === 'complete' ? 'This experiment is complete.' : 'No active workers reported for this selection.', 'empty'); td.colSpan = 6; $('workers').append(row);
  }
  $('clear-filter').hidden = !profile && !worker;
  $('log-selection').textContent = worker ? `${profile} / ${worker.slice(-12)}` : profile || 'All profiles';
}
async function update() {
  if (busy) return;
  busy = true;
  const version = generation;
  const query = new URLSearchParams();
  if (run) query.set('run', run);
  if (profile) query.set('profile', profile);
  if (worker) query.set('worker', worker);
  try {
    const response = await fetch(`/api/state?${query}`, {signal: AbortSignal.timeout(10000)});
    if (!response.ok) throw new Error(`Server returned ${response.status}`);
    const data = await response.json();
    if (version !== generation) return;
    current = data; render(data);
    if (following && $('logs').textContent !== data.logs) {
      $('logs').textContent = data.logs; $('logs').scrollTop = $('logs').scrollHeight;
    }
  } catch (error) {
    if (version !== generation) return;
    $('connection').textContent = 'CONNECTION LOST'; $('connection').classList.remove('live');
    $('error').hidden = false; $('error').textContent = `Refresh failed. Showing the last received data. ${error.message}`;
  } finally {
    busy = false;
    if (version !== generation) update();
  }
}
$('run').addEventListener('change', () => {
  run = $('run').value; profile = worker = ''; generation++;
  history.replaceState(null, '', `?run=${encodeURIComponent(run)}`); update();
});
$('clear-filter').addEventListener('click', () => select(''));
$('follow').addEventListener('click', () => {
  following = !following; $('follow').textContent = following ? 'Pause following' : 'Resume following';
  $('follow').setAttribute('aria-pressed', String(following)); if (following) update();
});
update(); setInterval(update, 2000);
