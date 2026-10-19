import { PRESETS, LIMITS } from './data.js'

const $ = (s, el = document) => el.querySelector(s)
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]))
const pct = (x) => Math.round(x * 100)
const words = (t) => (t.trim() ? t.trim().split(/\s+/).length : 0)
const TIER_COLOR = { dup: 'var(--hot)', high: 'var(--accent)', mid: 'var(--warn)', low: 'var(--mute)' }

const state = { tab: 'compare', health: null, a: PRESETS[0].a, b: PRESETS[0].b, preset: 'para', res: null, busy: false, err: '' }

/* ------------------------------------------------------------------ shell */
function renderStatus() {
  const el = $('#status'), h = state.health
  el.className = 'status ' + (h ? 'ok' : h === false ? 'bad' : '')
  $('span', el).textContent = h ? `API · ${h.mode} mode` : h === false ? 'API offline' : 'connecting…'
}

function setTab(tab) {
  state.tab = tab
  document.querySelectorAll('#tabs button').forEach((b) => {
    b.classList.toggle('on', b.dataset.tab === tab)
    b.setAttribute('aria-selected', b.dataset.tab === tab)
  })
  const view = $('#view')
  if (tab === 'compare') { view.innerHTML = compareView(); bindCompare(); renderResults() }
  if (tab === 'arch') view.innerHTML = archView()
  window.scrollTo({ top: 0 })
}

document.querySelectorAll('#tabs button').forEach((b) => b.addEventListener('click', () => setTab(b.dataset.tab)))
fetch('/api/health').then((r) => r.json()).then((h) => { state.health = h }).catch(() => { state.health = false }).finally(renderStatus)

/* ------------------------------------------------------------------ compare */
function compareView() {
  const doc = (n, v) => `
    <div class="doc">
      <div class="doc-head"><span>Document ${n}</span>
        <span class="muted"><span id="wc${n}">${words(v)}</span> words
          <label class="upload">upload .txt<input type="file" accept=".txt,text/plain" data-up="${n}" /></label></span></div>
      <textarea id="doc${n}" spellcheck="false" placeholder="Paste document ${n}…">${esc(v)}</textarea>
    </div>`
  return `
    <section class="hero">
      <h1>How alike are these two documents?</h1>
      <p>Four similarity models run in parallel and a sigmoid ensemble fuses them into one interpretable score — then every sentence is aligned across the pair.</p>
    </section>
    <section class="card">
      <div class="presets"><span class="muted">Try an example</span>
        ${PRESETS.map((p) => `<button class="chip ${state.preset === p.id ? 'on' : ''}" data-preset="${p.id}">${p.label}</button>`).join('')}
      </div>
      <div class="docs">${doc('A', state.a)}${doc('B', state.b)}</div>
      <div class="actions"><button class="primary" id="run">Compare documents</button><span class="err" id="err"></span></div>
    </section>
    <div id="results" class="results"></div>`
}

function bindCompare() {
  const ta = { A: $('#docA'), B: $('#docB') }
  const sync = () => {
    state.a = ta.A.value; state.b = ta.B.value
    $('#wcA').textContent = words(state.a); $('#wcB').textContent = words(state.b)
    $('#run').disabled = state.busy || !state.a.trim() || !state.b.trim()
  }
  const clearPreset = () => { state.preset = null; document.querySelectorAll('.chip').forEach((c) => c.classList.remove('on')) }
  ;['A', 'B'].forEach((n) => ta[n].addEventListener('input', () => { sync(); clearPreset() }))
  document.querySelectorAll('[data-preset]').forEach((c) => c.addEventListener('click', () => {
    const p = PRESETS.find((x) => x.id === c.dataset.preset)
    ta.A.value = p.a; ta.B.value = p.b; sync()
    state.preset = p.id
    document.querySelectorAll('.chip').forEach((x) => x.classList.toggle('on', x === c))
  }))
  document.querySelectorAll('[data-up]').forEach((inp) => inp.addEventListener('change', async () => {
    const f = inp.files?.[0]; if (!f) return
    ta[inp.dataset.up].value = await f.text(); sync(); clearPreset(); inp.value = ''
  }))
  $('#run').addEventListener('click', run)
  sync()
}

async function run() {
  const btn = $('#run'), err = $('#err')
  state.busy = true; btn.disabled = true; btn.textContent = 'Analysing…'; err.textContent = ''
  try {
    const r = await fetch('/api/similarity', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ doc_a: state.a, doc_b: state.b }) })
    const j = await r.json()
    if (!r.ok) throw new Error(j.error || 'Request failed')
    state.res = j
    renderResults()
    setTimeout(() => $('#results').scrollIntoView({ behavior: 'smooth', block: 'start' }), 60)
  } catch (e) {
    err.textContent = e.message === 'Failed to fetch' ? 'Cannot reach the API. Start it with: python api/app.py' : e.message
  } finally {
    state.busy = false; btn.textContent = 'Compare documents'; btn.disabled = !state.a.trim() || !state.b.trim()
  }
}

function gauge(r) {
  const C = Math.PI * 80, color = TIER_COLOR[r.tier]
  return `<div class="gauge">
    <svg viewBox="0 0 200 118" role="img" aria-label="Similarity ${pct(r.score)} percent">
      <path d="M20 100 A80 80 0 0 1 180 100" class="g-track"/>
      <path d="M20 100 A80 80 0 0 1 180 100" class="g-fill" style="stroke:${color};stroke-dasharray:${C * r.score} ${C}"/>
      <text x="100" y="92" text-anchor="middle" class="g-num">${pct(r.score)}<tspan class="g-pct">%</tspan></text>
    </svg>
    <div class="g-verdict" style="color:${color}">${esc(r.verdict)}</div></div>`
}

function renderResults() {
  const box = $('#results'), r = state.res
  if (!box) return
  if (!r) { box.innerHTML = ''; return }
  const terms = r.models.map((m) => `${m.weight.toFixed(1)}×${m.score.toFixed(2)}`).join(' + ')
  const lite = r.models.some((m) => m.approx)
  box.innerHTML = `
    <section class="card score-card">
      ${gauge(r)}
      <div>
        <div class="badges">
          <span class="badge">${r.latency_ms} ms</span><span class="badge">${r.cached ? 'cache hit' : 'cache miss'}</span>
          <span class="badge">${r.stats.shared_terms} shared terms</span><span class="badge">${esc(r.mode)} mode</span>
        </div>
        <div class="eq"><div class="eq-label">Ensemble · Eq. (1)</div>
          <code>ŷ = σ( ${r.bias} + ${terms} ) = σ( ${r.logit.toFixed(2)} ) = <b>${r.score.toFixed(3)}</b></code></div>
        <div class="row-actions"><button class="ghost" id="csv">Export CSV</button><button class="ghost" id="json">Copy JSON</button></div>
      </div>
    </section>
    <section class="card">
      <h2>Per-model scores</h2>
      <p class="sub">The ensemble exposes every member’s opinion, so a decision can be traced back.${lite ? ' Models tagged <em class="lite">lite</em> are lightweight stand-ins for the paper’s transformers; run in full mode for the real ones.' : ''}</p>
      <div class="models">${r.models.map((m) => `
        <div class="model">
          <div class="m-head"><span class="m-name">${esc(m.label)}${m.approx ? `<em title="${esc(m.desc)}">lite</em>` : ''}</span><span class="m-score">${m.score.toFixed(3)}</span></div>
          <div class="bar"><span style="width:${pct(m.score)}%"></span></div>
          <div class="m-foot"><span title="${esc(m.desc)}">w = ${m.weight.toFixed(1)}</span><span>${m.latency_ms} ms</span></div>
        </div>`).join('')}</div>
    </section>
    <section class="card">
      <h2>Sentence alignment</h2>
      <p class="sub">Rows are sentences of A, columns are sentences of B. Darker means more similar.${r.alignment.truncated ? ' (First 14 sentences shown.)' : ''}</p>
      <div id="heat"></div>
    </section>`
  $('#csv').onclick = downloadCSV
  $('#json').onclick = (e) => { navigator.clipboard?.writeText(JSON.stringify(r, null, 2)); e.target.textContent = 'Copied ✓'; setTimeout(() => (e.target.textContent = 'Copy JSON'), 1400) }
  renderHeat(r.alignment)
}

function renderHeat(al) {
  const box = $('#heat')
  if (!al.a.length || !al.b.length) { box.innerHTML = '<p class="muted">Add a few sentences to see an alignment.</p>'; return }
  const cell = Math.max(26, Math.min(46, Math.floor(420 / Math.max(al.a.length, al.b.length))))
  let grid = '<span></span>' + al.b.map((_, j) => `<span class="axis">B${j + 1}</span>`).join('')
  al.a.forEach((_, i) => {
    grid += `<span class="axis">A${i + 1}</span>`
    al.b.forEach((__, j) => {
      const s = al.matrix[i][j], best = al.best[i]?.b === j
      grid += `<button class="cell${best ? ' best' : ''}" data-i="${i}" data-j="${j}" style="height:${cell}px;background:color-mix(in srgb,var(--accent) ${pct(s)}%,var(--card))" aria-label="A${i + 1} to B${j + 1}: ${s.toFixed(2)}"></button>`
    })
  })
  box.innerHTML = `<div class="heat">
    <div class="heat-grid" style="grid-template-columns:auto repeat(${al.b.length},${cell}px)">${grid}</div>
    <div class="heat-detail" id="hd"></div></div>`
  const show = (i, j) => {
    document.querySelectorAll('.cell').forEach((c) => c.classList.toggle('on', +c.dataset.i === i && +c.dataset.j === j))
    $('#hd').innerHTML = `<div class="hd-score">A${i + 1} ↔ B${j + 1} <b>${al.matrix[i][j].toFixed(2)}</b></div>
      <p><span class="tag">A</span>${esc(al.a[i])}</p><p><span class="tag b">B</span>${esc(al.b[j])}</p>
      <div class="legend"><span>0</span><i></i><span>1</span><small>outlined = best match for each A sentence</small></div>`
  }
  document.querySelectorAll('.cell').forEach((c) => {
    const go = () => show(+c.dataset.i, +c.dataset.j)
    c.addEventListener('mouseenter', go); c.addEventListener('focus', go); c.addEventListener('click', go)
  })
  show(al.best[0]?.a ?? 0, al.best[0]?.b ?? 0)
}

function downloadCSV() {
  const r = state.res
  const rows = [['metric', 'value'], ['ensemble_score', r.score], ['verdict', r.verdict], ...r.models.map((m) => [m.key + '_score', m.score]), [],
    ['a_idx', 'a_sentence', 'best_b_idx', 'b_sentence', 'score'],
    ...r.alignment.best.map((x) => [x.a + 1, r.alignment.a[x.a], x.b + 1, r.alignment.b[x.b], x.score])]
  const csv = rows.map((row) => row.map((c) => `"${String(c ?? '').replace(/"/g, '""')}"`).join(',')).join('\n')
  const url = URL.createObjectURL(new Blob([csv], { type: 'text/csv' }))
  Object.assign(document.createElement('a'), { href: url, download: 'simdoc-result.csv' }).click()
  URL.revokeObjectURL(url)
}

/* ------------------------------------------------------------------ architecture */
function archView() {
  const mods = [['TF-IDF', 'Sparse, 2 ms, interpretable'], ['BERT', 'Mean-pooled, 768-d'], ['RoBERTa', 'Mean-pooled, 768-d'], ['SBERT', 'Siamese, tuned for STS']]
  return `
    <section class="hero"><h1>How SimDoc works</h1>
      <p>Classical and neural similarity run side by side; a small learned meta-model decides how much to trust each.</p></section>
    <section class="card"><div class="flow">
      <div class="node"><b>Input</b><span>text · URL · file</span></div><i class="arrow"></i>
      <div class="node"><b>Preprocess</b><span>tokenize · lowercase · clean</span></div><i class="arrow"></i>
      <div class="stack">${mods.map(([t, s]) => `<div class="node small"><b>${t}</b><span>${s}</span></div>`).join('')}</div><i class="arrow"></i>
      <div class="node hot"><b>Ensemble</b><span>σ(w₀ + Σ wᵢsᵢ)</span><em>+ 24 h result cache</em></div><i class="arrow"></i>
      <div class="stack"><div class="node small"><b>REST API</b><span>POST /similarity</span></div><div class="node small"><b>Dashboard</b><span>heatmap · export</span></div></div>
    </div></section>
    <div class="two">
      <section class="card"><h2>Call the API</h2>
<pre>curl -X POST http://localhost:5000/similarity \\
  -H "Content-Type: application/json" \\
  -d '{"doc_a": "Heavy rain hit Mumbai today.",
       "doc_b": "Mumbai received intense rainfall."}'</pre>
        <p class="sub">Returns the ensemble <code>score</code>, a <code>verdict</code>, each model’s score / weight / latency, and a sentence-level alignment matrix.</p></section>
      <section class="card"><h2>Known limitations <small>from the paper</small></h2>
        <ul class="limits">${LIMITS.map(([t, d]) => `<li><b>${t}.</b> ${d}</li>`).join('')}</ul></section>
    </div>
    <section class="card note"><h2>About this implementation</h2>
      <p>A lightweight reference implementation of the paper’s design. In <b>lite mode</b> (default, no downloads) TF-IDF is the real baseline, while the BERT / RoBERTa / SBERT slots use fast stand-ins (character n-grams, word-order matching, synonym-normalised concepts) and the ensemble weights are hand-set rather than trained. Switch to <b>full mode</b> (<code>SIMDOC_MODE=full</code>) to run the real transformer models. Redis is replaced by an in-process TTL cache.</p></section>`
}

setTab('compare')
