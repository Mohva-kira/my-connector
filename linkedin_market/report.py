"""Rapport HTML autonome (Chart.js) + JSON de résultats.

Le JSON a la même structure que les données injectées dans la page : une
future application web pourra le consommer tel quel.
"""

from __future__ import annotations

import html
import json
import re
from datetime import datetime
from pathlib import Path

from . import config

_TEMPLATE = """<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Marché LinkedIn — __KEYWORD__</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/marked/12.0.2/marked.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/dompurify/3.1.6/purify.min.js"></script>
<style>
:root {
  color-scheme: light;
  --bg: #f6f6f4; --surface: #fcfcfb; --border: #e3e2de;
  --text: #0b0b0b; --text-2: #52514e; --muted: #8a8984; --grid: #ecebe7;
  --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a; --neg: #e34948;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --bg: #121211; --surface: #1a1a19; --border: #2e2e2b;
    --text: #ffffff; --text-2: #c3c2b7; --muted: #8f8e86; --grid: #2a2a27;
    --s1: #3987e5; --s2: #d95926; --s3: #199e70; --neg: #e66767;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --bg: #121211; --surface: #1a1a19; --border: #2e2e2b;
  --text: #ffffff; --text-2: #c3c2b7; --muted: #8f8e86; --grid: #2a2a27;
  --s1: #3987e5; --s2: #d95926; --s3: #199e70; --neg: #e66767;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--text);
  font: 15px/1.55 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
main { max-width: 1180px; margin: 0 auto; padding: 28px 16px 64px; }
h1 { font-size: 26px; margin: 0 0 4px; }
h2 { font-size: 17px; margin: 0 0 4px; }
.sub { color: var(--text-2); margin: 0 0 24px; }
.hint { color: var(--muted); font-size: 13px; margin: 0 0 12px; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin-bottom: 20px; }
.tile, .card { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; }
.tile { padding: 14px 16px; }
.tile b { display: block; font-size: 26px; font-variant-numeric: tabular-nums; }
.tile span { color: var(--text-2); font-size: 13px; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 520px), 1fr)); gap: 16px; }
.card { padding: 18px; margin-bottom: 16px; min-width: 0; }
.chart { position: relative; height: 360px; }
.chart.short { height: 260px; }
.legend { display: flex; gap: 16px; font-size: 13px; color: var(--text-2); margin-bottom: 8px; }
.legend i { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 6px; vertical-align: middle; }
.table-wrap { overflow-x: auto; }
table { width: 100%; border-collapse: collapse; font-size: 14px; }
th, td { text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--border); vertical-align: top; }
th { color: var(--text-2); font-weight: 600; }
td.num { font-variant-numeric: tabular-nums; white-space: nowrap; }
a { color: var(--s1); }
.tag { display: inline-block; padding: 1px 8px; margin: 2px 4px 2px 0; border-radius: 999px;
  border: 1px solid var(--border); font-size: 12px; color: var(--text-2); }
.analysis { max-width: 820px; }
ul.summary { margin: 6px 0 0; padding-left: 18px; color: var(--text-2); font-size: 13px; }
.analysis h2 { margin-top: 20px; }
.note { color: var(--text-2); font-style: italic; }
details summary { cursor: pointer; color: var(--text-2); font-size: 13px; margin-top: 8px; }
</style>
</head>
<body>
<main>
  <h1>Marché LinkedIn — __KEYWORD__</h1>
  <p class="sub" id="sub"></p>
  <div class="tiles" id="tiles"></div>

  <div class="card analysis">
    <h2>Interprétation</h2>
    <p class="hint" id="interp-src"></p>
    <div id="analysis"></div>
  </div>

  <div class="grid">
    <div class="card"><h2>Termes les plus fréquents</h2>
      <p class="hint">Nombre de documents (offres, posts, profils) contenant le terme.</p>
      <div class="chart"><canvas id="c-terms"></canvas></div></div>
    <div class="card"><h2>Compétences demandées dans les offres</h2>
      <p class="hint">Part des offres qui citent la compétence.</p>
      <div class="chart"><canvas id="c-skills"></canvas></div></div>
  </div>

  <div class="grid">
    <div class="card"><h2>Termes et compétences en hausse</h2>
      <p class="hint" id="rising-hint"></p>
      <div class="chart"><canvas id="c-rising"></canvas></div></div>
    <div class="card"><h2>Évolution hebdomadaire</h2>
      <p class="hint">Part des éléments de la semaine qui citent la compétence.</p>
      <div class="chart"><canvas id="c-weekly"></canvas></div></div>
  </div>

  <div class="card"><h2>Écart entre ton profil et les offres</h2>
    <p class="hint">Poids de chaque compétence dans la demande (pondéré par le score des offres).</p>
    <div class="legend"><span><i style="background:var(--s1)"></i>Tu l'as</span><span><i style="background:var(--s2)"></i>Il te manque</span></div>
    <div class="chart"><canvas id="c-gap"></canvas></div></div>

  <div class="card"><h2>Offres classées par score de matching</h2>
    <div class="table-wrap"><table id="jobs"><thead><tr>
      <th>Score</th><th>Offre</th><th>Publiée</th><th>Compétences manquantes</th></tr></thead><tbody></tbody></table></div></div>

  <div class="grid">
    <div class="card"><h2>Poids des critères de matching</h2>
      <p class="hint" id="model-hint"></p>
      <div class="chart short"><canvas id="c-weights"></canvas></div></div>
    <div class="card"><h2>Descente de gradient — perte</h2>
      <p class="hint" id="loss-hint"></p>
      <div class="chart short"><canvas id="c-loss"></canvas></div></div>
  </div>
</main>
<script id="data" type="application/json">__DATA__</script>
<script>
const D = JSON.parse(document.getElementById('data').textContent);
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const C = { s1: css('--s1'), s2: css('--s2'), s3: css('--s3'), neg: css('--neg'),
            text: css('--text'), text2: css('--text-2'), grid: css('--grid'), surface: css('--surface') };
Chart.defaults.color = C.text2;
Chart.defaults.font.family = getComputedStyle(document.body).fontFamily;
Chart.defaults.plugins.legend.labels.boxWidth = 10;
const axis = (extra = {}) => ({ grid: { color: C.grid }, border: { display: false }, ticks: { color: C.text2 }, ...extra });
const pct = (v) => (v * 100).toFixed(0) + ' %';
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

function hbar(id, labels, values, opts = {}) {
  const el = document.getElementById(id);
  if (!labels.length) { el.parentElement.innerHTML = '<p class="note">Pas assez de données.</p>'; return; }
  new Chart(el, { type: 'bar',
    data: { labels, datasets: [{ data: values, backgroundColor: opts.colors || C.s1,
      borderRadius: 4, borderSkipped: 'start', barPercentage: 0.8, categoryPercentage: 0.9 }] },
    options: { indexAxis: 'y', maintainAspectRatio: false, plugins: { legend: { display: false },
      tooltip: { callbacks: { label: (ctx) => opts.fmt ? opts.fmt(ctx.raw, ctx.dataIndex) : String(ctx.raw) } } },
      scales: { x: axis({ ticks: { color: C.text2, callback: opts.tick || ((v) => v) } }),
                y: axis({ grid: { display: false }, ticks: { color: C.text2, autoSkip: false } }) } } });
}

// En-tête
document.getElementById('sub').textContent =
  `Généré le ${D.generated_at.slice(0, 16).replace('T', ' ')} · fenêtre de ${D.window_days} jours · ` +
  `${D.request.remote_only ? 'full remote' : 'tous modes'}${D.request.location ? ' · ' + D.request.location : ''} · profil : ${D.profile.origin}`;
const tiles = [
  [D.counts.job || 0, 'offres analysées'], [D.counts.post || 0, 'posts'], [D.counts.person || 0, 'profils'],
  [D.gap.coverage + ' %', 'couverture de tes compétences'],
  [D.top_jobs.length ? D.top_jobs[0].score + ' %' : '—', 'meilleur score'],
];
document.getElementById('tiles').innerHTML = tiles.map(([v, l]) => `<div class="tile"><b>${esc(v)}</b><span>${esc(l)}</span></div>`).join('');

// Interprétation
document.getElementById('interp-src').textContent = D.interpreter === 'claude'
  ? `Rédigée par Claude (${D.interpreter_model}) à partir des agrégats ci-dessous.`
  : D.interpreter === 'local' ? 'Rédigée par l’interprète local (règles + poids du modèle de matching).' : 'Interprétation désactivée.';
document.getElementById('analysis').innerHTML = D.interpretation ? DOMPurify.sanitize(marked.parse(D.interpretation)) : '';

hbar('c-terms', D.top_terms.slice(0, 20).map((t) => t.term), D.top_terms.slice(0, 20).map((t) => t.docs),
     { fmt: (v) => `${v} documents` });
const jobSkills = D.skill_demand.filter((s) => s.job).slice(0, 20);
hbar('c-skills', jobSkills.map((s) => s.skill), jobSkills.map((s) => s.job), { fmt: pct, tick: pct });

// Tendances
const R = D.rising;
document.getElementById('rising-hint').textContent = R.note ||
  `Croissance (log du ratio de fréquence) des 15 derniers jours (${R.recent_docs} docs) vs les 15 précédents (${R.older_docs} docs).`;
const risingSkills = R.skills.filter((s) => s.growth > 0).slice(0, 8);
const seen = new Set(risingSkills.map((s) => s.term));
const rising = [...risingSkills, ...R.terms.filter((t) => !seen.has(t.term)).slice(0, 8)];
hbar('c-rising', rising.map((r) => r.term), rising.map((r) => r.growth),
     { fmt: (v, i) => `croissance ${v} (${rising[i].older} → ${rising[i].recent} mentions)` });

const W = D.weekly;
const weeklyColors = [C.s1, C.s2, C.s3];
const weeklyEl = document.getElementById('c-weekly');
if (Object.keys(W.series).length) {
  new Chart(weeklyEl, { type: 'line',
    data: { labels: W.weeks.map((w, i) => `sem. du ${w.slice(5)} (${W.counts[i]})`),
      datasets: Object.entries(W.series).map(([k, v], i) => ({ label: k, data: v, borderColor: weeklyColors[i],
        backgroundColor: weeklyColors[i], borderWidth: 2, pointRadius: 4, spanGaps: true, tension: 0 })) },
    options: { maintainAspectRatio: false, interaction: { mode: 'index', intersect: false },
      plugins: { legend: { position: 'top', align: 'start' },
        tooltip: { callbacks: { label: (ctx) => `${ctx.dataset.label} : ${ctx.raw == null ? '—' : pct(ctx.raw)}` } } },
      scales: { x: axis({ grid: { display: false } }), y: axis({ beginAtZero: true, ticks: { color: C.text2, callback: pct } }) } } });
} else { weeklyEl.parentElement.innerHTML = '<p class="note">Pas assez de données.</p>'; }

// Écart
const gap = D.gap.skills.slice(0, 20);
hbar('c-gap', gap.map((g) => g.skill), gap.map((g) => g.weight),
     { colors: gap.map((g) => g.owned ? C.s1 : C.s2), fmt: (v, i) => `${gap[i].owned ? 'acquise' : 'manquante'} · poids ${v}` });

// Offres
document.querySelector('#jobs tbody').innerHTML = D.top_jobs.map((j) => `<tr>
  <td class="num"><b>${j.score} %</b></td>
  <td><a href="${esc(j.url)}" target="_blank" rel="noopener">${esc(j.title || 'Sans titre')}</a><br>
      <span class="hint">${esc(j.company)} ${j.location ? '· ' + esc(j.location) : ''}</span>
      ${(j.summary || []).length ? `<ul class="summary">${j.summary.map((s) => `<li>${esc(s)}</li>`).join('')}</ul>` : ''}
      <details><summary>Détail des critères</summary>${Object.entries(j.criteria).map(([k, v]) =>
        `<span class="tag">${esc(k)} ${v}</span>`).join('')}</details></td>
  <td class="num">${esc(j.posted_at)}</td>
  <td>${j.missing_skills.map((s) => `<span class="tag">${esc(s)}</span>`).join('') || '—'}</td></tr>`).join('');

// Modèle
const M = D.model;
document.getElementById('model-hint').textContent = M.trained
  ? `Appris par descente de gradient sur ${M.n_labels} offres annotées (précision d'entraînement ${pct(M.train_accuracy)}).`
  : `Poids a priori : annote au moins ${D.min_labels} offres (pertinentes et non pertinentes) pour entraîner le modèle.`;
hbar('c-weights', M.importance.map((r) => r.feature), M.importance.map((r) => r.weight),
     { colors: M.importance.map((r) => r.weight >= 0 ? C.s1 : C.neg), fmt: (v) => `poids ${v}` });
const lossEl = document.getElementById('c-loss');
document.getElementById('loss-hint').textContent = M.loss_history.length
  ? 'Entropie croisée régularisée, mesurée toutes les 25 itérations.' : 'Aucun entraînement pour l’instant.';
if (M.loss_history.length) {
  new Chart(lossEl, { type: 'line',
    data: { labels: M.loss_history.map((_, i) => i * 25),
      datasets: [{ data: M.loss_history, borderColor: C.s1, borderWidth: 2, pointRadius: 0 }] },
    options: { maintainAspectRatio: false, interaction: { mode: 'index', intersect: false },
      plugins: { legend: { display: false }, tooltip: { callbacks: { title: (it) => `itération ${it[0].label}`, label: (ctx) => `perte ${ctx.raw}` } } },
      scales: { x: axis({ grid: { display: false }, title: { display: true, text: 'itération', color: C.text2 } }), y: axis() } } });
} else { lossEl.parentElement.innerHTML = '<p class="note">Aucun entraînement pour l’instant.</p>'; }
</script>
</body>
</html>
"""


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40] or "recherche"


def write(results: dict) -> tuple[Path, Path]:
    config.ensure_data_dirs()
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    base = config.REPORTS_DIR / f"{_slug(results['request']['keyword'])}-{stamp}"
    json_path = base.with_suffix(".json")
    html_path = base.with_suffix(".html")
    payload = json.dumps(results, ensure_ascii=False, default=str)
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    page = (_TEMPLATE
            .replace("__KEYWORD__", html.escape(results["request"]["keyword"]))
            .replace("__DATA__", payload.replace("</", "<\\/")))
    html_path.write_text(page, encoding="utf-8")
    return html_path, json_path
