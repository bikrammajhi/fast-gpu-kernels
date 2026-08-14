"""NVIDIA-Nsight-Compute-style HTML report.

`render_html(report)` takes the dict from `report.build` and renders a
self-contained dark, app-like page: sidebar (kernels + section tree),
sticky per-kernel stat strip, verdict banner, prioritized recommendations,
and ncu-style section cards (Speed Of Light bars, stacked warp-state,
achieved-vs-theoretical occupancy, launch statistics, tables with in-cell
bars, provenance tooltips on derived rows). Everything renders client-side
from an embedded JSON payload; the output is deterministic (no timestamps).
"""

from __future__ import annotations

import html
import json

from . import __version__

CSS = """
:root {
  --bg: #0f1216; --panel: #161b22; --panel2: #1c222b; --panel3: #232a35;
  --line: #273040; --line2: #313b4e;
  --text: #d9dee6; --dim: #8792a1; --faint: #5b6575;
  --accent: #4da3ff; --accent-dim: #2f6fae;
  --good: #54c27c; --warn: #e6b54c; --crit: #e06c5c; --info: #7fa8d9;
  --chip: #2a3341;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
html { scroll-behavior: smooth; }
body {
  background: var(--bg); color: var(--text);
  font: 13px/1.45 -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  display: flex; flex-direction: column; min-height: 100vh;
}
.num { font-variant-numeric: tabular-nums; }
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-thumb { background: #2b3545; border-radius: 5px; }
::-webkit-scrollbar-track { background: transparent; }

/* ---------- header ---------- */
header {
  display: flex; align-items: center; gap: 14px; padding: 10px 18px;
  background: var(--panel); border-bottom: 1px solid var(--line);
  position: sticky; top: 0; z-index: 50;
}
.brand { font-weight: 700; letter-spacing: .4px; font-size: 14px; }
.brand b { color: var(--accent); }
.brand .ver { color: var(--faint); font-weight: 500; font-size: 11px; margin-left: 4px; }
.prov { color: var(--dim); font-size: 12px; white-space: nowrap; overflow: hidden;
  text-overflow: ellipsis; flex: 1; }
.prov .src { color: var(--faint); }
.count { font-size: 11px; color: var(--dim); background: var(--chip); padding: 3px 9px;
  border-radius: 10px; white-space: nowrap; }

/* ---------- layout ---------- */
.layout { display: flex; flex: 1; }
aside {
  width: 218px; flex: 0 0 218px; background: var(--panel);
  border-right: 1px solid var(--line); padding: 12px 8px; overflow-y: auto;
  position: sticky; top: 43px; height: calc(100vh - 43px);
}
aside h5 {
  font-size: 10px; letter-spacing: 1.2px; text-transform: uppercase;
  color: var(--faint); margin: 10px 8px 6px;
}
aside h5:first-child { margin-top: 2px; }
.nav-item {
  display: flex; align-items: center; gap: 8px; padding: 5px 8px; margin: 1px 0;
  border-radius: 6px; color: var(--dim); cursor: pointer; font-size: 12.5px;
  border: 1px solid transparent; user-select: none;
}
.nav-item:hover { background: var(--panel2); color: var(--text); }
.nav-item.active { background: var(--panel3); color: var(--text);
  border-color: var(--line2); }
.nav-item .dot { width: 7px; height: 7px; border-radius: 50%; flex: 0 0 7px; }
.nav-item .k { color: var(--faint); font-size: 10.5px; margin-left: auto; }
.nav-item .sec { flex: 0 0 8px; }
.nav-item.section-item { padding-left: 14px; font-size: 12px; }

main { flex: 1; min-width: 0; padding: 18px 26px 60px; }

/* ---------- stat strip ---------- */
.stats {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(128px, 1fr));
  gap: 10px; margin-bottom: 16px;
}
.stat { background: var(--panel); border: 1px solid var(--line); border-radius: 8px;
  padding: 8px 12px; min-width: 0; overflow: hidden; }
.stat .l { font-size: 10.5px; color: var(--faint); letter-spacing: .5px;
  text-transform: uppercase; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.stat .v { font-size: 16px; font-weight: 600; overflow-wrap: anywhere; }
.stat .v small { font-size: 11px; color: var(--dim); font-weight: 500; }

/* ---------- verdict banner ---------- */
.verdict {
  display: flex; align-items: flex-start; gap: 14px; padding: 14px 18px;
  border-radius: 10px; border: 1px solid; margin-bottom: 16px;
  position: relative; overflow: hidden;
}
.verdict::before {
  content: ''; position: absolute; inset: 0 auto 0 0; width: 4px;
  background: currentColor; opacity: .55;
}
.verdict .ic {
  flex: 0 0 auto; width: 34px; height: 34px; border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  font-size: 16px; background: rgba(255,255,255,.06);
}
.verdict .t {
  display: flex; align-items: center; gap: 10px; flex-wrap: wrap;
  font-weight: 800; font-size: 15px; letter-spacing: .3px;
}
.verdict .t .kname {
  font-weight: 600; font-size: 11.5px; color: var(--dim);
  font-variant-numeric: tabular-nums;
}
.verdict .m { color: var(--dim); font-size: 12.5px; margin-top: 5px;
  overflow-wrap: anywhere; line-height: 1.55; }
.verdict .mid { flex: 1; min-width: 0; }
.verdict.sev-critical { background: linear-gradient(135deg, #2b1a17, #201412); border-color: #5c3229; color: var(--crit); }
.verdict.sev-critical .ic { color: var(--crit); }
.verdict.sev-warning { background: linear-gradient(135deg, #2b2415, #201c10); border-color: #5c4a24; color: var(--warn); }
.verdict.sev-warning .ic { color: var(--warn); }
.verdict.sev-suggestion { background: linear-gradient(135deg, #1a2431, #131a24); border-color: #2b3e57; color: var(--accent); }
.verdict.sev-suggestion .ic { color: var(--accent); }
.verdict.sev-info { background: linear-gradient(135deg, #16211f, #111817); border-color: #2a3a36; color: var(--info); }
.verdict.sev-info .ic { color: var(--info); }

/* ---------- sections ---------- */
.sec {
  background: var(--panel); border: 1px solid var(--line); border-radius: 8px;
  margin-bottom: 14px; overflow: hidden;
}
.sec-head {
  display: flex; align-items: center; gap: 10px; padding: 10px 16px;
  cursor: pointer; user-select: none;
}
.sec-head:hover { background: var(--panel2); }
.sec-head .chev { color: var(--faint); transition: transform .15s; font-size: 11px; }
.sec.collapsed .chev { transform: rotate(-90deg); }
.sec-head h3 { font-size: 13.5px; font-weight: 600; }
.sec-head .src-tag { font-size: 10px; color: var(--accent); border: 1px solid var(--accent-dim);
  padding: 1px 6px; border-radius: 9px; }
.sec-body { padding: 4px 16px 14px; }
.sec.collapsed .sec-body { display: none; }
.sec .desc { color: var(--dim); font-size: 12px; margin: 2px 0 10px; }

table { width: 100%; border-collapse: collapse; font-size: 12.5px; table-layout: auto; }
th {
  text-align: left; color: var(--faint); font-size: 10.5px; letter-spacing: .7px;
  text-transform: uppercase; padding: 6px 10px; border-bottom: 1px solid var(--line2);
  font-weight: 600;
}
td { padding: 6px 10px; border-bottom: 1px solid var(--line); vertical-align: middle; overflow-wrap: anywhere; }
td.l { overflow-wrap: anywhere; }
tr:last-child td { border-bottom: none; }
tbody tr:hover { background: var(--panel2); }
td.l { color: var(--dim); }
td.v { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
td.u { color: var(--faint); font-size: 11px; text-align: right; padding-left: 2px; white-space: nowrap; }
.notcoll { color: var(--faint); font-style: italic; }
.notcoll b { color: var(--dim); font-style: normal; }

.derived { border-bottom: 1px dotted var(--faint); cursor: help; }
.derived:hover { color: var(--text); }

/* ---------- bars ---------- */
.bar-track { background: #232b38; border-radius: 4px; height: 12px; width: 100%;
  min-width: 90px; overflow: hidden; }
.bar { height: 100%; border-radius: 4px; }
.pct { font-variant-numeric: tabular-nums; font-weight: 600; text-align: right;
  white-space: nowrap; padding-left: 10px; }
.sol-row { display: grid; grid-template-columns: 168px 1fr 64px; align-items: center;
  gap: 12px; padding: 6px 0; }
.sol-row .l { color: var(--dim); font-size: 12.5px; overflow: hidden;
  text-overflow: ellipsis; white-space: nowrap; }
.sol-hero .l { font-weight: 600; color: var(--text); }
.sol-hero .bar-track { height: 16px; }
.sol-hero .pct { font-size: 15px; }

/* warp state stacked bar */
.stack-track { display: flex; height: 26px; border-radius: 6px; overflow: hidden;
  background: #232b38; margin: 8px 0 10px; }
.stack-seg { height: 100%; }
.legend { display: flex; flex-wrap: wrap; gap: 6px 14px; margin: 6px 0 12px; }
.legend .li { display: flex; align-items: center; gap: 6px; font-size: 11.5px; color: var(--dim); }
.legend .sw { width: 10px; height: 10px; border-radius: 3px; }

/* occupancy hero */
.occ-hero { display: grid; grid-template-columns: 170px 1fr 64px; align-items: center;
  gap: 12px; padding: 8px 0; }
.occ-hero .l { font-weight: 600; }
.theo { color: var(--faint); font-size: 11.5px; }

/* ---------- rules ---------- */
.rules { margin-bottom: 16px; }
.rules h2 { font-size: 12px; letter-spacing: 1.2px; text-transform: uppercase;
  color: var(--faint); margin: 0 0 8px 2px; }
.rule {
  display: flex; gap: 10px; align-items: flex-start; padding: 9px 12px;
  margin-bottom: 7px; background: var(--panel); border: 1px solid var(--line);
  border-left: 3px solid var(--faint); border-radius: 6px;
}
.rule .dot { width: 8px; height: 8px; border-radius: 50%; margin-top: 4px; flex: 0 0 8px; }
.rule.sev-critical { border-left-color: var(--crit); }
.rule.sev-warning { border-left-color: var(--warn); }
.rule.sev-suggestion { border-left-color: var(--accent); }
.rule.sev-info { border-left-color: var(--info); }
.rule .name { font-weight: 600; font-size: 12.5px; }
.rule .msg { color: var(--dim); font-size: 12px; margin-top: 2px; overflow-wrap: anywhere; line-height: 1.5; }
.rule .focus { margin-top: 4px; font-size: 11px; color: var(--faint); }
.rule .focus b { color: var(--dim); font-weight: 600; }
.rule .focus .fhint { color: var(--faint); }
.derived-sec { display: none; }
.toggle { color: var(--dim); font-size: 12px; cursor: pointer;
  display: inline-flex; align-items: center; gap: 6px; user-select: none; }
.toggle:hover { color: var(--text); }
.nav-item.derived-item { display: none; }
.badges { display: flex; gap: 6px; margin-left: auto; flex: 0 0 auto; }
.badge { font-size: 10px; padding: 2px 8px; border-radius: 9px; font-weight: 600;
  letter-spacing: .3px; white-space: nowrap; }
.badge.sev { color: var(--text); }
.badge.sev-critical { background: #4a2722; color: #f0a99f; }
.badge.sev-warning { background: #453a1e; color: #f2d18a; }
.badge.sev-suggestion { background: #1f3a57; color: #9cc7f2; }
.badge.sev-info { background: #223a34; color: #a5d6c4; }
.badge.src { background: var(--chip); color: var(--dim); }
.badge.src.ncu { background: #1d3343; color: #7fb8e8; }
.badge.est { background: #263326; color: #8fd4a8; }

/* ---------- summary ---------- */
.sum-hero { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 10px; margin-bottom: 18px; }
.verdict-row { display: flex; gap: 12px; align-items: flex-start; background: var(--panel);
  border: 1px solid var(--line); border-radius: 8px; padding: 12px 14px; margin-bottom: 10px; }
.verdict-row .dot { width: 10px; height: 10px; border-radius: 50%; flex: 0 0 10px;
  margin-top: 4px; }
.verdict-row .kname { font-weight: 600; width: 150px; flex: 0 0 150px; min-width: 0;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis; font-size: 12px; }
.verdict-row .vname { color: var(--text); font-weight: 700; font-size: 12px;
  letter-spacing: .4px; text-transform: uppercase; }
.verdict-row .vmsg { color: var(--dim); font-size: 12px; overflow-wrap: anywhere;
  margin-top: 2px; line-height: 1.5; }
.verdict-row .badges { flex: 0 0 auto; display: flex; gap: 6px; flex-wrap: wrap; }
.verdict-row .mid { flex: 1; min-width: 0; }

/* mini bars in series table */
.mini { display: flex; align-items: center; justify-content: flex-end; gap: 8px; }
.mini .mt { width: 90px; }
.mini .mv { font-variant-numeric: tabular-nums; color: var(--dim); font-size: 11.5px; white-space: nowrap; }

@media (max-width: 900px) {
  aside { display: none; }
  .layout { flex-direction: column; }
  .verdict-row { flex-wrap: wrap; }
  .verdict-row .kname { flex: 1 1 100%; width: auto; }
}
"""

JS = """
const DATA = __DATA__;
const $ = (s, el) => (el || document).querySelector(s);
const $$ = (s, el) => Array.from((el || document).querySelectorAll(s));
const esc = s => String(s == null ? '' : s).replace(/[&<>"']/g,
  c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt = (v, d) => v == null ? '—' : Number(v).toLocaleString('en-US', {maximumFractionDigits: d == null ? 2 : d});

const SEV = {critical: 0, warning: 1, suggestion: 2, info: 3};
const SEVNAME = {critical: 'Critical', warning: 'Warning', suggestion: 'Suggestion', info: 'Info'};
const ICONS = {critical: '▲', warning: '●', suggestion: '◆', info: 'ℹ'};
const BARCOL = v => v >= 80 ? 'var(--good)' : v >= 50 ? 'var(--warn)' : 'var(--crit)';
const STALLPALETTE = ['#4da3ff','#e6b54c','#54c27c','#e06c5c','#c08ae0','#5ac8c8',
  '#e08ac0','#9aa8e0','#7fa8d9','#d9a05c','#8ad0a0','#c8c05c','#a08ae0','#5cc8d9','#e0d05c'];

let curKernel = DATA.kernels.length ? DATA.kernels[0].key : null;
let curView = 'kernel'; // 'kernel' | 'summary'

function solRow(label, pct, hero) {
  const c = BARCOL(pct);
  return `<div class="sol-row${hero ? ' sol-hero' : ''}">
    <div class="l" title="${esc(label)}">${esc(label)}</div>
    <div class="bar-track"><div class="bar" style="width:${Math.min(pct,100)}%;background:${c}"></div></div>
    <div class="pct num" style="color:${c}">${fmt(pct,1)}%</div></div>`;
}

function rowHtml(r) {
  const v = r.value == null ? '—' : esc(r.value);
  const val = r.derived ? `<span class="derived num" title="${esc(r.note || 'derived from ncu counters')}">${v}</span>` : `<span class="num">${v}</span>`;
  const bar = r.bar != null
    ? `<td style="width:26%"><div class="bar-track"><div class="bar" style="width:${Math.min(r.bar,100)}%;background:${BARCOL(r.bar)}"></div></div></td>`
    : '<td></td>';
  return `<tr><td class="l">${esc(r.label)}</td>${bar}<td class="v">${val}</td>
    <td class="u">${r.unit ? esc(r.unit) : ''}</td></tr>`;
}

function tableHtml(table) {
  const [head, ...rows] = table;
  return `<table><thead><tr>${head.map(h => `<th>${esc(h)}</th>`).join('')}</tr></thead>
    <tbody>${rows.map(r => `<tr>${r.map((c, i) =>
      i === 0 ? `<td class="l">${esc(c)}</td>` : `<td class="num">${esc(c)}</td>`).join('')}</tr>`).join('')}</tbody></table>`;
}

function secHtml(s) {
  let body = '';
  if (s.rows.length) body += `<table><tbody>${s.rows.map(rowHtml).join('')}</tbody></table>`;
  if (s.table && s.table.length) body += '<div style="margin-top:10px">' + tableHtml(s.table) + '</div>';
  return `<div class="sec" id="sec-${esc(s.sid)}" data-sid="${esc(s.sid)}">
    <div class="sec-head"><span class="chev">▼</span><h3>${esc(s.title)}</h3>
    ${s.src ? `<span class="src-tag">${esc(s.src)}</span>` : ''}</div>
    <div class="sec-body">${s.desc ? `<div class="desc">${esc(s.desc)}</div>` : ''}${body}</div></div>`;
}

function solSectionHtml(k) {
  const sec = k.sections.find(s => s.sid === 'speedoflight');
  if (!sec) return '';
  const rows = sec.rows.filter(r => r.bar != null);
  const compute = rows.find(r => /tensor/i.test(r.label)) || rows[0];
  const memory = rows.find(r => /dram/i.test(r.label));
  let out = '<div class="sec" id="sec-speedoflight"><div class="sec-head"><span class="chev">▼</span><h3>GPU Speed Of Light Throughput</h3></div><div class="sec-body">';
  if (compute) out += solRow(compute.label, compute.bar, true);
  if (memory) out += solRow(memory.label, memory.bar, true);
  out += '<div style="margin-top:8px">' + rows.filter(r => r !== compute && r !== memory).map(r => solRow(r.label, r.bar)).join('') + '</div>';
  out += `<div class="desc" style="margin-top:10px">${esc(sec.description || '')}</div></div></div>`;
  return out;
}

function warpStateHtml(k) {
  const sec = k.sections.find(s => s.sid === 'warpstate');
  if (!sec) return '';
  const rows = sec.rows.filter(r => r.bar != null);
  if (!rows.length) return secHtml(sec);
  const tot = rows.reduce((a, r) => a + r.bar, 0) || 1;
  const segs = rows.map((r, i) => `<div class="stack-seg" title="${esc(r.label)}: ${fmt(r.bar,1)}%"
    style="width:${r.bar / tot * 100}%;background:${STALLPALETTE[i % STALLPALETTE.length]}"></div>`).join('');
  const legend = rows.map((r, i) => `<span class="li"><span class="sw" style="background:${STALLPALETTE[i % STALLPALETTE.length]}"></span>${esc(r.label)} · ${fmt(r.bar,1)}%</span>`).join('');
  return `<div class="sec" id="sec-warpstate"><div class="sec-head"><span class="chev">▼</span>
    <h3>Warp State</h3></div><div class="sec-body"><div class="stack-track">${segs}</div>
    <div class="legend">${legend}</div>
    <table><tbody>${rows.map(rowHtml).join('')}</tbody></table>
    <div class="desc" style="margin-top:10px">${esc(sec.description || '')}</div></div></div>`;
}

function occupancyHtml(k) {
  const occ = k.sections.find(s => s.sid === 'occupancy');
  if (!occ) return '';
  const achieved = occ.rows.find(r => /achieved/i.test(r.label) && r.bar != null);
  const theo = occ.rows.find(r => /theoretical/i.test(r.label));
  let out = '<div class="sec" id="sec-occupancy"><div class="sec-head"><span class="chev">▼</span><h3>Occupancy</h3></div><div class="sec-body">';
  if (achieved) out += `<div class="occ-hero"><div class="l">${esc(achieved.label)}</div>
    <div class="bar-track" style="height:16px"><div class="bar" style="width:${Math.min(achieved.bar,100)}%;background:${BARCOL(achieved.bar)}"></div></div>
    <div class="pct num" style="color:${BARCOL(achieved.bar)}">${fmt(achieved.bar,1)}%</div></div>`;
  if (theo) out += `<div class="theo">${esc(theo.label)}: ${esc(theo.value)}${theo.unit || ''}</div>`;
  const rest = occ.rows.filter(r => r !== achieved && r !== theo);
  if (rest.length) out += '<table style="margin-top:8px"><tbody>' + rest.map(rowHtml).join('') + '</tbody></table>';
  out += '</div></div>';
  return out;
}

function focusEvidence(r) {
  const info = r.focus_info || {};
  const parts = Object.keys(r.focus || {}).map(n =>
    `${esc(n)}: <b>${fmt(r.focus[n], 3)}</b>${info[n] ? ` <span class="fhint">(${esc(info[n])})</span>` : ''}`);
  return parts.length ? `<div class="focus">Focus metrics: ${parts.join(' · ')}</div>` : '';
}

function rulesHtml(rules, title) {
  if (!rules.length) return '';
  const sorted = rules.slice().sort((a, b) => (SEV[a.severity] ?? 9) - (SEV[b.severity] ?? 9));
  const items = sorted.map(r => `
    <div class="rule sev-${esc(r.severity)}">
      <span class="dot" style="background:${r.severity === 'critical' ? 'var(--crit)' : r.severity === 'warning' ? 'var(--warn)' : r.severity === 'suggestion' ? 'var(--accent)' : 'var(--info)'}"></span>
      <div style="flex:1;min-width:0">
        <div class="name">${ICONS[r.severity] || ''} ${esc(r.name)}</div>
        <div class="msg">${esc(r.message)}</div>
        ${focusEvidence(r)}
      </div>
      <div class="badges">
        ${r.est != null ? `<span class="badge est">est. ${esc(r.est)}x</span>` : ''}
        <span class="badge sev sev-${esc(r.severity)}">${SEVNAME[r.severity] || r.severity}</span>
        <span class="badge src${r.source === 'ncu' ? ' ncu' : ''}">${esc(r.source)}</span>
      </div>
    </div>`).join('');
  return `<div class="rules"><h2>${esc(title)}</h2>${items}</div>`;
}

function statChip(l, v, unit) {
  return `<div class="stat"><div class="l">${esc(l)}</div>
    <div class="v num">${v == null ? '—' : esc(v)}${unit ? `<small> ${esc(unit)}</small>` : ''}</div></div>`;
}

const NVIDIA_COVER = {SpeedOfLight: 'speedoflight', SchedulerStats: 'schedulerstats',
  WarpStateStats: 'warpstate', ComputeWorkloadAnalysis: 'computeworkload',
  MemoryWorkloadAnalysis_Tables: 'memoryworkload', Occupancy: 'occupancy',
  'PM Sampling': 'pmsampling'};

function kernelPage(k) {
  const s = k.stats || {};
  const v = k.verdict;
  const strip = statChip('Duration', s.time_us != null ? fmt(s.time_us, 0) : null, 'µs')
    + statChip('Tensor TFLOPS', s.tflops != null ? fmt(s.tflops, 1) : null, 'TF/s')
    + statChip('Tensor pipe', s.pipe_pct != null ? fmt(s.pipe_pct, 1) : null, '%')
    + statChip('DRAM bandwidth', s.dram_pct != null ? fmt(s.dram_pct, 1) : null, '%')
    + statChip('Achieved occupancy', s.occupancy_pct != null ? fmt(s.occupancy_pct, 1) : null, '%')
    + statChip('Stall / issue', s.stall_cycles != null ? fmt(s.stall_cycles, 2) : null, 'cyc');
  const banner = v ? `<div class="verdict sev-${esc(v.severity)}">
      <span class="ic">${ICONS[v.severity] || '◆'}</span>
      <div class="mid"><div class="t">${esc(v.name)}
          <span class="kname">${esc(k.name)}</span></div>
        <div class="m">${esc(v.message)}</div></div>
      <span class="badge sev sev-${esc(v.severity)}" style="flex:0 0 auto">
        ${SEVNAME[v.severity] || v.severity}</span></div>` : '';
  const ncu = k.ncu_sections || [];
  const detailed = ncu.filter(x => x.detailed);
  const oneliners = ncu.filter(x => !x.detailed);
  // NVIDIA-first: when NVIDIA's own full table is present for a topic, it
  // becomes the primary section; our derived one is hidden behind a toggle.
  const detOurs = new Set(detailed.map(d => NVIDIA_COVER[d.sid]).filter(Boolean));
  const heroes = {speedoflight: solSectionHtml(k), warpstate: warpStateHtml(k),
    occupancy: occupancyHtml(k)};
  const primaryOurs = k.sections.filter(s => !detOurs.has(s.sid))
    .map(sid => heroes[sid.sid] || secHtml(sid)).join('');
  const hiddenOurs = k.sections.filter(s => detOurs.has(s.sid))
    .map(sid => heroes[sid.sid] || secHtml(sid)).join('');
  const derivedBlock = hiddenOurs
    ? `<div class="derived-sec" data-derived="1">${hiddenOurs}</div>` : '';
  const toggle = hiddenOurs ? `<div style="text-align:right;margin-bottom:10px">
    <label class="toggle"><input type="checkbox" id="derived-toggle">
    show derived (ours)</label></div>` : '';
  const ourRules = k.rules.filter(r => r.rid !== 'verdict');
  return `<div class="stats">${strip}</div>${banner}
    ${rulesHtml(k.ncu_rules, 'Recommendations (NVIDIA rule engine)')}
    ${rulesHtml(ourRules, 'Recommendations (ncu-view rules)')}
    ${toggle}
    ${detailed.map(secHtml).join('')}
    ${primaryOurs}
    ${derivedBlock}
    ${oneliners.map(secHtml).join('')}`;
}

function miniBar(v, color) {
  if (v == null) return '<div class="mini"><span class="mv">—</span></div>';
  return `<div class="mini"><div class="mt"><div class="bar-track"><div class="bar" style="width:${Math.min(v,100)}%;background:${color || BARCOL(v)}"></div></div></div><span class="mv">${fmt(v,1)}%</span></div>`;
}

function summaryPage() {
  const series = DATA.series || [];
  const best = series.filter(r => r.tflops != null).sort((a, b) => b.tflops - a.tflops)[0];
  let rows = '';
  series.forEach((r, i) => {
    const k = DATA.kernels.find(x => x.key === r.key);
    const v = k && k.verdict;
    const sevc = v ? (v.severity === 'critical' ? 'var(--crit)' : v.severity === 'warning' ? 'var(--warn)' : v.severity === 'suggestion' ? 'var(--accent)' : 'var(--info)') : 'var(--faint)';
    rows += `<tr>
      <td class="l"><b>${esc(r.name)}</b></td>
      <td class="v num">${r.time_us != null ? fmt(r.time_us, 0) : '—'}</td>
      <td class="v num">${r.tflops != null ? fmt(r.tflops, 1) : '—'}</td>
      <td>${miniBar(r.pipe_pct)}</td>
      <td>${miniBar(r.dram_pct)}</td>
      <td>${miniBar(r.occupancy_pct)}</td>
      <td class="v num">${r.stall_cycles != null ? fmt(r.stall_cycles, 2) : '—'}</td>
      <td><span class="dot" style="display:inline-block;width:8px;height:8px;border-radius:50%;background:${sevc};margin-right:6px"></span><span style="color:${sevc};font-weight:600">${v ? esc(v.name) : '—'}</span></td>
    </tr>`;
  });
  const sevColor = s => s === 'critical' ? 'var(--crit)' : s === 'warning' ? 'var(--warn)'
    : s === 'suggestion' ? 'var(--accent)' : 'var(--info)';
  const recs = DATA.kernels.map(k => k.rules.filter(r => r.rid !== 'verdict').map(r => ({...r, kname: k.name}))).flat()
    .concat(DATA.kernels.map(k => (k.ncu_rules || []).map(r => ({...r, kname: k.name}))).flat())
    .sort((a, b) => (SEV[a.severity] ?? 9) - (SEV[b.severity] ?? 9));
  const recHtml = recs.map(r => `<div class="verdict-row">
      <span class="dot" style="background:${sevColor(r.severity)}"></span>
      <span class="kname" title="${esc(r.kname)}">${esc(r.kname)}</span>
      <div class="mid">
        <div class="vname" style="color:${sevColor(r.severity)}">${esc(r.name)}</div>
        <div class="vmsg">${esc(r.message)}</div></div>
      <div class="badges">
        ${r.est != null ? `<span class="badge est">est. ${esc(r.est)}x</span>` : ''}
        <span class="badge src${r.source === 'ncu' ? ' ncu' : ''}">${esc(r.source)}</span></div></div>`).join('');
  return `<div class="sum-hero">${statChip('Kernels profiled', DATA.kernels.length)}
    ${statChip('Best kernel', best ? best.name : '—')}
    ${statChip('Best TFLOPS', best && best.tflops != null ? fmt(best.tflops, 1) : null, 'TF/s')}
    ${statChip('Best duration', best && best.time_us != null ? fmt(best.time_us, 0) : null, 'µs')}</div>
    <div class="rules"><h2>Prioritized recommendations (all kernels)</h2>${recHtml || '<div class="desc">No recommendations.</div>'}</div>
    <div class="sec"><div class="sec-head"><span class="chev">▼</span><h3>Kernel series</h3></div>
    <div class="sec-body"><table><thead><tr>
      <th>Kernel</th><th style="text-align:right">Duration µs</th><th style="text-align:right">TFLOPS</th>
      <th>Tensor pipe</th><th>DRAM</th><th>Occupancy</th><th style="text-align:right">Stall/issue</th><th>Verdict</th>
    </tr></thead><tbody>${rows}</tbody></table></div></div>`;
}

function renderMain() {
  const main = $('#main');
  if (curView === 'summary') {
    main.innerHTML = summaryPage();
    bindCollapse();
    return;
  }
  const k = DATA.kernels.find(x => x.key === curKernel);
  if (!k) return;
  main.innerHTML = kernelPage(k);
  bindCollapse();
  bindDerived();
  spy();
}

function sidebar() {
  let kn = '<h5>Kernels</h5>'
    + `<div class="nav-item" data-view="summary" data-key="__summary__">
        <span class="dot" style="background:var(--accent)"></span>Summary<span class="k">${DATA.kernels.length}</span></div>`;
  DATA.kernels.forEach(k => {
    const v = k.verdict;
    const c = v ? (v.severity === 'critical' ? 'var(--crit)' : v.severity === 'warning' ? 'var(--warn)' : 'var(--accent)') : 'var(--faint)';
    kn += `<div class="nav-item" data-view="kernel" data-key="${esc(k.key)}">
      <span class="dot" style="background:${c}"></span>${esc(k.name)}</div>`;
  });
  const k0 = DATA.kernels[0];
  const detOurs = new Set((k0 ? (k0.ncu_sections || []) : [])
    .filter(s => s.detailed).map(s => NVIDIA_COVER[s.sid]).filter(Boolean));
  const own = k0 ? k0.sections.map(s => ({
    sid: s.sid, title: s.title,
    c: ['sol','warpstate','occupancy','launchstats','memory','compute'].includes(s.sid)
      ? 'var(--accent)' : 'var(--faint)', ncu: false,
    derived: detOurs.has(s.sid),
  })) : [];
  const theirs = k0 ? (k0.ncu_sections || []).map(s => ({
    sid: s.sid, title: s.title, c: 'var(--accent)', ncu: true,
    derived: false,
  })) : [];
  const secs = [...own, ...theirs].map(s =>
    `<div class="nav-item section-item${s.derived ? ' derived-item' : ''}" data-sec="${esc(s.sid)}" data-key="__sec__">
      <span class="sec">▸</span>${esc(s.title)}
      ${s.ncu ? '<span class="k">NVIDIA</span>' : ''}</div>`).join('');
  $('#sidebar').innerHTML = kn + '<h5>Sections</h5>' + secs;
  $$('#sidebar .nav-item').forEach(el => el.addEventListener('click', () => {
    $$('#sidebar .nav-item').forEach(x => x.classList.remove('active'));
    el.classList.add('active');
    if (el.dataset.view === 'summary') { curView = 'summary'; renderMain(); }
    else if (el.dataset.key === '__sec__') {
      const t = $('#sec-' + el.dataset.sec);
      if (t) { t.scrollIntoView({behavior: 'smooth', block: 'start'}); }
    } else {
      curView = 'kernel'; curKernel = el.dataset.key; renderMain();
    }
  }));
}

function bindCollapse() {
  $$('.sec-head').forEach(h => h.addEventListener('click', () =>
    h.parentElement.classList.toggle('collapsed')));
}

function bindDerived() {
  const t = $('#derived-toggle');
  if (!t) return;
  const apply = show => {
    $$('.derived-sec').forEach(x => { x.style.display = show ? 'block' : 'none'; });
    $$('#sidebar .derived-item').forEach(x => { x.style.display = show ? '' : 'none'; });
  };
  t.addEventListener('change', () => apply(t.checked));
}

function spy() {
  const secs = $$('#main .sec').map(el => el).filter(el => el.id);
  if (!secs.length) return;
  if (typeof IntersectionObserver === 'undefined') return; // e.g. jsdom
  const io = new IntersectionObserver(es => {
    es.forEach(e => {
      if (!e.isIntersecting) return;
      const sid = e.target.dataset.sid;
      $$('#sidebar .nav-item[data-key="__sec__"]').forEach(x =>
        x.classList.toggle('active', x.dataset.sec === sid));
    });
  }, {rootMargin: '-10% 0px -80% 0px'});
  secs.forEach(s => io.observe(s));
}

document.addEventListener('DOMContentLoaded', () => {
  $('#app-title').textContent = DATA.meta.input;
  const src = DATA.meta.source;
  $('#app-src').textContent = src ? ' · ' + src : '';
  $('#app-count').textContent = DATA.kernels.length + ' kernel' + (DATA.kernels.length === 1 ? '' : 's');
  sidebar();
  renderMain();
});
"""


def render_html(report: dict) -> str:
    payload = json.dumps(report, sort_keys=True)
    js = JS.replace("__DATA__", payload)
    title = "ncu-view"
    if report.get("meta", {}).get("input"):
        title += " — " + report["meta"]["input"]
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>{CSS}</style>
<script>{js}</script>
</head>
<body>
<header>
  <div class="brand">ncu<b>view</b><span class="ver">{html.escape(__version__)}</span></div>
  <div class="prov"><span id="app-title"></span><span class="src" id="app-src"></span></div>
  <span class="count" id="app-count"></span>
</header>
<div class="layout">
  <aside id="sidebar"></aside>
  <main id="main"></main>
</div>
</body>
</html>
"""
