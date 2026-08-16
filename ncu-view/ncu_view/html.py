"""NVIDIA-Nsight-Compute-style HTML report.

`render_html(report)` takes the dict from `report.build` and renders a
self-contained dark app-like page: sidebar (kernels + section tree),
sticky per-kernel stat strip, verdict banner, prioritized recommendations,
and ncu-style section cards (Speed Of Light bars, stacked warp-state,
achieved-vs-theoretical occupancy, launch statistics, tables with in-cell
bars, provenance tooltips on derived rows). Everything renders client-side
from an embedded JSON payload; the output is deterministic (no timestamps).

Beyond the base report the page adds the chrome a profiling tool should
have: a light/dark theme, kernel search, keyboard navigation, deep links,
a series trend chart, a stall-mix donut, copy-to-clipboard, and
section-collapse state that survives reloads.
"""

from __future__ import annotations

import html
import json

from . import __version__

CSS = r"""
:root{
  --bg:#0d1015; --bg-grad:radial-gradient(1200px 600px at 70% -10%,#131a26 0%,#0d1015 55%);
  --panel:#141a23; --panel2:#1a2130; --panel3:#202a3c; --line:#1f2836; --line2:#2a3548;
  --text:#dbe1ea; --dim:#8b97a8; --faint:#5c6878;
  --accent:#4da3ff; --accent2:#8a6cff; --good:#4fd07c; --warn:#e9b64d; --crit:#ec6b5a; --info:#7fb0e6;
  --chip:#232d3d; --ring:rgba(77,163,255,.45);
  --shadow:0 1px 0 rgba(255,255,255,.03),0 8px 24px rgba(0,0,0,.35);
  --radius:10px; --mono:ui-monospace,"SF Mono",SFMono-Regular,Menlo,Consolas,monospace;
}
[data-theme=light]{
  --bg:#f5f7fa; --bg-grad:radial-gradient(1200px 600px at 70% -10%,#eef3fb 0%,#f5f7fa 55%);
  --panel:#ffffff; --panel2:#f2f5fa; --panel3:#e9eef7; --line:#e3e9f2; --line2:#d3dcea;
  --text:#1a2230; --dim:#5a6575; --faint:#8a94a4;
  --accent:#1976e0; --accent2:#6a4fd0; --good:#2f9e5b; --warn:#b5811a; --crit:#d6453a; --info:#2e6fb0;
  --chip:#e7edf6; --ring:rgba(25,118,224,.35);
  --shadow:0 1px 0 rgba(20,30,50,.04),0 8px 24px rgba(30,50,80,.10);
}
*{box-sizing:border-box;margin:0;padding:0}
html{scroll-behavior:smooth}
html,body{background:var(--bg)}
body{
  color:var(--text);
  font:13px/1.5 -apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;
  background:var(--bg-grad) fixed;
  display:flex;flex-direction:column;min-height:100vh;
  -webkit-font-smoothing:antialiased;
}
a{color:var(--accent)}
.num{font-variant-numeric:tabular-nums}
::selection{background:rgba(77,163,255,.3)}
:focus-visible{outline:2px solid var(--ring);outline-offset:2px;border-radius:4px}
::-webkit-scrollbar{width:10px;height:10px}
::-webkit-scrollbar-thumb{background:#2b3545;border-radius:5px;border:2px solid transparent;background-clip:padding-box}
::-webkit-scrollbar-track{background:transparent}

/* ---------- header ---------- */
header{
  display:flex;align-items:center;gap:12px;padding:10px 16px;
  background:color-mix(in srgb,var(--panel) 82%,transparent);
  backdrop-filter:blur(10px);-webkit-backdrop-filter:blur(10px);
  border-bottom:1px solid var(--line);position:sticky;top:0;z-index:50;
}
.brand{font-weight:800;letter-spacing:.4px;font-size:14px;display:flex;align-items:baseline;gap:6px;white-space:nowrap}
.brand b{background:linear-gradient(90deg,var(--accent),var(--accent2));-webkit-background-clip:text;background-clip:text;color:transparent}
.brand .ver{color:var(--faint);font-weight:500;font-size:10.5px}
.prov{color:var(--dim);font-size:12px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;flex:1;min-width:0}
.prov .src{color:var(--faint)}
.count{font-size:11px;color:var(--dim);background:var(--chip);padding:3px 10px;border-radius:10px;white-space:nowrap}
.hdr-actions{display:flex;align-items:center;gap:6px;flex:0 0 auto}
.search{position:relative;display:flex;align-items:center}
.search input{
  width:190px;background:var(--chip);border:1px solid var(--line);color:var(--text);
  border-radius:8px;padding:6px 10px 6px 28px;font-size:12.5px;transition:width .2s,border-color .15s;
}
.search input:focus{width:240px;border-color:var(--accent);outline:none}
.search .kbd{position:absolute;right:8px;color:var(--faint);font-size:10px;font-family:var(--mono);pointer-events:none}
.search svg{position:absolute;left:8px;top:50%;transform:translateY(-50%);color:var(--faint);pointer-events:none}
.icon-btn{
  display:inline-flex;align-items:center;justify-content:center;gap:6px;width:30px;height:30px;
  background:var(--chip);border:1px solid var(--line);color:var(--dim);border-radius:8px;
  cursor:pointer;font-size:13px;transition:color .15s,border-color .15s,transform .1s;
}
.icon-btn:hover{color:var(--text);border-color:var(--line2)}
.icon-btn:active{transform:scale(.94)}
.icon-btn svg{width:15px;height:15px;stroke:currentColor;fill:none;stroke-width:2;stroke-linecap:round;stroke-linejoin:round}

/* ---------- layout ---------- */
.layout{display:flex;flex:1;align-items:flex-start}
aside{
  width:220px;flex:0 0 220px;background:color-mix(in srgb,var(--panel) 86%,transparent);
  border-right:1px solid var(--line);padding:12px 8px 40px;overflow-y:auto;
  position:sticky;top:51px;height:calc(100vh - 51px);
}
aside h5{
  font-size:10px;letter-spacing:1.4px;text-transform:uppercase;color:var(--faint);
  margin:12px 8px 6px;display:flex;align-items:center;gap:6px;
}
aside h5:first-child{margin-top:2px}
aside h5 .cnt{color:var(--faint);font-weight:500;letter-spacing:0}
.nav-item{
  display:flex;align-items:center;gap:8px;padding:5px 8px;margin:1px 0;
  border-radius:7px;color:var(--dim);cursor:pointer;font-size:12.5px;
  border:1px solid transparent;user-select:none;transition:background .12s,color .12s;
}
.nav-item:hover{background:var(--panel2);color:var(--text)}
.nav-item.active{background:var(--panel3);color:var(--text);border-color:var(--line2)}
.nav-item .dot{width:7px;height:7px;border-radius:50%;flex:0 0 7px}
.nav-item .k{color:var(--faint);font-size:10.5px;margin-left:auto;font-variant-numeric:tabular-nums}
.nav-item .sec{flex:0 0 10px;color:var(--faint);font-size:9px}
.nav-item.section-item{padding-left:14px;font-size:12px}
.nav-empty{color:var(--faint);font-size:12px;padding:8px;font-style:italic}
main{flex:1;min-width:0;padding:20px 28px 80px}

/* ---------- stat strip ---------- */
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(132px,1fr));gap:10px;margin-bottom:16px}
.stat{
  background:var(--panel);border:1px solid var(--line);border-radius:var(--radius);padding:10px 12px;
  box-shadow:var(--shadow);position:relative;overflow:hidden;transition:transform .12s,border-color .15s;
}
.stat:hover{transform:translateY(-1px);border-color:var(--line2)}
.stat::before{content:'';position:absolute;inset:0 0 auto 0;height:2px;background:linear-gradient(90deg,var(--stat,var(--accent)),transparent);opacity:.7}
.stat .l{font-size:10px;color:var(--faint);letter-spacing:.7px;text-transform:uppercase;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.stat .v{font-size:17px;font-weight:700;margin-top:2px;overflow-wrap:anywhere;cursor:pointer}
.stat .v small{font-size:11px;color:var(--dim);font-weight:500;margin-left:2px}
.stat .sub{font-size:10.5px;color:var(--faint);margin-top:1px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.stat .trend{position:absolute;top:9px;right:9px;font-size:10px;font-weight:700;opacity:.9}

/* ---------- verdict banner ---------- */
.verdict{
  display:flex;align-items:flex-start;gap:14px;padding:16px 18px;border-radius:var(--radius);
  border:1px solid;margin-bottom:16px;position:relative;overflow:hidden;box-shadow:var(--shadow);
}
.verdict::before{content:'';position:absolute;inset:0 auto 0 0;width:5px;background:currentColor;opacity:.65}
.verdict .ic{
  flex:0 0 auto;width:38px;height:38px;border-radius:11px;display:flex;align-items:center;justify-content:center;
  font-size:17px;background:rgba(255,255,255,.07);border:1px solid rgba(255,255,255,.08);
}
.verdict .t{display:flex;align-items:center;gap:10px;flex-wrap:wrap;font-weight:800;font-size:15px;letter-spacing:.3px}
.verdict .t .kname{font-weight:600;font-size:11.5px;color:var(--dim);font-variant-numeric:tabular-nums}
.verdict .m{color:var(--dim);font-size:12.5px;margin-top:6px;overflow-wrap:anywhere;line-height:1.6}
.verdict .mid{flex:1;min-width:0}
.verdict .focus{display:flex;gap:6px;flex-wrap:wrap;margin-top:9px}
.verdict .fchip{font-size:11px;color:var(--dim);background:rgba(255,255,255,.05);border:1px solid var(--line2);padding:3px 8px;border-radius:7px;font-variant-numeric:tabular-nums}
.verdict .fchip b{color:var(--text)}
.verdict.sev-critical{background:linear-gradient(135deg,color-mix(in srgb,var(--crit) 14%,var(--panel)),var(--panel));border-color:color-mix(in srgb,var(--crit) 45%,var(--line))}
.verdict.sev-warning{background:linear-gradient(135deg,color-mix(in srgb,var(--warn) 14%,var(--panel)),var(--panel));border-color:color-mix(in srgb,var(--warn) 45%,var(--line))}
.verdict.sev-suggestion{background:linear-gradient(135deg,color-mix(in srgb,var(--accent) 12%,var(--panel)),var(--panel));border-color:color-mix(in srgb,var(--accent) 40%,var(--line))}
.verdict.sev-info{background:linear-gradient(135deg,color-mix(in srgb,var(--info) 12%,var(--panel)),var(--panel));border-color:color-mix(in srgb,var(--info) 40%,var(--line))}
.verdict .ic{color:inherit}

/* ---------- sections ---------- */
.sec{
  background:var(--panel);border:1px solid var(--line);border-radius:var(--radius);
  margin-bottom:14px;overflow:hidden;box-shadow:var(--shadow);
}
.sec-head{display:flex;align-items:center;gap:10px;padding:11px 16px;cursor:pointer;user-select:none;transition:background .12s}
.sec-head:hover{background:var(--panel2)}
.sec-head .chev{color:var(--faint);transition:transform .18s;font-size:10px;width:10px}
.sec.collapsed .chev{transform:rotate(-90deg)}
.sec-head .sic{width:24px;height:24px;border-radius:7px;display:flex;align-items:center;justify-content:center;background:var(--chip);color:var(--accent);font-size:12px;flex:0 0 24px}
.sec-head h3{font-size:13.5px;font-weight:700}
.sec-head .meta{display:flex;align-items:center;gap:6px;margin-left:auto;flex:0 0 auto}
.sec-head .src-tag{font-size:9.5px;color:var(--accent);border:1px solid color-mix(in srgb,var(--accent) 45%,transparent);padding:1px 7px;border-radius:9px;letter-spacing:.4px;text-transform:uppercase}
.sec-head .rows{font-size:10.5px;color:var(--faint);font-variant-numeric:tabular-nums}
.sec-body{padding:6px 16px 14px}
.sec.collapsed .sec-body{display:none}
.sec .desc{color:var(--dim);font-size:12px;margin:6px 0 10px;line-height:1.55}

table{width:100%;border-collapse:collapse;font-size:12.5px}
th{text-align:left;color:var(--faint);font-size:10px;letter-spacing:.8px;text-transform:uppercase;padding:6px 10px;border-bottom:1px solid var(--line2);font-weight:700}
td{padding:6px 10px;border-bottom:1px solid var(--line);vertical-align:middle;overflow-wrap:anywhere}
tr:last-child td{border-bottom:none}
tbody tr{transition:background .1s}
tbody tr:hover{background:var(--panel2)}
td.l{color:var(--dim);overflow-wrap:anywhere}
td.v{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap;font-family:var(--mono);font-size:12px}
td.u{color:var(--faint);font-size:11px;text-align:right;padding-left:2px;white-space:nowrap}
.notcoll{color:var(--faint);font-style:italic}
.notcoll b{color:var(--dim);font-style:normal}
.derived{border-bottom:1px dotted var(--faint);cursor:help}
.derived:hover{color:var(--text)}

/* ---------- bars ---------- */
.bar-track{background:color-mix(in srgb,var(--line) 70%,transparent);border-radius:4px;height:12px;width:100%;min-width:90px;overflow:hidden;position:relative}
.bar{height:100%;border-radius:4px;transition:width .5s cubic-bezier(.2,.8,.2,1)}
.pct{font-variant-numeric:tabular-nums;font-weight:700;text-align:right;white-space:nowrap;padding-left:10px;font-family:var(--mono)}
.sol-row{display:grid;grid-template-columns:168px 1fr 64px;align-items:center;gap:12px;padding:6px 0}
.sol-row .l{color:var(--dim);font-size:12.5px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.sol-hero .l{font-weight:700;color:var(--text)}
.sol-hero .bar-track{height:18px}
.sol-hero .pct{font-size:15px}

/* warp state stacked bar + donut */
.wrap{display:flex;align-items:center;gap:20px;flex-wrap:wrap;margin:10px 0}
.warp-main{flex:1;min-width:240px}
.stack-track{display:flex;height:28px;border-radius:7px;overflow:hidden;background:var(--line);box-shadow:inset 0 1px 3px rgba(0,0,0,.3)}
.stack-seg{height:100%;transition:opacity .2s}
.stack-seg:hover{opacity:.85}
.legend{display:flex;flex-wrap:wrap;gap:6px 14px;margin:8px 0 4px}
.legend .li{display:flex;align-items:center;gap:6px;font-size:11.5px;color:var(--dim)}
.legend .sw{width:10px;height:10px;border-radius:3px}
.donut{flex:0 0 auto;text-align:center}
.donut svg{display:block}

/* occupancy hero */
.occ-hero{display:grid;grid-template-columns:170px 1fr 64px;align-items:center;gap:12px;padding:8px 0}
.occ-hero .l{font-weight:700}
.theo{color:var(--faint);font-size:11.5px}

/* ---------- rules ---------- */
.rules{margin-bottom:16px}
.rules h2{font-size:11px;letter-spacing:1.3px;text-transform:uppercase;color:var(--faint);margin:0 0 9px 2px;display:flex;align-items:center;gap:8px}
.rules h2 .badge{font-size:9px}
.rule{
  display:flex;gap:11px;align-items:flex-start;padding:11px 13px;margin-bottom:8px;
  background:var(--panel);border:1px solid var(--line);border-left:3px solid var(--faint);
  border-radius:8px;box-shadow:var(--shadow);transition:transform .12s,border-color .15s;
}
.rule:hover{transform:translateY(-1px)}
.rule .dot{width:9px;height:9px;border-radius:50%;margin-top:4px;flex:0 0 9px}
.rule.sev-critical{border-left-color:var(--crit)}
.rule.sev-warning{border-left-color:var(--warn)}
.rule.sev-suggestion{border-left-color:var(--accent)}
.rule.sev-info{border-left-color:var(--info)}
.rule .name{font-weight:700;font-size:12.5px}
.rule .msg{color:var(--dim);font-size:12px;margin-top:2px;overflow-wrap:anywhere;line-height:1.55}
.rule .focus{margin-top:5px;font-size:11px;color:var(--faint);display:flex;flex-wrap:wrap;gap:4px 12px}
.rule .focus b{color:var(--dim);font-weight:600}
.rule .focus .fhint{color:var(--faint)}
.toggle{color:var(--dim);font-size:12px;cursor:pointer;display:inline-flex;align-items:center;gap:7px;user-select:none}
.toggle:hover{color:var(--text)}
.toggle input{accent-color:var(--accent);cursor:pointer}
.badges{display:flex;gap:6px;margin-left:auto;flex:0 0 auto;flex-wrap:wrap}
.badge{font-size:9.5px;padding:2px 8px;border-radius:9px;font-weight:700;letter-spacing:.3px;white-space:nowrap;text-transform:uppercase}
.badge.sev{color:var(--text)}
.badge.sev-critical{background:color-mix(in srgb,var(--crit) 22%,transparent);color:color-mix(in srgb,var(--crit) 70%,white)}
.badge.sev-warning{background:color-mix(in srgb,var(--warn) 22%,transparent);color:color-mix(in srgb,var(--warn) 60%,white)}
.badge.sev-suggestion{background:color-mix(in srgb,var(--accent) 22%,transparent);color:color-mix(in srgb,var(--accent) 70%,white)}
.badge.sev-info{background:color-mix(in srgb,var(--info) 20%,transparent);color:color-mix(in srgb,var(--info) 60%,white)}
.badge.src{background:var(--chip);color:var(--dim)}
.badge.src.ncu{background:color-mix(in srgb,var(--accent) 18%,transparent);color:var(--accent)}
.badge.est{background:color-mix(in srgb,var(--good) 18%,transparent);color:var(--good)}

/* ---------- summary ---------- */
.sum-hero{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-bottom:18px}
.verdict-row{
  display:flex;gap:12px;align-items:flex-start;background:var(--panel);border:1px solid var(--line);
  border-radius:var(--radius);padding:12px 14px;margin-bottom:10px;box-shadow:var(--shadow);
  transition:transform .12s;
}
.verdict-row:hover{transform:translateY(-1px)}
.verdict-row .dot{width:10px;height:10px;border-radius:50%;flex:0 0 10px;margin-top:4px}
.verdict-row .kname{font-weight:600;width:150px;flex:0 0 150px;min-width:0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;font-size:12px}
.verdict-row .vname{color:var(--text);font-weight:800;font-size:11.5px;letter-spacing:.4px;text-transform:uppercase}
.verdict-row .vmsg{color:var(--dim);font-size:12px;overflow-wrap:anywhere;margin-top:2px;line-height:1.5}
.verdict-row .badges{flex:0 0 auto;display:flex;gap:6px;flex-wrap:wrap}
.verdict-row .mid{flex:1;min-width:0}
.chart-card{background:var(--panel);border:1px solid var(--line);border-radius:var(--radius);padding:14px 16px;margin-bottom:16px;box-shadow:var(--shadow)}
.chart-card h3{font-size:12px;color:var(--faint);letter-spacing:1px;text-transform:uppercase;margin-bottom:10px}
.chart-card svg{width:100%;height:auto;display:block}
.mini{display:flex;align-items:center;justify-content:flex-end;gap:8px}
.mini .mt{width:90px}
.mini .mv{font-variant-numeric:tabular-nums;color:var(--dim);font-size:11.5px;white-space:nowrap;font-family:var(--mono)}
.axis{color:var(--faint);font-size:10px;font-family:var(--mono)}
.gridline{stroke:var(--line);stroke-width:1}
.spark{fill:none;stroke:var(--accent);stroke-width:2;stroke-linecap:round;stroke-linejoin:round}
.spark-dot{fill:var(--accent)}
.empty{text-align:center;color:var(--dim);padding:80px 20px}
.empty .big{font-size:40px;margin-bottom:12px}
.empty h3{font-size:15px;margin-bottom:6px;color:var(--text)}
.empty p{font-size:12.5px;color:var(--faint)}

@media (max-width:900px){
  aside{display:none}
  .layout{flex-direction:column}
  .verdict-row{flex-wrap:wrap}
  .verdict-row .kname{flex:1 1 100%;width:auto}
  .search input{width:120px}
  .search input:focus{width:150px}
  main{padding:16px}
}
@media print{
  header,aside{display:none}
  body{background:var(--bg)}
  main{padding:0}
  .sec{box-shadow:none;break-inside:avoid}
  .verdict,.stat,.rule{box-shadow:none}
}
@media (prefers-reduced-motion:reduce){
  *{transition:none!important;animation:none!important}
  html{scroll-behavior:auto}
}
"""

ICONS_JS = r"""
const DATA = __DATA__;
const $=(s,el)=>(el||document).querySelector(s);
const $$=(s,el)=>Array.from((el||document).querySelectorAll(s));
const esc=s=>String(s==null?'':s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt=(v,d)=>v==null?'—':Number(v).toLocaleString('en-US',{maximumFractionDigits:d==null?2:d});
const SEV={critical:0,warning:1,suggestion:2,info:3};
const SEVNAME={critical:'Critical',warning:'Warning',suggestion:'Suggestion',info:'Info'};
const ICONS={critical:'▲',warning:'●',suggestion:'◆',info:'ℹ'};
const BARCOL=v=>v>=80?'var(--good)':v>=50?'var(--warn)':'var(--crit)';
const STALLPALETTE=['#4da3ff','#e9b64d','#4fd07c','#ec6b5a','#c08ae0','#5ac8c8','#e08ac0','#9aa8e0','#7fb0e6','#d9a05c','#8ad0a0','#c8c05c','#a08ae0','#5cc8d9','#e0d05c'];
const SEC_ICONS={speedoflight:'◎',warpstate:'≋',occupancy:'▤',launchstats:'▲',scheduler:'⇄',computeworkload:'⬛',memoryworkload:'◫',instruction:'𝕴',pmsampling:'≋',nvlink:'⧉',schedulerstats:'⇄',warpstate:'≋',occupancy:'▤',launchstats:'▲',sourcecounters:'℗',memory:'◫'};
const SECT_SID_ICON={'speedoflight':'◎','warpstate':'≋','occupancy':'▤','launchstats':'▲','schedulerstats':'⇄','computeworkload':'⬛','memoryworkload':'◫','instruction':'𝕴','pmsampling':'⊹','nvlink':'⧉','sourcecounters':'℗','memory':'◫'};

let curKernel=DATA.kernels.length?DATA.kernels.reduce((a,b)=>((b.stats.time_us??0)>(a.stats.time_us??0)?b:a)).key:null;
let curView='kernel';
let searchQ='';

// storage with a safe in-memory fallback (tests run without localStorage)
const store=(()=>{let m={};try{const s=typeof localStorage!=='undefined'?localStorage:null;
  return{get(k){return s?s.getItem(k):(k in m?m[k]:null)},set(k,v){if(s)s.setItem(k,v);else m[k]=v}};}catch(e){return{get:()=>null,set(){}}}})();

function copyText(t){
  try{ if(navigator.clipboard&&navigator.clipboard.writeText){navigator.clipboard.writeText(t);return true;} }catch(e){}
  try{ const ta=document.createElement('textarea');ta.value=t;document.body.appendChild(ta);ta.select();document.execCommand('copy');ta.remove();return true;}catch(e){}
  return false;
}
function setTheme(theme){
  const de=document.documentElement;
  if(de&&de.setAttribute)de.setAttribute('data-theme',theme);
  try{store.set('ncu-view-theme',theme);}catch(e){}
}
function themeInit(){
  let t=null;try{t=store.get('ncu-view-theme');}catch(e){}
  if(!t)t=(typeof matchMedia!=='undefined'&&matchMedia('(prefers-color-scheme: light)').matches)?'light':'dark';
  setTheme(t);
}

function solRow(label,pct,hero){
  const c=BARCOL(pct);
  return `<div class="sol-row${hero?' sol-hero':''}">
    <div class="l" title="${esc(label)}">${esc(label)}</div>
    <div class="bar-track"><div class="bar" style="width:${Math.min(pct,100)}%;background:${c}"></div></div>
    <div class="pct num" style="color:${c}">${fmt(pct,1)}%</div></div>`;
}
function rowHtml(r){
  const v=r.value==null?'—':esc(r.value);
  const val=r.derived?`<span class="derived num" title="${esc(r.note||'derived from ncu counters')}">${v}</span>`:`<span class="num">${v}</span>`;
  const bar=r.bar!=null
    ?`<td style="width:26%"><div class="bar-track"><div class="bar" style="width:${Math.min(r.bar,100)}%;background:${BARCOL(r.bar)}"></div></div></td>`
    :'<td></td>';
  return `<tr><td class="l">${esc(r.label)}</td>${bar}<td class="v">${val}</td><td class="u">${r.unit?esc(r.unit):''}</td></tr>`;
}
function tableHtml(table){
  const[head,...rows]=table;
  return `<table><thead><tr>${head.map(h=>`<th>${esc(h)}</th>`).join('')}</tr></thead>
    <tbody>${rows.map(r=>`<tr>${r.map((c,i)=>i===0?`<td class="l">${esc(c)}</td>`:`<td class="num">${esc(c)}</td>`).join('')}</tr>`).join('')}</tbody></table>`;
}
function secHtml(s){
  let body='';
  if(s.rows.length)body+=`<table><tbody>${s.rows.map(rowHtml).join('')}</tbody></table>`;
  if(s.table&&s.table.length)body+='<div style="margin-top:10px">'+tableHtml(s.table)+'</div>';
  const nrows=s.rows.length+(s.table&&s.table.length?s.table.length-1:0);
  return `<div class="sec" id="sec-${esc(s.sid)}" data-sid="${esc(s.sid)}">
    <div class="sec-head"><span class="chev">▼</span><span class="sic">${SECT_SID_ICON[s.sid]||'▦'}</span><h3>${esc(s.title)}</h3>
    <span class="meta">${s.src?`<span class="src-tag">${esc(s.src)}</span>`:''}${nrows?`<span class="rows">${nrows}</span>`:''}</span></div>
    <div class="sec-body">${s.desc?`<div class="desc">${esc(s.desc)}</div>`:''}${body}</div></div>`;
}
function focusEvidence(r){
  const info=r.focus_info||{};
  const parts=Object.keys(r.focus||{}).map(n=>`${esc(n)}: <b>${fmt(r.focus[n],3)}</b>${info[n]?` <span class="fhint">(${esc(info[n])})</span>`:''}`);
  return parts.length?`<div class="focus">${parts.join(' · ')}</div>`:'';
}
function rulesHtml(rules,title){
  if(!rules.length)return '';
  const sorted=rules.slice().sort((a,b)=>(SEV[a.severity]??9)-(SEV[b.severity]??9));
  const items=sorted.map(r=>`
    <div class="rule sev-${esc(r.severity)}">
      <span class="dot" style="background:${r.severity==='critical'?'var(--crit)':r.severity==='warning'?'var(--warn)':r.severity==='suggestion'?'var(--accent)':'var(--info)'}"></span>
      <div style="flex:1;min-width:0">
        <div class="name">${ICONS[r.severity]||''} ${esc(r.name)}</div>
        <div class="msg">${esc(r.message)}</div>
        ${focusEvidence(r)}
      </div>
      <div class="badges">
        ${r.est!=null?`<span class="badge est">est. ${esc(r.est)}x</span>`:''}
        <span class="badge sev sev-${esc(r.severity)}">${SEVNAME[r.severity]||r.severity}</span>
        <span class="badge src${r.source==='ncu'?' ncu':''}">${esc(r.source)}</span>
      </div>
    </div>`).join('');
  return `<div class="rules"><h2>${esc(title)}</h2>${items}</div>`;
}
function statChip(l,v,unit,sub,trend,note){
  const c=trend==='up'?'var(--crit)':trend==='down'?'var(--good)':'var(--faint)';
  const subHtml=sub!=null?`<div class="sub">${fmt(sub,1)}% of peak</div>`
    :(v==null&&note)?`<div class="sub" title="${esc(note)}">n/a — reason on hover</div>`:'';
  return `<div class="stat" style="--stat:${BARCOL(sub!=null?sub:0)}"><div class="l">${esc(l)}</div>
    <div class="v num" title="${note?esc(note):'click to copy'}">${v==null?'—':esc(v)}${unit?`<small> ${esc(unit)}</small>`:''}</div>
    ${subHtml}
    ${trend?`<span class="trend" style="color:${c}">${trend==='up'?'▲':trend==='down'?'▼':'—'}</span>`:''}</div>`;
}

function trendFor(idx,metric,higherIsGood){
  const cur=DATA.series[idx],prev=DATA.series[idx-1];
  if(!prev||cur[metric]==null||prev[metric]==null||prev[metric]===0)return null;
  const delta=(cur[metric]-prev[metric])/prev[metric];
  if(Math.abs(delta)<0.02)return null;
  const up=delta>0;
  const good=up===!!higherIsGood;
  return good?null:(up?'up':'down');
}
function kernelPage(k){
  const s=k.stats||{};const v=k.verdict;const idx=DATA.series.findIndex(x=>x.key===k.key);
  const strip=statChip('Duration',s.time_us!=null?fmt(s.time_us,0):null,'µs')
    +statChip('SM clock',s.clock_ghz!=null?fmt(s.clock_ghz,3):null,'GHz')
    +statChip('Tensor pipe',s.pipe_pct!=null?fmt(s.pipe_pct,1):null,'%',s.pipe_pct)
    +statChip('DRAM Throughput',s.dram_pct!=null?fmt(s.dram_pct,1):null,'%',s.dram_pct)
    +statChip('Achieved occupancy',s.occupancy_pct!=null?fmt(s.occupancy_pct,1):null,'%',s.occupancy_pct)
    +statChip('Stall / issue',s.stall_cycles!=null?fmt(s.stall_cycles,2):null,'cyc',null,trendFor(idx,'stall_cycles',false));
  const banner=v?`<div class="verdict sev-${esc(v.severity)}">
      <span class="ic">${ICONS[v.severity]||'◆'}</span>
      <div class="mid"><div class="t">${esc(v.name)}<span class="kname">${esc(k.name)}</span></div>
        <div class="m">${esc(v.message)}</div>
        ${focusEvidence(v)}</div>
      <span class="badge sev sev-${esc(v.severity)}" style="flex:0 0 auto">${SEVNAME[v.severity]||v.severity}</span></div>`:'';
  const secs=k.sections||[];
  const sections=secs.length?secs.map(secHtml).join('')
    :'<div class="desc">No NVIDIA sections for this input — export them with '
     +'<code>ncu --import &lt;rep&gt; --section &lt;X&gt; --csv</code> (see the README).</div>';
  return `<div class="stats">${strip}</div>${banner}
    ${rulesHtml(k.rules,'Recommendations (NVIDIA rule engine)')}
    ${sections}`;
}
function miniBar(v,color){
  if(v==null)return '<div class="mini"><span class="mv">—</span></div>';
  return `<div class="mini"><div class="mt"><div class="bar-track"><div class="bar" style="width:${Math.min(v,100)}%;background:${color||BARCOL(v)}"></div></div></div><span class="mv">${fmt(v,1)}%</span></div>`;
}
function summaryPage(){
  const series=DATA.series||[];
  const best=series.slice().sort((a,b)=>(a.time_us??Infinity)-(b.time_us??Infinity))[0];
  let rows='';
  series.forEach(r=>{
    const k=DATA.kernels.find(x=>x.key===r.key);
    const v=k&&k.verdict;
    const sevc=v?(v.severity==='critical'?'var(--crit)':v.severity==='warning'?'var(--warn)':v.severity==='suggestion'?'var(--accent)':'var(--info)'):'var(--faint)';
    rows+=`<tr>
      <td class="l"><b>${esc(r.name)}</b></td>
      <td class="v num">${r.time_us!=null?fmt(r.time_us,0):'—'}</td>
      <td>${miniBar(r.pipe_pct)}</td>
      <td>${miniBar(r.dram_pct)}</td>
      <td>${miniBar(r.occupancy_pct)}</td>
      <td class="v num">${r.stall_cycles!=null?fmt(r.stall_cycles,2):'—'}</td>
      <td><span class="dot" style="display:inline-block;width:8px;height:8px;border-radius:50%;background:${sevc};margin-right:6px"></span><span style="color:${sevc};font-weight:600">${v?esc(v.name):'—'}</span></td>
    </tr>`;
  });
  const sevColor=s=>s==='critical'?'var(--crit)':s==='warning'?'var(--warn)':s==='suggestion'?'var(--accent)':'var(--info)';
  const recs=DATA.kernels.map(k=>k.rules.map(r=>({...r,kname:k.name}))).flat()
    .sort((a,b)=>(SEV[a.severity]??9)-(SEV[b.severity]??9));
  const recHtml=recs.map(r=>`<div class="verdict-row">
      <span class="dot" style="background:${sevColor(r.severity)}"></span>
      <span class="kname" title="${esc(r.kname)}">${esc(r.kname)}</span>
      <div class="mid"><div class="vname" style="color:${sevColor(r.severity)}">${esc(r.name)}</div>
        <div class="vmsg">${esc(r.message)}</div></div>
      <div class="badges">
        ${r.est!=null?`<span class="badge est">est. ${esc(r.est)}x</span>`:''}
        <span class="badge src${r.source==='ncu'?' ncu':''}">${esc(r.source)}</span></div></div>`).join('');
  return `<div class="sum-hero">${statChip('Kernels profiled',DATA.kernels.length)}
    ${statChip('Best kernel',best?best.name:'—')}
    ${statChip('Best duration',best&&best.time_us!=null?fmt(best.time_us,0):null,'µs')}</div>
    <div class="rules"><h2>Prioritized recommendations (all kernels)</h2>${recHtml||'<div class="desc">No recommendations.</div>'}</div>
    <div class="sec"><div class="sec-head"><span class="chev">▼</span><span class="sic">▤</span><h3>Kernel series</h3></div>
    <div class="sec-body"><table><thead><tr>
      <th>Kernel</th><th style="text-align:right">Duration µs</th>
      <th>Tensor pipe</th><th>DRAM</th><th>Occupancy</th><th style="text-align:right">Stall/issue</th><th>Verdict</th>
    </tr></thead><tbody>${rows}</tbody></table></div></div>`;
}
function renderMain(){
  const main=$('#main');
  if(!main)return;
  if(curView==='summary'){main.innerHTML=summaryPage();bindCollapse();bindCopy();markActive();return;}
  const k=DATA.kernels.find(x=>x.key===curKernel);
  if(!k)return;
  main.innerHTML=kernelPage(k);
  bindCollapse();bindCopy();spy();markActive();
}
function markActive(){
  $$('#sidebar .nav-item').forEach(x=>x.classList.remove('active'));
  if(curView==='summary'){const el=$('#sidebar .nav-item[data-view="summary"]');if(el)el.classList.add('active');return;}
  const el=$('#sidebar .nav-item[data-view="kernel"][data-key="'+curKernel+'"]');
  if(el)el.classList.add('active');
}
function applyFilter(){
  const k0=DATA.kernels[0];
  const list=DATA.kernels.filter(k=>!searchQ||k.name.toLowerCase().includes(searchQ.toLowerCase()));
  const kn=list.map(k=>{const v=k.verdict;
    const c=v?(v.severity==='critical'?'var(--crit)':v.severity==='warning'?'var(--warn)':'var(--accent)'):'var(--faint)';
    return `<div class="nav-item" data-view="kernel" data-key="${esc(k.key)}"><span class="dot" style="background:${c}"></span>${esc(k.name)}</div>`;}).join('');
  const secs=(k0?(k0.sections||[]):[]).map(s=>`<div class="nav-item section-item" data-sec="${esc(s.sid)}" data-key="__sec__"><span class="sec">▸</span>${esc(s.title)}<span class="k">NVIDIA</span></div>`).join('');
  const sb=$('#sidebar');
  if(!sb)return;
  sb.innerHTML='<h5>Kernels<span class="cnt">'+DATA.kernels.length+'</span></h5>'
    +`<div class="nav-item" data-view="summary" data-key="__summary__"><span class="dot" style="background:var(--accent)"></span>Summary<span class="k">${DATA.kernels.length}</span></div>`
    +kn+(list.length? '' :'<div class="nav-empty">no kernels match "'+esc(searchQ)+'"</div>')
    +'<h5>Sections</h5>'+secs;
  $$('#sidebar .nav-item').forEach(el=>el.addEventListener('click',()=>{
    $$('#sidebar .nav-item').forEach(x=>x.classList.remove('active'));
    el.classList.add('active');
    if(el.dataset.view==='summary'){curView='summary';pushHash('#s');renderMain();}
    else if(el.dataset.key==='__sec__'){const t=$('#sec-'+el.dataset.sec);if(t)t.scrollIntoView({behavior:'smooth',block:'start'});}
    else{curView='kernel';curKernel=el.dataset.key;pushHash('#k-'+el.dataset.key);renderMain();}
  }));
  markActive();
}
function sidebar(){applyFilter();}
function bindCollapse(){
  $$('.sec-head').forEach(h=>h.addEventListener('click',()=>{
    const sec=h.parentElement;
    if(!sec)return;
    const sid=sec.dataset.sid;
    const on=sec.classList.toggle('collapsed');
    if(sid){try{const saved=JSON.parse(store.get('ncu-view-collapsed')||'[]');const set=new Set(saved);
      on?set.add(sid):set.delete(sid);store.set('ncu-view-collapsed',JSON.stringify([...set]));}catch(e){}}
  }));
}
function bindCopy(){
  $$('.stat .v').forEach(el=>el.addEventListener('click',()=>copyText(el.textContent.trim())));
}
function restoreCollapsed(){
  let saved=[];try{saved=JSON.parse(store.get('ncu-view-collapsed')||'[]');}catch(e){}
  saved.forEach(sid=>{const el=$('#sec-'+sid);if(el)el.classList.add('collapsed');});
}
function spy(){
  const secs=$$('#main .sec').filter(el=>el.id);
  if(!secs.length)return;
  if(typeof IntersectionObserver==='undefined')return;
  const io=new IntersectionObserver(es=>{es.forEach(e=>{
    if(!e.isIntersecting)return;
    const sid=e.target.dataset.sid;
    $$('#sidebar .nav-item[data-key="__sec__"]').forEach(x=>x.classList.toggle('active',x.dataset.sec===sid));});},
    {rootMargin:'-10% 0px -80% 0px'});
  secs.forEach(s=>io.observe(s));
}
function pushHash(h){
  if(typeof history!=='undefined'&&history.replaceState)history.replaceState(null,'',h);
}
function applyHash(){
  const h=typeof location!=='undefined'?location.hash:'';
  if(!h)return;
  if(h==='#s'){curView='summary';renderMain();return;}
  if(h.startsWith('#sec-')){curView='kernel';renderMain();const sid=h.slice(5);
    setTimeout(()=>{const t=$('#sec-'+sid);if(t)t.scrollIntoView({behavior:'smooth'});},30);return;}
  const k=DATA.kernels.find(x=>x.key===h.slice(3));
  if(k){curView='kernel';curKernel=k.key;renderMain();}
}
function keynav(e){
  if(e.target&&(e.target.tagName==='INPUT'||e.target.tagName==='TEXTAREA'))return;
  const idx=DATA.kernels.findIndex(x=>x.key===curKernel);
  if(e.key==='ArrowRight'&&idx<DATA.kernels.length-1){curView='kernel';curKernel=DATA.kernels[idx+1].key;pushHash('#k-'+curKernel);renderMain();}
  else if(e.key==='ArrowLeft'&&idx>0){curView='kernel';curKernel=DATA.kernels[idx-1].key;pushHash('#k-'+curKernel);renderMain();}
  else if(e.key==='s'||e.key==='S'){curView='summary';pushHash('#s');renderMain();}
  else if(e.key==='/'&&e.target!==$('#search')){$('#search').focus();e.preventDefault();}
}
document.addEventListener('DOMContentLoaded',()=>{
  themeInit();
  const app=$('#app-title');if(app)app.textContent=DATA.meta.input;
  const src=$('#app-src');if(src)src.textContent=DATA.meta.source?' · '+DATA.meta.source:'';
  const count=$('#app-count');if(count)count.textContent=DATA.kernels.length+' kernel'+(DATA.kernels.length===1?'':'s');
  const themeBtn=$('#theme-toggle');if(themeBtn)themeBtn.addEventListener('click',()=>{
    setTheme(document.documentElement.getAttribute('data-theme')==='light'?'dark':'light');});
  const search=$('#search');if(search)search.addEventListener('input',()=>{searchQ=search.value.trim();applyFilter();});
  const printBtn=$('#print-btn');if(printBtn)printBtn.addEventListener('click',()=>{if(typeof window!=='undefined'&&window.print)window.print();});
  if(typeof window!=='undefined')window.addEventListener('hashchange',applyHash);
  sidebar();renderMain();restoreCollapsed();applyHash();
  document.addEventListener('keydown',keynav);
});
"""


def render_html(report: dict) -> str:
    payload = json.dumps(report, sort_keys=True)
    js = ICONS_JS.replace("__DATA__", payload)
    title = "ncu-view"
    if report.get("meta", {}).get("input"):
        title += " — " + report["meta"]["input"]
    return f"""<!DOCTYPE html>
<html lang="en" data-theme="dark">
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
  <div class="hdr-actions">
    <span class="search">
      <svg viewBox="0 0 24 24" width="14" height="14" aria-hidden="true"><circle cx="11" cy="11" r="7"/><path d="M20 20l-4-4"/></svg>
      <input id="search" type="text" placeholder="Find kernel…" aria-label="Find kernel" autocomplete="off">
      <span class="kbd">/</span>
    </span>
    <button class="icon-btn" id="theme-toggle" title="Toggle theme" aria-label="Toggle theme">
      <svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>
    </button>
    <button class="icon-btn" id="print-btn" title="Print / save PDF" aria-label="Print">
      <svg viewBox="0 0 24 24"><path d="M6 9V2h12v7"/><path d="M6 18H4a2 2 0 0 1-2-2v-5a2 2 0 0 1 2-2h16a2 2 0 0 1 2 2v5a2 2 0 0 1-2 2h-2"/><rect x="6" y="14" width="12" height="8"/></svg>
    </button>
  </div>
</header>
<div class="layout">
  <aside id="sidebar"></aside>
  <main id="main"></main>
</div>
</body>
</html>
"""
