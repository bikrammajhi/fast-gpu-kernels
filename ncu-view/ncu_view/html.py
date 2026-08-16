"""Render a standalone Nsight-Compute-style HTML report.

Black/white Nsight Compute dashboard (reference design): fixed header with
global search, sidebar (report nav, kernel navigation, Nsight compute
sections), KPI strip, hero grid (verdict / optimization signals / stall
donut), tabbed panels (Performance Analysis / Metric Model / Collection &
Replay / Reference). Every number is NVIDIA's own exported data from the
profile — section rows, rule results, counters, device attributes; the
documentation panels are a synthesis of the NVIDIA Profiling Guide.
"""

REFERENCE_CSS = r"""
:root{
  --bg:#05080c;--panel:#070c12;--panel2:#0a121a;--panel3:#0e1822;
  --line:#14222e;--line2:#1c2f40;--text:#bff5d2;--muted:#7cb391;--dim:#4b6b5c;
  --accent:#38f07c;--good:#38f07c;--warn:#ffd47e;--bad:#ff7070;
  --sidebar:278px;--header:60px;
}
html[data-theme="light"]{
  --bg:#f2f4ef;--panel:#fafbf7;--panel2:#eef1ea;--panel3:#e4e9df;
  --line:#cfd6c9;--line2:#b9c4b2;--text:#14241b;--muted:#4c6153;--dim:#75867c;
  --accent:#0e9d4f;--good:#1f7a3d;--warn:#8a6d1f;--bad:#9c3a3a;
}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{
  margin:0;background:var(--bg);color:var(--text);
  font:12.5px/1.55 ui-monospace,"JetBrains Mono",SFMono-Regular,Menlo,Consolas,"Liberation Mono",monospace;
}
body::before{
  content:"";position:fixed;inset:0;z-index:9998;pointer-events:none;
  background:repeating-linear-gradient(0deg,rgba(56,240,124,.022) 0 1px,transparent 1px 4px);
}
button,input,select{font:inherit}
button{color:inherit}
a{color:inherit}
::-webkit-scrollbar{width:9px;height:9px}
::-webkit-scrollbar-thumb{background:#1c2f40;border-radius:9px}
::-webkit-scrollbar-track{background:#04060a}

header{
  position:fixed;z-index:100;left:0;right:0;top:0;height:var(--header);
  display:flex;align-items:center;padding:0 18px;border-bottom:1px solid var(--line);
  background:rgba(4,6,10,.97);backdrop-filter:blur(12px)
}
html[data-theme="light"] header{background:rgba(250,251,247,.97)}
.brand{display:flex;align-items:center;gap:10px;min-width:340px}
.logo{font-weight:700;font-size:19px;letter-spacing:2px;color:var(--accent);text-shadow:0 0 8px rgba(56,240,124,.35)}
.eye{width:25px;height:14px;border:2px solid var(--accent);border-radius:50%;position:relative;box-shadow:0 0 6px rgba(56,240,124,.3)}
.eye:after{content:"";position:absolute;width:6px;height:6px;background:var(--accent);border-radius:50%;left:7px;top:2px;box-shadow:0 0 5px rgba(56,240,124,.7)}
.vline{height:25px;width:1px;background:var(--line2);margin:0 6px}
.brand-title{font-size:13px;font-weight:650;color:var(--muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.tty-status{font-size:9px;color:var(--dim);letter-spacing:1px;white-space:nowrap}
.tty-status b{color:var(--accent);font-weight:700}
.header-spacer{flex:1}
.global-search{height:35px;width:300px;border:1px solid var(--line2);background:var(--panel2);border-radius:0;display:flex;align-items:center;padding:0 10px;color:var(--dim);position:relative}
.global-search input{background:none;border:0;outline:0;color:var(--text);width:100%;font-size:11px}
.search-pop{display:none;position:absolute;top:34px;right:-1px;width:440px;max-height:430px;overflow-y:auto;background:var(--panel2);border:1px solid var(--line2);z-index:60;box-shadow:0 8px 24px rgba(0,0,0,.5)}
.search-grp{font:8px monospace;letter-spacing:1px;text-transform:uppercase;color:var(--dim);padding:7px 9px 3px;border-top:1px solid var(--line)}
.search-grp:first-child{border-top:0}
.search-row{display:grid;grid-template-columns:18px 1fr;gap:7px;align-items:center;padding:6px 9px;cursor:pointer}
.search-row .k{font:8px monospace;color:var(--dim);border:1px solid var(--line2);text-align:center;width:16px;height:16px;line-height:14px}
.search-row.active{background:var(--line)}
.search-row b{font-size:9.5px;color:var(--text);font-weight:600}
.search-row i{display:block;font-style:normal;font-size:8px;color:var(--dim);margin-top:2px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.search-empty{font-size:9px;color:var(--dim);padding:10px}
kbd{border:1px solid var(--line2);border-radius:0;padding:2px 5px;color:var(--accent);font-size:9px}
.hbtn{height:36px;border:0;border-left:1px solid var(--line);background:transparent;padding:0 15px;font-size:10px;letter-spacing:.5px;cursor:pointer;color:var(--muted)}
.hbtn:hover{background:var(--accent);color:#03130b;text-shadow:none}

aside{
  position:fixed;left:0;top:var(--header);bottom:0;width:var(--sidebar);
  border-right:1px solid var(--line);background:var(--bg);overflow:auto;padding:17px 12px
}
.side-label{font-size:9px;font-weight:700;letter-spacing:1.5px;color:var(--dim);margin:6px 9px 9px;border-left:2px solid var(--accent);padding-left:7px}
.side-search{height:33px;border:1px solid var(--line2);background:var(--panel2);border-radius:0;display:flex;align-items:center;padding:0 9px;margin:0 4px 12px;color:var(--dim)}
.side-search input{border:0;outline:0;background:none;color:var(--text);width:100%;font-size:10px;margin-left:6px}
.nav-item{height:32px;border-radius:0;display:flex;align-items:center;padding:0 9px;color:var(--muted);font-size:10.5px;cursor:pointer}
.nav-item:hover{background:var(--panel3);color:var(--text)}
.nav-item.active{background:var(--panel3);color:var(--accent);box-shadow:inset 2px 0 var(--accent)}
.nav-icon{width:19px;color:var(--muted)}
.badge-count{margin-left:auto;color:var(--dim)}
.sep{height:1px;background:var(--line);margin:14px 8px}
.kernel{height:31px;display:flex;align-items:center;border-radius:0;padding:0 7px;color:var(--muted);font-size:10px;cursor:pointer}
.kernel:hover{background:var(--panel3)}
.kernel.active{background:var(--panel3);color:var(--text);box-shadow:inset 2px 0 var(--accent)}
.kernel .r{width:23px;color:var(--dim)}
.kernel .n{flex:1;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.kname{cursor:pointer;color:var(--text)}
.kname .kf{display:none}
.kname.full .ks{display:none}
.kname.full .kf{display:inline;white-space:normal;overflow-wrap:anywhere}
.kernel .t{font-size:9px;color:var(--dim)}

main{margin-left:var(--sidebar);padding-top:var(--header)}
.content{max-width:1500px;margin:auto;padding:22px 20px 80px}
.page-head{display:flex;align-items:center;margin:0 0 12px 4px;flex-wrap:wrap;gap:6px}
.page-head h1{font-size:18px;margin:0;font-weight:700;letter-spacing:0;overflow-wrap:anywhere}
.page-head h1::before{content:"❯ ";color:var(--accent)}
.page-head .sub{margin-left:15px;color:var(--dim);font-size:10px}
.page-head .right{margin-left:auto;color:var(--dim);font-size:10px}
.source-link{border-bottom:1px dotted var(--dim);cursor:pointer;color:var(--accent)}

.kpis{display:grid;grid-template-columns:repeat(8,1fr);border:1px solid var(--line);border-radius:0;background:var(--panel);overflow:hidden;margin-bottom:10px}
.kpi{padding:14px 16px;border-right:1px solid var(--line);min-width:0}
.kpi:last-child{border:0}
.kpi .value{font:700 17px monospace;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;color:var(--accent)}
.kpi .label{font-size:9px;color:var(--muted);margin-top:5px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.kpi .track{height:4px;background:var(--line);border-radius:0;overflow:hidden;margin-top:8px}
.kpi .fill{height:100%;background:var(--accent)}

.card{background:var(--panel);border:1px solid var(--line);border-radius:0;overflow:hidden;margin-bottom:10px}
.card-head{height:43px;border-bottom:1px solid var(--line);display:flex;align-items:center;padding:0 13px;cursor:pointer;background:var(--panel2)}
.card-head .arrow{width:21px;color:var(--accent)}
.card-head .title{font-size:11px;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:var(--accent)}
.card-head .title::after{content:"▌";margin-left:7px;color:var(--accent);animation:ttyblink 1.1s steps(1) infinite}
.card-head .meta{margin-left:auto;color:var(--dim);font-size:9px}
@keyframes ttyblink{50%{opacity:0}}
.src-tag{font:8px monospace;border:1px solid var(--line2);border-radius:0;padding:1px 6px;margin-left:8px;letter-spacing:.5px;text-transform:uppercase;color:var(--muted)}
.card.collapsed .body{display:none}
.card.collapsed .arrow{transform:rotate(-90deg)}
.body{padding:16px 18px}

.hero-grid{display:grid;grid-template-columns:1fr 1.45fr 1.15fr;gap:10px}
.hero-grid .card{min-height:178px}
.eyebrow{text-transform:uppercase;letter-spacing:1px;color:var(--dim);font-size:9px;margin-bottom:13px}
.verdict{font-size:18px;font-weight:700;margin-bottom:8px;overflow-wrap:anywhere}
.verdict.sev-critical{color:var(--bad);text-shadow:0 0 12px rgba(255,112,112,.35)}
.verdict.sev-warning{color:var(--warn);text-shadow:0 0 12px rgba(255,212,126,.3)}
.verdict.sev-suggestion,.verdict.sev-info{color:var(--text)}
.copy{color:var(--muted);font-size:10.5px;line-height:1.65}
.focus{margin-top:9px;display:flex;flex-direction:column;gap:3px;border-top:1px solid var(--line);padding-top:8px}
.fpair{display:flex;align-items:baseline;gap:8px;font-size:10px;line-height:1.5}
.fpair .fname{color:var(--dim);flex:1;min-width:0;overflow-wrap:anywhere}
.fpair b{color:var(--text);font-weight:700;font-size:10.5px;font-family:monospace;flex:0 0 auto}
.rec-title{font-size:8.5px;color:var(--dim);letter-spacing:.8px;text-transform:uppercase;margin-bottom:9px}
.rec{display:flex;align-items:center;min-height:29px;gap:8px;cursor:pointer;padding:2px 0}
.recno{width:20px;height:20px;border:1px solid var(--line2);border-radius:0;display:grid;place-items:center;color:var(--accent);font-size:9px;flex:0 0 20px}
.rectext{flex:1;font-size:10px;color:var(--text);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.rec-name{display:block;font-weight:700;color:var(--text)}
.impact{font-size:8.5px;border:1px solid var(--line2);border-radius:0;padding:3px 7px;color:var(--text);flex:0 0 auto}
.recmsg{font-size:9.5px;color:var(--muted);line-height:1.55;padding:2px 0 6px 28px;display:none}
.recmsg.open{display:block}
.donutrow{display:flex;align-items:center;gap:18px;flex-wrap:wrap}
.donut{width:118px;height:118px;border-radius:50%;position:relative;flex:0 0 118px}
.donut:after{content:"";position:absolute;inset:24px;background:var(--panel);border-radius:50%}
.dc{position:absolute;inset:0;z-index:2;display:grid;place-content:center;text-align:center}
.dc b{font:700 18px monospace;color:var(--accent)}
.dc span{font-size:8px;color:var(--dim)}
.legend{flex:1;min-width:150px}
.legend-row{height:21px;display:flex;align-items:center;color:var(--muted);font-size:9px}
.dot{width:6px;height:6px;border-radius:50%;margin-right:7px;flex:0 0 6px}
.legend-row b{margin-left:auto;color:var(--text);font-size:9px}

.tabs{display:flex;gap:5px;border-bottom:1px solid var(--line);margin:0 0 10px;overflow-x:auto}
.tab{padding:9px 12px;color:var(--dim);font-size:10px;cursor:pointer;border-bottom:2px solid transparent;white-space:nowrap;letter-spacing:.5px}
.tab.active,.tab:hover{color:var(--accent);border-color:var(--accent)}
.tab-panel{display:none}
.tab-panel.active{display:block}

.grid5{display:grid;grid-template-columns:repeat(5,1fr)}
.metric{padding:0 18px;border-right:1px solid var(--line);min-height:66px}
.metric:first-child{padding-left:0}
.metric:last-child{border:0}
.metric .label{font-size:9px;color:var(--muted)}
.metric .value{font:600 14px monospace;margin-top:7px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.metric .small{font-size:8.5px;color:var(--dim);margin-top:6px}
.track{height:6px;background:var(--line);border-radius:0;overflow:hidden;margin-top:8px}
.fill{height:100%;background:var(--accent);box-shadow:0 0 6px rgba(56,240,124,.5)}

.two{display:grid;grid-template-columns:1.18fr 1fr;gap:28px}
.three{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}
.chart-title{font-size:9px;color:var(--muted);margin-bottom:8px}
.mem-block{margin-top:16px}
.mc-unit{stroke-width:1.2}
.mc-lon{fill:#1c3f2c;stroke:#46a860}
.mc-pon{fill:#1a3a52;stroke:#3f9fd8}
.mc-loff,.mc-poff{fill:#161b20;stroke:#2a323a}
.mc-t{fill:#cfe8d8;font:9px monospace;pointer-events:none}
.mc-pon .mc-t,.mc-t{fill:#d4e7f5}
.mc-toff{fill:#5d6672}
.mc-link{font:8px monospace;pointer-events:none}
.mc-lg{fill:var(--dim);font:8px monospace}
.mem-table thead th{font-weight:600;text-align:right;color:var(--muted);font-size:8.5px;padding:4px 8px;border-bottom:1px solid var(--line)}
.mem-table td.l{white-space:nowrap}
.mem-table td.num{text-align:right}
.mem-table-block{margin-top:14px;padding-top:10px;border-top:1px solid var(--line)}
.mem-table-block:first-of-type{margin-top:10px}
.mem-t-title{font-size:9px;font-weight:600;color:var(--accent);text-transform:uppercase;letter-spacing:.05em;margin-bottom:6px}
.mcu{font-size:8px;color:var(--dim)}
svg{display:block;width:100%;height:auto}
.gridline{stroke:var(--line);stroke-width:1}
.axis{fill:var(--dim);font:8px monospace}
.chartline{fill:none;stroke:var(--accent);stroke-width:1.5}
.bar-row{display:grid;grid-template-columns:145px 1fr 50px;gap:9px;align-items:center;height:30px;color:var(--muted);font-size:9px}
.bar{height:6px;background:var(--line);border-radius:0;overflow:hidden}
.bar i{display:block;height:100%;background:var(--accent);box-shadow:0 0 6px rgba(56,240,124,.5)}
.bar-row b{font:9px monospace;color:var(--text);text-align:right}

.data-table{width:100%;border-collapse:collapse;font-size:9.5px;margin-top:10px}
.data-table th{text-align:left;text-transform:uppercase;letter-spacing:.7px;color:var(--accent);font-size:8px;padding:7px 9px;border-bottom:1px solid var(--line2)}
.data-table td{padding:8px 9px;border-bottom:1px solid var(--line);color:var(--muted);vertical-align:top}
.data-table td.l{color:var(--text);white-space:nowrap}
.data-table td.num{text-align:right;color:var(--text);font-family:monospace;white-space:nowrap}
.data-table td.wrap{white-space:normal;overflow-wrap:anywhere;text-align:left;line-height:1.55}
.data-table td.mref-name{white-space:normal;overflow-wrap:anywhere}
.data-table tr:hover td{background:var(--panel2)}
.data-table td code{font:9px ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--text)}
.mref-absent{color:var(--dim)}
.mref-str{color:var(--accent)}

.section-list{display:grid;grid-template-columns:repeat(2,1fr);gap:7px}
.section-item{border:1px solid var(--line2);background:var(--panel2);border-radius:0;padding:11px;cursor:pointer}
.section-item:hover{border-color:var(--accent);background:var(--panel3)}
.section-item h3{font-size:10px;margin:0 0 4px;color:var(--text)}
.section-item h3 .present{font-size:7.5px;border:1px solid var(--good);color:var(--good);border-radius:0;padding:1px 5px;margin-left:6px;text-transform:uppercase;letter-spacing:.5px}
.section-item p{font-size:9px;color:var(--dim);margin:0;line-height:1.45}
.section-id{font-size:8px;color:var(--dim);margin-top:6px}

.callout{border:1px solid var(--line2);border-left:2px solid var(--accent);background:var(--panel2);border-radius:0;padding:11px 12px;color:var(--muted);font-size:9.5px;margin-bottom:9px}
.callout::before{content:"❯ ";color:var(--accent)}
.derived-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(215px,1fr));gap:9px;margin-bottom:9px}
.dgroup{grid-column:1/-1;font:700 9px monospace;letter-spacing:1.4px;color:var(--accent);text-transform:uppercase;border-bottom:1px dashed var(--line2);padding:2px 0 6px}
.dgroup::before{content:'❯ '}
.dmetric{border:1px dashed var(--line2);border-radius:0;padding:10px 11px;background:var(--panel2);cursor:pointer;user-select:none}
.dmetric .dname{font-size:9px;font-weight:700;letter-spacing:.4px;text-transform:uppercase;color:var(--muted);margin-bottom:6px}
.dmetric .dtog{float:right;font:10px monospace;color:var(--dim);transition:transform .15s}
.dmetric.open .dtog{transform:rotate(90deg)}
.dmetric .dval{font:600 18px/1.2 monospace;color:var(--accent);text-shadow:0 0 10px rgba(56,240,124,.25)}
.dmetric .dunit{font:10px monospace;color:var(--muted);margin-left:5px}
.dmetric .ddesc{font-size:9px;font-style:italic;color:var(--muted);margin-top:5px;line-height:1.45}
.dmetric .ddetails{display:none;margin-top:7px}
.dmetric.open .ddetails{display:block}
.dmetric .dform{font:8.5px monospace;color:var(--muted);margin-top:7px;line-height:1.5}
.dmetric .dsrc{font:8px monospace;color:var(--dim);margin-top:4px;letter-spacing:.4px;text-transform:uppercase}
.dmetric .dnote{font-size:8.5px;color:var(--muted);margin-top:5px;line-height:1.5;border-top:1px solid var(--line2);padding-top:5px}
.src-tag.ours{border-color:var(--accent);color:var(--accent)}
.callout strong{color:var(--text)}
.code{background:var(--bg);border:1px solid var(--line);border-radius:0;padding:12px;font:9px/1.7 ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--text);overflow:auto;white-space:pre}
.pill{display:inline-block;border:1px solid var(--line2);border-radius:0;padding:2px 6px;font-size:8px;color:var(--text);margin:2px}
.definition{display:grid;grid-template-columns:210px 1fr;border-bottom:1px solid var(--line);padding:8px 0;gap:15px}
.definition:last-child{border:0}
.definition b{font:9px monospace;color:var(--text)}
.definition span{font-size:9px;color:var(--muted)}

.roofline{height:300px;background:#03050a;border:1px solid var(--line2);border-radius:0;position:relative;padding:10px}
.roofline svg{height:270px}
.rl-pt{fill:var(--accent);stroke:#03050a;stroke-width:1.5;filter:drop-shadow(0 0 4px rgba(56,240,124,.8))}
.rl-pt-label{font:8.5px monospace;fill:var(--text)}
.rl-note{font-size:8px;color:var(--dim)}
.rl-tables{margin:12px 0 4px}
.rl-t-title{font-size:9.5px;font-weight:700;letter-spacing:.5px;text-transform:uppercase;color:var(--muted);margin:12px 0 6px}
.rl-tables .data-table{width:100%;border-collapse:collapse}
.rl-tables .data-table td{font-size:9px;padding:4px 8px;border-bottom:1px solid var(--line)}
.rl-tables .data-table td.num{text-align:right;font-family:monospace}
.unit{border:1px solid var(--line2);border-radius:0;padding:10px;background:var(--panel2)}
.unit h4{font-size:10px;margin:0 0 5px}
.unit p{font-size:8.5px;color:var(--dim);margin:0}
.unit.active{border-color:var(--accent)}

.faq{border-bottom:1px solid var(--line);padding:10px 0}
.faq:last-child{border:0}
.faq h3{font-size:10px;margin:0 0 4px}
.faq p{font-size:9px;color:var(--muted);margin:0;line-height:1.55}
.footer{color:var(--dim);font-size:9px;padding:20px 4px;line-height:1.6}
.footer .source-link{color:var(--text)}

@media(max-width:1000px){.hero-grid,.three,.two{grid-template-columns:1fr}.grid5{grid-template-columns:1fr 1fr}.section-list{grid-template-columns:1fr}.kpis{grid-template-columns:repeat(4,1fr)}}
@media(max-width:900px){aside{display:none}main{margin-left:0}}
@media print{header,aside{display:none}main{margin-left:0}main{padding-top:0}.card{break-inside:avoid}}
@media (prefers-reduced-motion:reduce){*{transition:none!important;animation:none!important}html{scroll-behavior:auto}}
"""

ICONS_JS = r"""
const DATA = __DATA__;
const $=(s,el)=>(el||document).querySelector(s);
const $$=(s,el)=>Array.from((el||document).querySelectorAll(s));
const esc=s=>String(s==null?'':s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function kdisp(n){n=String(n==null?'':n);return n.replace(/^matmul[_\-.]/,'')||n;}
function kshort(n){n=kdisp(n);if(n.length<=28)return n;return n.slice(0,14)+'…'+(n.split('_').slice(-3).join('_')||n.slice(-15));}
function knameHtml(n){
  n=kdisp(n);
  const s=kshort(n);
  if(s===n)return esc(n);
  return `<span class="kname" title="click to show full name" onclick="event.stopPropagation();this.classList.toggle('full')"><span class="ks">${esc(s)}</span><span class="kf">${esc(n)}</span></span>`;
}
function estVal(r){const e=String(r.est==null?'':r.est).trim();if(!e)return -1;const n=parseFloat(e);if(Number.isNaN(n))return -1;return e.endsWith('%')?n/100:n;}
function estSort(a,b){return estVal(b)-estVal(a)||((SEV[a.severity]??9)-(SEV[b.severity]??9));}
const fmt=(v,d)=>v==null?'—':Number(v).toLocaleString('en-US',{maximumFractionDigits:d==null?2:d});
const SYS={K:1e3,M:1e6,G:1e9,T:1e12,P:1e15};
const SYSORD=['','K','M','G','T','P'];
function smart(v,u){
  if(v==null||isNaN(v))return ['—',''];
  if(v===0)return ['0',u||''];
  const un=u||'',a=Math.abs(v);
  if(a<0.001)return [Number(v).toExponential(3),un];
  const tm=un.match(/^(ns|us|µs|ms|s)(\/.*)?$/i);
  if(tm){
    const sec=v*{'ns':1e-9,'us':1e-6,'µs':1e-6,'ms':1e-3,'s':1}[tm[1].toLowerCase()];
    const a2=Math.abs(sec),suf=tm[2]||'';
    if(a2>=1)return [sec.toFixed(a2>=10?1:2),'s'+suf];
    if(a2>=1e-3)return [(sec*1e3).toFixed(a2>=1e-2?1:2),'ms'+suf];
    if(a2>=1e-6)return [(sec*1e6).toFixed(1),'µs'+suf];
    return [(sec*1e9).toFixed(1),'ns'+suf];
  }
  const rescale=(x,root,glue)=>{
    let n=x,e=0;
    while(Math.abs(n)>=1000&&e<5){n/=1000;e++;}
    while(Math.abs(n)<1&&e>0){n*=1000;e--;}
    if(e>=5&&Math.abs(n)>=1000)return [Number(x).toExponential(3),root];
    return [n.toFixed(Math.abs(n)>=100?1:2),glue&&root?SYSORD[e]+' '+root:SYSORD[e]+root];
  };
  const bRaw=un.match(/^B(\/.*)?$/);
  if(bRaw)return rescale(v,un);
  const bScaled=un.match(/^([KMGTP])B(\/.*)?$/);
  if(bScaled)return rescale(v*SYS[bScaled[1]],un.slice(1));
  const fl=un.match(/^(FLOP)(\/.*)?$/);
  if(fl){
    if(a>=1e15)return [(v/1e15).toFixed(3),'P'+un];
    if(a>=1e12)return [(v/1e12).toFixed(3),'T'+un];
    if(a>=1e9)return [(v/1e9).toFixed(3),'G'+un];
    if(a>=1e6)return [(v/1e6).toFixed(3),'M'+un];
    if(a>=1e3)return [(v/1e3).toFixed(2),'K'+un];
    return [fmt(v,3),un];
  }
  if(a>=1e3)return rescale(v,un,true);
  return [fmt(v,3),un];
}
const SEV={critical:0,warning:1,suggestion:2,info:3};
const SEVNAME={critical:'Critical',warning:'Warning',suggestion:'Suggestion',info:'Info'};
const ICONS={critical:'▲',warning:'●',suggestion:'◆',info:'ℹ'};
const MONO=['#ddd','#999','#666','#444','#222'];
const SID=s=>String(s||'').toLowerCase().replace(/[^a-z0-9]/g,'');

let curKernel=DATA.kernels.length?DATA.kernels.reduce((a,b)=>((b.stats.time_us??0)>(a.stats.time_us??0)?b:a)).key:null;
let curView='kernel';
let searchQ='';
let curTab='analysis';

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
  if(!t)t='dark';
  setTheme(t);
}

// ---------- data helpers (NVIDIA-verbatim only) ----------
function secOf(k,sid){return (k.sections||[]).find(s=>SID(s.sid)===SID(sid))||null;}
function rowsOf(k,sid){const s=secOf(k,sid);return s?s.rows:[];}
function rowOf(k,sid,label){return rowsOf(k,sid).find(r=>String(r.label).toLowerCase().startsWith(String(label).toLowerCase()))||null;}
function pct(r){return r?parseFloat(r.value):null;}
function rowVal(r){return r?r.value:null;}

function tableHtml(rows){
  return `<table class="data-table"><tbody>${rows.map(r=>{
    const isPct=r.unit==='%'&&r.bar!=null;
    const lw=String(r.label).length>42?' wrap':'';
    const vw=!isPct&&String(r.value).length>48?' wrap':'';
    const num=!isPct&&String(r.value).trim()!==''&&!isNaN(Number(r.value));
    const [dv,du]=num?smart(Number(r.value),r.unit||''):[String(r.value),''];
    return `<tr><td class="l${lw}">${esc(r.label)}</td>${isPct?`<td><div class="track"><div class="fill" style="width:${Math.min(r.bar,100)}%"></div></div></td>`:''}<td class="num${vw}">${esc(dv)}${du?` <span style="color:var(--dim)">${esc(du)}</span>`:''}</td></tr>`;}).join('')}</tbody></table>`;
}

function metric(label,val,unit,bar,note){
  const barHtml=bar!=null?`<div class="track"><div class="fill" style="width:${Math.min(bar,100)}%"></div></div>`:'';
  let vh='—';
  if(val!=null){
    if(bar!=null&&unit==='%')vh=fmt(val,Math.abs(val)<10?2:1);
    else {const [nv,nu]=smart(val,unit||'');vh=nv+(nu?` <span style="font-size:9px;color:var(--dim);font-family:inherit">${esc(nu)}</span>`:'');}
  }
  return `<div class="metric" title="${note?esc(note):''}"><div class="label">${esc(label)}</div><div class="value">${vh}</div>${barHtml}${note?`<div class="small">${esc(note)}</div>`:''}</div>`;
}

// ---------- hero ----------
function heroGrid(k){
  const v=k.verdict;
  const verdict=v?`<div class="verdict sev-${esc(v.severity)}">${esc(v.name)}</div>
      <div class="copy">${esc(v.message)}</div>
      ${focusEvidence(v)}
      <div class="small" style="font-size:8.5px;color:var(--dim);margin-top:8px;text-transform:uppercase;letter-spacing:.7px">${SEVNAME[v.severity]||v.severity} · NVIDIA rule result</div>`
    :'<div class="verdict">No verdict</div><div class="copy">No NVIDIA rule result for this input.</div>';
  const rules=(k.rules||[]).slice().sort(estSort);
  const recs=rules.length?`<div class="rec-title">Recommendations (NVIDIA rule engine)</div>`+rules.map((r,i)=>`
      <div class="rec" title="click to expand/collapse">
        <div class="recno">${i+1}</div>
        <div class="rectext">${esc(r.name)}</div>
        <div class="impact">${r.est?'est. '+esc(r.est):SEVNAME[r.severity]||r.severity}</div>
      </div>
      <div class="recmsg">${esc(r.message)}${focusHtml(r)}</div>`).join('')
    :'<div class="copy">No NVIDIA rule results for this input.</div>';
  const stall=donutHtml((k.stats&&k.stats.stall_reasons)||[]);
  return `<div class="hero-grid">
    <section class="card"><div class="body"><div class="eyebrow">Verdict</div>${verdict}</div></section>
    <section class="card"><div class="body"><div class="eyebrow">Top optimization signals</div>${recs||'<div class="copy">No NVIDIA rule results for this input.</div>'}</div></section>
    <section class="card"><div class="body"><div class="eyebrow">Stall reason breakdown</div>${stall||'<div class="copy">No stall counter data in this input.</div>'}</div></section>
  </div>`;
}
const fv=x=>{const a=Math.abs(x);if(a>0&&a<0.01)return Number(x).toExponential(2);if(a>=1e4){const [nv,nu]=smart(x,'');return nv+nu;}return fmt(x,3);};
function focusHtml(r){
  const info=r.focus_info||{};
  const pairs=Object.keys(r.focus||{}).map(n=>{
    const hint=info[n]?`<div style="font-size:8.5px;color:var(--dim);line-height:1.5;overflow-wrap:anywhere">${esc(info[n])}</div>`:'';
    return `<div class="fpair"><span class="fname">${esc(n)}:</span> <b>${fv(r.focus[n])}</b></div>${hint}`;
  });
  return pairs.length?`<div class="focus">${pairs.join('')}</div>`:'';
}
function focusEvidence(v){
  const info=v.focus_info||{};
  const pairs=Object.keys(v.focus||{}).map(n=>{
    const hint=info[n]?`<div style="font-size:8.5px;color:var(--dim);line-height:1.5;overflow-wrap:anywhere">${esc(info[n])}</div>`:'';
    return `<div class="fpair"><span class="fname">${esc(n)}:</span> <b>${fv(v.focus[n])}</b></div>${hint}`;
  });
  return pairs.length?`<div class="focus">${pairs.join('')}</div>`:'';
}
function donutHtml(reasons){
  const rows=reasons.filter(r=>r.cycles>=0.01);
  if(!rows.length)return '';
  const sum=rows.reduce((a,r)=>a+r.cycles,0);
  const top=rows.slice(0,5);
  const other=rows.slice(5).reduce((a,r)=>a+r.cycles,0);
  if(other>=0.01)top.push({name:'Other',cycles:other});
  let acc=0;const segs=top.map(r=>{const from=acc;acc+=r.cycles/sum*100;return [from,acc,r];});
  const grad=segs.map(([a,b,r],i)=>`${MONO[i%MONO.length]} ${a.toFixed(2)}% ${b.toFixed(2)}%`).join(',');
  const share=x=>(x.cycles/sum*100);
  return `<div class="donutrow"><div class="donut" style="background:conic-gradient(${grad})"><div class="dc"><b>${fmt(share(top[0]),1)}%</b><span>STALL</span></div></div>
    <div class="legend">${top.map((r,i)=>`<div class="legend-row"><i class="dot" style="background:${MONO[i%MONO.length]}"></i>${esc(r.name)}<b>${fmt(share(r),1)}%</b></div>`).join('')}</div></div>`;
}

// ---------- KPI strip ----------
function kpis(k){
  const s=k.stats||{};
  const smClock=rowOf(k,'SpeedOfLight','SM Frequency');
  const dram=rowOf(k,'SpeedOfLight','DRAM Throughput');
  const memBusy=rowOf(k,'SpeedOfLight','Memory Throughput');
  const elapsed=rowOf(k,'SpeedOfLight','Elapsed Cycles');
  const c=[];
  c.push(metric('Duration',s.time_us!=null?s.time_us:null,'µs'));
  c.push(metric('SM clock',smClock?pct(smClock):null,smClock?smClock.unit:'',null,'NVIDIA SOL SM Frequency'));
  c.push(metric('Tensor pipe',s.pipe_pct!=null?s.pipe_pct:null,'%',s.pipe_pct));
  c.push(metric('Memory Busy',memBusy?pct(memBusy):null,'%',memBusy?pct(memBusy):null,'NVIDIA SOL Memory Throughput'));
  c.push(metric('DRAM Throughput',dram?pct(dram):null,'%',dram?pct(dram):null));
  c.push(metric('Stall / issue',s.stall_cycles!=null?s.stall_cycles:null,'cyc',null,'Warp cycles per issued instruction — NVIDIA Warp State Statistics'));
  c.push(metric('Achieved occupancy',s.occupancy_pct!=null?s.occupancy_pct:null,'%',s.occupancy_pct));
  c.push(metric('Elapsed Cycles',elapsed?pct(elapsed):null,elapsed?elapsed.unit:'',null));
  return `<div class="kpis">${c.join('').replaceAll('class="metric"','class="kpi"')}</div>`;
}

// ---------- tabs ----------
function tabsHtml(){
  return `<div class="tabs">
    <div class="tab active" data-panel="analysis">PERFORMANCE ANALYSIS</div>
    <div class="tab" data-panel="metrics">METRIC MODEL</div>
    <div class="tab" data-panel="collection">COLLECTION &amp; REPLAY</div>
    <div class="tab" data-panel="reference">REFERENCE</div></div>`;
}
function showTab(name){
  curTab=name;
  $$('#main .tab').forEach(x=>x.classList.toggle('active',x.dataset.panel===name));
  $$('#main .tab-panel').forEach(p=>p.classList.toggle('active',p.id==='panel-'+name));
  $$('#sidebar .nav-item[data-tab]').forEach(x=>x.classList.toggle('active',x.dataset.tab===name));
}
function bindTabs(){
  $$('#main .tab').forEach(t=>t.addEventListener('click',()=>showTab(t.dataset.panel)));
}

// ---------- section cards ----------
function sectionCard(k,s){
  const sid=SID(s.sid);
  const head=`<div class="card-head"><span class="arrow">⌄</span><span class="title">${esc(s.title)}</span>
    <span class="meta">${esc(s.sid)}<span class="src-tag">${esc(s.src||'NVIDIA')}</span></span></div>`;
  let body;
  if(sid==='speedoflight')body=solBody(k,s);
  else if(sid==='computeworkloadanalysis')body=computeBody(k,s);
  else if(sid==='memoryworkloadanalysis')body=memBody(k,s);
  else if(sid==='schedulerstats')body=schedBody(k,s);
  else if(sid==='warpstatestats')body=warpBody(k,s);
  else if(sid==='occupancy')body=occBody(k,s);
  else if(sid==='sourcecounters')body=sourceBody(k,s);
  else if(sid==='pmsampling')body=samplingBody(k,s);
  else if(sid==='launchstats')body=launchBody(k,s);
  else if(sid==='instructionstats')body=instructionBody(k,s);
  else body=tableBody(s);
  return `<section class="card" id="sec-${esc(s.sid)}" data-sid="${esc(s.sid)}">${head}<div class="body">${body}</div></section>`;
}
function tableBody(s){
  return (s.desc?`<div class="callout">${esc(s.desc)}</div>`:'')+tableHtml(s.rows||[]);
}
function solBody(k,s){
  const r=l=>rowOf(k,'SpeedOfLight',l);
  const g5=metric('Compute Throughput',pct(r('Compute (SM) Throughput')),'%',pct(r('Compute (SM) Throughput')),'NVIDIA SOL row')
    +metric('Memory Throughput',pct(r('Memory Throughput')),'%',pct(r('Memory Throughput')))
    +metric('L2 Throughput',pct(r('L2 Cache Throughput')),'%',pct(r('L2 Cache Throughput')))
    +metric('DRAM Throughput',pct(r('DRAM Throughput')),'%',pct(r('DRAM Throughput')))
    +metric('Duration',pct(r('Duration')),r('Duration')?r('Duration').unit:'',null);
  return `<div class="grid5">${g5}</div>
    <div style="height:14px"></div>
    <div class="roofline">${rlChart(k)}</div>
    ${rlNote(k)}
    <div class="callout" style="margin-top:10px"><strong>How to read it:</strong> the vertical axis is FLOPS and the horizontal axis is arithmetic intensity; both are logarithmic in the NVIDIA roofline model. The sloped boundary is the memory-bandwidth ceiling, the flat boundary is peak compute, and the intersection is the ridge point.</div>
    ${tableHtml(s.rows||[])}`;
}
function rlChart(k){
  const rl=k.roofline||{};
  const a=rl.achieved,e=rl.envelope;
  if(!a||!e||a.ai==null||a.flops_s==null||!e.peak_dram_bw)return '';
  const X0=-7,X1=3,Y0=5,Y1=15,L=70,R=30,T=35,B=42,W=900-L-R,H=270-T-B;
  const fx=x=>L+(Math.log10(x)-X0)/(X1-X0)*W;
  const fy=y=>T+H-(Math.log10(y)-Y0)/(Y1-Y0)*H;
  const clim=v=>Math.min(T+H,Math.max(T,v));
  const ridge=e.ridge&&e.ridge>a.ai*10?e.ridge:a.ai*10;
  const ridgeX=Math.min(ridge,Math.pow(10,X1));
  const hasCompute=e.peak_compute_flops?true:false;
  const xs=[Math.pow(10,X0),ridgeX];
  const roofY1=clim(fy(e.peak_dram_bw*xs[0]));
  const roofY2=clim(fy(e.peak_dram_bw*xs[1]));
  let svg=`<svg viewBox="0 0 900 270">`;
  for(let d=0;d<=10;d+=2){
    const x=Math.pow(10,X0+d);
    svg+=`<line x1="${fx(x)}" y1="${T}" x2="${fx(x)}" y2="${T+H}" class="gridline"/>`;
    if(d%4===0)svg+=`<text x="${fx(x)+3}" y="${270-B+12}" class="axis">1e${X0+d}</text>`;
  }
  for(let d=0;d<=10;d+=2){
    const y=Math.pow(10,Y0+d);
    svg+=`<line x1="${L}" y1="${fy(y)}" x2="${900-R}" y2="${fy(y)}" class="gridline"/>`;
    if(d%4===0)svg+=`<text x="${L-4}" y="${fy(y)+3}" text-anchor="end" class="axis">1e${Y0+d-12}</text>`;
  }
  svg+=`<polyline points="${fx(xs[0]).toFixed(1)},${roofY1.toFixed(1)} ${fx(xs[1]).toFixed(1)},${roofY2.toFixed(1)}" fill="none" stroke="#777" stroke-width="2"/>`;
  if(hasCompute){
    svg+=`<polyline points="${fx(xs[1]).toFixed(1)},${clim(fy(e.peak_compute_flops)).toFixed(1)} ${900-R},${clim(fy(e.peak_compute_flops)).toFixed(1)}" fill="none" stroke="#777" stroke-width="2"/>`;
    svg+=`<line x1="${fx(xs[1]).toFixed(1)}" y1="${T}" x2="${fx(xs[1]).toFixed(1)}" y2="${T+H}" stroke="#555" stroke-dasharray="4 4"/>`;
    const rlx=fx(xs[1]);
    const flip=rlx+95>900-R;
    svg+=`<text x="${flip?rlx-6:rlx+4}" y="${roofY2+12}" class="axis"${flip?' text-anchor="end"':''}>Ridge Point</text>`;
  }
  const px=Math.min(900-R,Math.max(L,fx(a.ai))),py=Math.min(T+H,Math.max(T,fy(a.flops_s)));
  svg+=`<circle class="rl-pt" cx="${px.toFixed(1)}" cy="${py.toFixed(1)}" r="4.5"/>`;
  svg+=`<text x="${px+9}" y="${py-6}" class="rl-pt-label">Achieved</text>`;
  svg+=`<text x="76" y="${270-B-8}" class="axis">Memory Bound</text>`;
  if(hasCompute)svg+=`<text x="700" y="${clim(fy(e.peak_compute_flops))-8}" class="axis">Compute Bound</text>`;
  svg+=`<text x="73" y="260" class="axis">Arithmetic Intensity (FLOP/B) →</text><text x="5" y="50" class="axis">TFLOP/s ↑</text>`;
  svg+=`</svg>`;
  return svg;
}
function rlNote(k){
  const rl=k.roofline||{};
  const a=rl.achieved,e=rl.envelope;
  if(!a||!e||a.ai==null||a.flops_s==null)return `<div class="rl-note">Achieved point requires FLOP counters, which this export does not carry — no roofline geometry drawn.</div>`;
  const fma=a.fma&&a.fma.length?`${a.fma.map(f=>f.name.replace(/\s*\(.*\)/,'')+' '+smart(f.flops,'FLOP')[0]+' '+smart(f.flops,'FLOP')[1]).join(' + ')}`:'no FMA counters';
  const ai=smart(a.ai,'FLOP/B');
  const tpaths=(a.tensor||[]);
  const tf=tpaths.length?` + NVIDIA tensor ops-path ${tpaths.map(t=>t.name+' '+smart(t.flops,'FLOP')[0]+' '+smart(t.flops,'FLOP')[1]).join(' + ')} (${tpaths[0].pct!=null?fmt(tpaths[0].pct,2)+'% of NVIDIA peak':'n/a'})`:' (no tensor FLOPs in this profile)';
  const roof=e.peak_dram_bw?(smart(e.peak_dram_bw,'B/s')[0]+' '+smart(e.peak_dram_bw,'B/s')[1]):'—';
  const fs=smart(a.flops_s,'FLOP/s');
  const comp=e.peak_compute_flops?(smart(e.peak_compute_flops,'FLOP/s')[0]+' '+smart(e.peak_compute_flops,'FLOP/s')[1]+' ('+(e.peak_compute_source==='tensor'?'NVIDIA tensor ops-path peak':'NVIDIA FMA peak_sustained')+')'):'—';
  return `<div class="rl-note"><span class="src-tag ours" style="margin-right:6px">calculated by ncu-view</span>`+
    `Achieved point (${fs[0]} ${fs[1]} at ${ai[0]} ${ai[1]}) = NVIDIA's SASS FMA FLOP counters (${fma})${tf}. `+
    `Envelope: memory roof ${roof} (NVIDIA achieved bandwidth ÷ NVIDIA %-of-peak), compute roof ${comp}; ridge = compute roof ÷ memory roof.</div>`;
}
function computeBody(k,s){
  const r=l=>rowOf(k,'ComputeWorkloadAnalysis',l);
  const g5=metric('Executed IPC',pct(r('Executed Ipc Active')),r('Executed Ipc Active')?r('Executed Ipc Active').unit:'')
    +metric('Issued IPC',pct(r('Issued Ipc Active')),r('Issued Ipc Active')?r('Issued Ipc Active').unit:'')
    +metric('Issue Slots Busy',pct(r('Issue Slots Busy')),'%',pct(r('Issue Slots Busy')))
    +metric('SM Busy',pct(r('SM Busy')),'%',pct(r('SM Busy')))
    +metric('Tensor pipe',(k.stats&&k.stats.pipe_pct)||null,'%',(k.stats&&k.stats.pipe_pct));
  return `<div class="grid5">${g5}</div>
    <div class="callout" style="margin-top:13px"><strong>Interpretation:</strong> Compute Workload Analysis examines SM compute resources, achieved IPC and pipeline utilization. A highly utilized pipeline can become the limiting resource; use the individual pipeline breakdown to identify which one.</div>
    ${tableHtml(s.rows||[])}`;
}
function memBody(k,s){
  const sol=rowsOf(k,'SpeedOfLight');
  const r=l=>rowOf(k,'SpeedOfLight',l);
  const g5=metric('Mem Busy',pct(r('Memory Throughput')),'%',pct(r('Memory Throughput')),'SOL Memory Throughput')
    +metric('L1/TEX Throughput',pct(r('L1/TEX Cache Throughput')),'%',pct(r('L1/TEX Cache Throughput')))
    +metric('L2 Throughput',pct(r('L2 Cache Throughput')),'%',pct(r('L2 Cache Throughput')))
    +metric('DRAM Throughput',pct(r('DRAM Throughput')),'%',pct(r('DRAM Throughput')))
    +metric('Duration',pct(r('Duration')),r('Duration')?r('Duration').unit:'');
  return `<div class="grid5">${g5}</div>
    <div class="callout" style="margin-top:13px"><strong>Memory Workload Analysis</strong> examines memory utilization, bandwidth, pipes and access patterns. The rows below are NVIDIA's own rule results for this profile, verbatim.</div>
    ${tableHtml(s.rows||[])}
    ${memChartHtml(k)}${memTablesHtml(k)}`;
}
function memChartHtml(k){
  const mem=k.memory||{};
  const units=mem.units||[],links=mem.links||[];
  if(!units.length&&!links.length)return '<div class="rl-note">No memory counters in this export — the Memory Chart needs NVIDIA l1tex__*/lts__*/dram__* counters.</div>';
  const W=900,H=432,BW=105,BH=34,GP=12,LX=8,RX=724;
  const logical=units.filter(u=>u.kind==='logical'&&u.name!=='Kernel');
  const physical=units.filter(u=>u.kind==='physical');
  const idx={};units.forEach(u=>idx[u.name]=u);
  logical.forEach((u,i)=>{u.x=LX+i*(BW+GP);u.y=76;});
  const pw=physical.length*(BW+GP)-GP;
  physical.forEach((u,i)=>{u.x=LX+(RX-LX-pw)/2+i*(BW+GP);u.y=302;});
  const kk={x:LX+(RX-LX-BW)/2,y:8,name:'Kernel'};idx['Kernel']=kk;
  const clim=v=>Math.max(0,Math.min(100,v));
  const lcol=p=>{
    if(p==null)return '#3a434d';
    const t=clim(p)/100;
    const r0=18,g0=48,b0=74,r1=92,g1=196,b1=255;
    return `rgb(${Math.round(r0+(r1-r0)*t)},${Math.round(g0+(g1-g0)*t)},${Math.round(b0+(b1-b0)*t)})`;
  };
  const wrap=s=>{
    if(s.length<=15)return [s];
    const i=s.lastIndexOf(' ',13);
    return [s.slice(0,i>0?i:13),s.slice((i>0?i:13)+1)];
  };
  const box=u=>{
    const cls=u.kind==='logical'
      ?(u.active?'mc-lon':'mc-loff')
      :(u.active?'mc-pon':'mc-poff');
    const [l1,l2]=wrap(u.name);
    return `<rect x="${u.x}" y="${u.y}" width="${BW}" height="${BH}" rx="6" class="mc-unit ${cls}"/><text x="${u.x+BW/2}" y="${u.y+(l2?14:20)}" text-anchor="middle" class="mc-t${u.active?'':' mc-toff'}">${esc(l1)}</text>${l2?`<text x="${u.x+BW/2}" y="${u.y+26}" text-anchor="middle" class="mc-t${u.active?'':' mc-toff'}">${esc(l2)}</text>`:''}`;
  };
  let svg=`<svg viewBox="0 0 ${W} ${H}" class="memchart">`;
  links.forEach(l=>{
    const a=idx[l.from],b=idx[l.to];
    if(!a||!b)return;
    const x1=a.x+BW/2,y1=a.y+BH,x2=b.x+BW/2,y2=b.y,col=lcol(l.pct);
    const [nv,nu]=smart(l.value,l.unit);
    const lbl=l.pct!=null?`${nv} ${nu} · ${fmt(l.pct,1)}%`:`${nv} ${nu}`;
    const my=(y1+y2)/2;
    svg+=`<g><title>${esc(l.formula||'')}</title><line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${col}" stroke-width="2.5"/><text x="${x1}" y="${my-5}" text-anchor="middle" class="mc-link" fill="${l.pct==null?'#6b7682':col}">${esc(lbl)}</text></g>`;
  });
  svg+=`<g><title>Kernel — the profiled launch</title>${box(kk)}</g>`;
  units.filter(u=>u.name!=='Kernel').forEach(u=>svg+=`<g><title>${esc(u.name)} ${u.active?'active':'no traffic'}</title>${box(u)}</g>`);
  const lg=280,ly=84;
  svg+=`<text x="820" y="66" text-anchor="middle" class="mc-lg">Link % of</text><text x="820" y="80" text-anchor="middle" class="mc-lg">peak sustained</text>`;
  const defs=`<defs><linearGradient id="mcgrad" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="rgb(92,196,255)"/><stop offset="1" stop-color="rgb(18,48,74)"/></linearGradient></defs>`;
  svg+=`${defs}<rect x="816" y="${ly}" width="8" height="${lg}" rx="3" fill="url(#mcgrad)"/>`;
  svg+=`<text x="832" y="${ly+lg+8}" class="mc-lg">0%</text><text x="832" y="${ly-4}" class="mc-lg">100%</text>`;
  links.forEach(l=>{
    if(l.pct==null)return;
    const y=ly+lg-clim(l.pct)/100*lg;
    svg+=`<polygon points="810,${y} 816,${y-5} 816,${y+5}" fill="${lcol(l.pct)}"/>`;
  });
  svg+=`</svg>`;
  return `<div class="mem-block"><div class="chart-title">Memory Chart <span class="src-tag ours">calculated by ncu-view</span></div>${svg}
    <div class="rl-note">Mirrors the NVIDIA Memory Workload Analysis chart: green units generate memory traffic (logical), blue units service it (physical). Link values are NVIDIA's own counters (inst = executed SASS instructions, req = requests, sectors/B = traffic); link color is NVIDIA's % of peak sustained (grey: no pct counter exported; units grey = no traffic). All numbers come from the profile — nothing is invented.</div></div>`;
}
function memTablesHtml(k){
  const mem=k.memory||{};
  const tables=['shared','l1','l2','texops','evict','dram'].map(x=>mem.tables[x]).filter(Boolean);
  if(!tables.length)return '<div class="rl-note">No memory tables — the profile carries no l1tex__*/lts__*/dram__* counters.</div>';
  return `<div class="mem-block"><div class="chart-title" style="margin-top:16px">Memory Tables <span class="src-tag ours">calculated by ncu-view</span></div>
    <div class="rl-note">The NVIDIA Memory Workload Analysis tables (Shared Memory, L1/TEX Cache, L2 Cache, L2 Eviction Policies, Device Memory), computed from NVIDIA's own counters. Hover a cell for its formula and source counters. — = the counter was not exported for this access type.</div>
    ${tables.map(t=>`<div class="mem-table-block"><div class="mem-t-title">${esc(t.title)}</div>${memTable(t)}</div>`).join('')}</div>`;
}
function memTable(t){
  const head=t.cols.map(c=>`<th>${esc(c)}</th>`).join('');
  const rows=t.rows.map(r=>{
    const cells=t.cols.map(c=>{
      const cell=r.cells[c];
      if(!cell||cell.value==null)return '<td class="num">—</td>';
      const [nv,nu]=smart(cell.value,cell.unit);
      const tip=(cell.formula||'')+(cell.sources&&cell.sources.length?' — '+cell.sources.join(' + '):'');
      return `<td class="num" title="${esc(tip)}">${esc(nv)}${nu?` <span class="mcu">${esc(nu)}</span>`:''}</td>`;
    }).join('');
    return `<tr><td class="l">${esc(r.label)}</td>${cells}</tr>`;
  }).join('');
  return `<table class="data-table mem-table"><thead><tr><th></th>${head}</tr></thead><tbody>${rows}</tbody></table>`;
}
function barRows(items){
  const max=Math.max(...items.map(x=>x.v||0),0.0001);
  return items.map(x=>`<div class="bar-row"><span>${esc(x.l)}</span><div class="bar"><i style="width:${(x.v||0)/max*100}%"></i></div><b>${x.v!=null?fmt(x.v,x.v<10?2:1):'—'}</b></div>`).join('');
}
function schedBody(k,s){
  const r=l=>rowOf(k,'SchedulerStats',l);
  const bars=barRows([
    {l:'Theoretical Warps',v:8},
    {l:'Active Warps Per Scheduler',v:pct(r('Active Warps Per Scheduler'))},
    {l:'Eligible Warps Per Scheduler',v:pct(r('Eligible Warps Per Scheduler'))},
    {l:'Issued Warp Per Scheduler',v:pct(r('Issued Warp Per Scheduler'))},
  ]);
  return `${bars}
    <div class="callout" style="margin-top:10px"><strong>Key rule:</strong> each scheduler maintains a pool of warps. Active warps that are not stalled are eligible; the scheduler selects an eligible warp to issue. A high number of skipped issue slots indicates poor latency hiding. NVIDIA advises focusing on stall reasons when schedulers fail to issue every cycle.</div>
    ${tableHtml(s.rows||[])}`;
}
function warpBody(k,s){
  const reasons=(k.stats&&k.stats.stall_reasons)||[];
  const sum=reasons.reduce((a,r)=>a+r.cycles,0)||1;
  const bars=barRows(reasons.map(r=>({l:r.name,v:r.cycles/sum*100})));
  return `${bars}
    <div class="callout" style="margin-top:10px"><strong>Important:</strong> warp stalls are not automatically bad. The guide explicitly recommends focusing on stall reasons when schedulers fail to issue every cycle.</div>
    ${tableHtml(s.rows||[])}`;
}
function occBody(k,s){
  const r=l=>rowOf(k,'Occupancy',l);
  const g5=metric('Achieved Occupancy',pct(r('Achieved Occupancy')),'%',pct(r('Achieved Occupancy')))
    +metric('Theoretical Occupancy',pct(r('Theoretical Occupancy')),'%',pct(r('Theoretical Occupancy')))
    +metric('Theoretical Warps / SM',pct(r('Theoretical Active Warps per SM')),r('Theoretical Active Warps per SM')?r('Theoretical Active Warps per SM').unit:'')
    +metric('Achieved Warps / SM',pct(r('Achieved Active Warps Per SM')),r('Achieved Active Warps Per SM')?r('Achieved Active Warps Per SM').unit:'')
    +metric('Overall GPU Occupancy',pct(r('Overall GPU Occupancy')),'%',pct(r('Overall GPU Occupancy')));
  return `<div class="grid5">${g5}</div>
    <div class="callout" style="margin-top:12px"><strong>Occupancy definition:</strong> ratio of active warps per multiprocessor to the maximum possible active warps. Higher occupancy does not always produce higher performance, but low occupancy reduces the ability to hide latency. Large theoretical-versus-achieved discrepancies can indicate workload imbalance.</div>
    ${tableHtml(s.rows||[])}`;
}
function sourceBody(k,s){
  return `<div class="three">
      <div class="unit"><h4>SASS Instruction Mix</h4><p>Executed low-level assembly instruction statistics. The mix exposes pipeline dependencies and opportunities to use multiple pipelines for latency hiding.</p></div>
      <div class="unit"><h4>Source Hotspots</h4><p>Source Counters can provide N highest/lowest values of selected metrics at source locations, helping map performance data back to code.</p></div>
      <div class="unit"><h4>Warp Stall Sampling</h4><p>Periodically sampled warp stall reasons. Use with scheduler activity rather than treating every sampled stall as a performance problem.</p></div>
    </div>
    <div style="height:10px"></div>${tableHtml(s.rows||[])}`;
}
function samplingBody(k,s){
  const d=(k.derived||[]).filter(x=>x.name.indexOf('PM sampler')===0||x.name==='Warp-sampling period'||x.name.indexOf('Warp sampling')===0);
  const cfg=d.length?`<div class="chart-title">Sampling configuration (calculated by ncu-view)</div>
      ${d.map(x=>{const [nv,nu]=smart(x.value,x.unit||'');return `<div class="definition"><b>${esc(x.name)}</b><span>${nv} ${nu}<br>${esc(x.formula)}</span></div>`;}).join('')}
      <div class="rl-note" style="margin-top:6px">NVIDIA's timeline chart plots per-timestamp sample values stored inside the .ncu-rep; this export carries the configuration and the aggregate counters above, but not the sampled values themselves, so the timeline cannot be reproduced from the profile data.</div>`
    :`<div class="rl-note">No sampling counters in this export.</div>`;
  return `<div class="two">
      <div>
        <div class="callout"><strong>PM Sampling:</strong> performance monitors can be sampled periodically at fixed intervals. Samples carry a GPU timestamp and value, enabling a timeline view of how workload behavior changes over runtime.</div>
        <div class="definition"><b>pmsampling:</b> prefix<span>Explicitly requests PM-sampling collection.</span></div>
        <div class="definition"><b>Timeline</b><span>A section can request metrics in a Timeline field; the pmsampling prefix is recommended to avoid naming conflicts.</span></div>
        <div class="definition"><b>Volta and earlier</b><span>PM sampling is not supported.</span></div>
        <div class="definition"><b>TU10x–GA100</b><span>Supported with sampling intervals ≥ 20,000 cycles.</span></div>
        <div class="definition"><b>GA10x and later</b><span>Supported with intervals ≥ 1,000 ns.</span></div>
      </div>
      <div>${cfg}</div>
    </div>${tableHtml(s.rows||[])}`;
}
function launchBody(k,s){
  return (s.rows&&s.rows.length)
    ? `<div class="callout"><strong>Launch Statistics</strong> describe the grid, block and resource configuration of the launch.</div>${tableHtml(s.rows||[])}`
    : '<div class="unit"><h4>Launch Statistics</h4><p>No Launch Statistics section in this profile.</p></div>';
}
function instructionBody(k,s){
  return `<div class="three">
      <div class="unit"><h4>SASS Instruction Mix</h4><p>Executed low-level assembly instruction statistics.</p></div>
      <div class="unit"><h4>Instruction Types</h4><p>Frequency of instruction types by pipeline.</p></div>
      <div class="unit"><h4>Sampled Instructions</h4><p>Instruction-level sampling with source attribution.</p></div>
    </div><div style="height:10px"></div>${tableHtml(s.rows||[])}`;
}

// ---------- catalog ----------
function catalogCard(k){
  const SECTS=[
    ['C2CLink','C2C Link','Compute-to-compute link utilization and properties.'],
    ['ComputeWorkloadAnalysis','Compute Workload Analysis','SM compute resources, achieved IPC and pipeline utilization.'],
    ['InstructionStats','Instruction Statistics','SASS instruction mix, types and frequency.'],
    ['LaunchStats','Launch Statistics','Grid, block and GPU resource configuration.'],
    ['MemoryWorkloadAnalysis','Memory Workload Analysis','Memory utilization, bandwidth, pipes, chart and tables.'],
    ['NumaAffinity','NUMA Affinity','Compute/memory distance for GPUs.'],
    ['Nvlink','NVLink','High-level NVLink utilization.'],
    ['Nvlink_Tables','NVLink Tables','Per-link properties.'],
    ['Nvlink_Topology','NVLink Topology','Logical connection topology and throughput.'],
    ['Occupancy','Occupancy','Active-warps ratio and theoretical/achieved occupancy.'],
    ['PmSampling','PM Sampling','Periodic performance-monitor timeline.'],
    ['PmSampling_WarpStates','PM Sampling: Warp States','Periodically sampled warp states.'],
    ['SchedulerStats','Scheduler Statistics','Theoretical, active, eligible and issued warps.'],
    ['SourceCounters','Source Counters','Source metrics, branch efficiency and sampled warp stalls.'],
    ['SpeedOfLight','Speed Of Light','High-level compute and memory throughput vs theoretical maximum.'],
    ['SpeedOfLight_HierarchicalDoubleRooflineChart','Speed Of Light — Hierarchical Roofline (Double)','Hierarchical roofline chart for double-precision workloads.'],
    ['SpeedOfLight_HierarchicalHalfRooflineChart','Speed Of Light — Hierarchical Roofline (Half)','Hierarchical roofline chart for half-precision workloads.'],
    ['SpeedOfLight_HierarchicalSingleRooflineChart','Speed Of Light — Hierarchical Roofline (Single)','Hierarchical roofline chart for single-precision workloads.'],
    ['SpeedOfLight_HierarchicalTensorRooflineChart','Speed Of Light — Hierarchical Roofline (Tensor)','Hierarchical roofline chart for tensor-core workloads.'],
    ['SpeedOfLight_RooflineChart','Speed Of Light — Roofline','Classic roofline model for the whole kernel.'],
    ['Tile','Tile Statistics','Tile launch configuration, execution and resource usage.'],
    ['WarpStateStats','Warp State Statistics','Warp readiness, stalls and cycles per instruction.'],
    ['WorkloadDistribution','Workload Distribution','GPU and memory workload distribution.'],
  ];
  const items=SECTS.map(([id,title,desc])=>{
    const hit=(k.sections||[]).find(s=>SID(s.sid)===SID(id));
    const target=hit?hit.sid:id;
    const present=!!hit;
    return `<div class="section-item" data-scroll="sec-${esc(target)}"><h3>${esc(title)}${present?'<span class="present">profiled</span>':''}</h3><p>${esc(desc)}</p><div class="section-id">${esc(id)}</div></div>`;
  }).join('');
  return `<section class="card" id="sections"><div class="card-head"><span class="arrow">⌄</span><span class="title">All Nsight Compute Sections</span><span class="meta">Profiling Guide · Sections &amp; Rules</span></div><div class="body">
    <div class="callout"><strong>Sections present in this profile are rendered verbatim above</strong>; the rest are the canonical Nsight Compute sections from the Profiling Guide, for reference.</div>
    <div class="section-list">${items}</div></div></section>`;
}
function nvlinkCard(k){
  const has=((k.sections||[]).some(s=>SID(s.sid).startsWith('nvlink')));
  const tables=has?'':`<div class="callout" style="margin-top:9px">This profile has no NVLink sections — single-GPU workloads do not generate NVLink traffic.</div>`;
  return `<section class="card" id="nvlink"><div class="card-head"><span class="arrow">⌄</span><span class="title">NVLink / NUMA / Multi-GPU</span><span class="meta">Nvlink · Nvlink_Tables · Nvlink_Topology · NumaAffinity</span></div><div class="body">
    <div class="three">
      <div class="unit active"><h4>NVLink</h4><p>High-level utilization: total received and transmitted memory plus overall link peak utilization.</p></div>
      <div class="unit"><h4>NVLink Tables</h4><p>Detailed properties and measurements for individual NVLink connections.</p></div>
      <div class="unit"><h4>NVLink Topology</h4><p>Logical topology with transmit/receive throughput between connected endpoints.</p></div>
    </div>
    <div style="height:8px"></div>
    <div class="unit"><h4>NUMA Affinity</h4><p>Compute and memory distances for GPUs. Useful when understanding placement and data locality in multi-GPU systems.</p></div>${tables}</div></section>`;
}

// ---------- analysis panel ----------
function rlTables(k){
  const rl=k.roofline||{};
  const lv=rl.levels||[];
  const a=rl.achieved||{};
  if(!lv.length&&!a.fma)return '';
  let h='';
  if(lv.length){
    h+=`<div class="rl-t-title">Memory hierarchy — achieved vs peak (calculated by ncu-view)</div>
      <table class="data-table"><tbody>
      <tr><td class="l">level</td><td class="num">bytes moved</td><td class="num">achieved BW</td><td class="num">NVIDIA % of peak</td><td class="num">peak BW (derived)</td><td class="num">AI (FLOP/B)</td></tr>
      ${lv.map(x=>`<tr><td class="l">${esc(x.level)}</td><td class="num">${smart(x.bytes,'B')[0]} ${smart(x.bytes,'B')[1]}</td><td class="num">${smart(x.bw,'B/s')[0]} ${smart(x.bw,'B/s')[1]}</td><td class="num">${x.pct!=null?fmt(x.pct,2)+' %':'—'}</td><td class="num">${x.peak_bw?smart(x.peak_bw,'B/s')[0]+' '+smart(x.peak_bw,'B/s')[1]:'—'}</td><td class="num">${x.ai!=null?fmt(x.ai,3):'—'}</td></tr>`).join('')}
      </tbody></table>
      <div class="rl-note">${esc((lv[0]||{}).formula||'')}${lv[1]?` · ${esc(lv[1].formula)}`:''}${lv[2]?` · ${esc(lv[2].formula)}`:''} — NVIDIA %-of-peak is the level's own counter; peak BW = achieved ÷ (NVIDIA % / 100).</div>`;
  }
  const fma=a.fma||[];
  if(fma.length||(a.tensor||[]).length){
    h+=`<div class="rl-t-title">FLOP accounting by precision (calculated by ncu-view)</div>
      <table class="data-table"><tbody>
      <tr><td class="l">instruction class</td><td class="num">FLOPS</td><td class="num">FLOP/s (achieved)</td></tr>
      ${fma.map(x=>`<tr><td class="l">${esc(x.name)}</td><td class="num">${smart(x.flops,'FLOP')[0]} ${smart(x.flops,'FLOP')[1]}</td><td class="num">${x.flops_s!=null?smart(x.flops_s,'FLOP/s')[0]+' '+smart(x.flops_s,'FLOP/s')[1]:'—'}</td></tr>`).join('')}
      ${(a.tensor||[]).map(x=>`<tr><td class="l">${esc('Tensor '+x.name.replace('_dst_','→'))}</td><td class="num">${smart(x.flops,'FLOP')[0]} ${smart(x.flops,'FLOP')[1]}</td><td class="num">${x.flops_s!=null?smart(x.flops_s,'FLOP/s')[0]+' '+smart(x.flops_s,'FLOP/s')[1]:'—'}</td></tr>`).join('')}
      </tbody></table>`;
  }
  return `<div class="rl-tables">${h}</div>`;
}
function derivedHtml(k){
  const d=k.derived||[];
  if(!d.length)return '';
  let cur='',items='';
  for(const m of d){
    const g=m.group||'';
    if(g&&g!==cur){cur=g;items+=`<div class="dgroup">${esc(g)}</div>`;}
    const sv=smart(m.value,m.unit||'');
    items+=`<div class="dmetric" onclick="this.classList.toggle('open')" title="click for details">
      <div class="dname">${esc(m.name)}<span class="dtog">▸</span></div>
      <div class="dval">${esc(sv[0])}<span class="dunit"> ${esc(sv[1])}</span></div>
      ${m.desc?`<div class="ddesc">${esc(m.desc)}</div>`:''}
      <div class="ddetails">
        <div class="dform">${esc(m.formula||'')}</div>
        <div class="dsrc">Sources: ${esc((m.sources||[]).join(', '))}</div>
        ${m.note?`<div class="dnote">${esc(m.note)}</div>`:''}
      </div></div>`;
  }
  return `<section class="card" id="derived"><div class="card-head"><span class="arrow">⌄</span><span class="title">Derived metrics</span><span class="meta">ncu-view calculations<span class="src-tag ours">ours</span></span></div><div class="body">
    <div class="callout"><strong>Calculated by ncu-view, not by NVIDIA</strong> — every value below is computed from NVIDIA's own rows in this profile (click a metric for its formula and source rows; no assumed hardware specifications).</div>
    <div class="derived-grid">${items}</div>${rlTables(k)}</div></section>`;
}
function analysisPanel(k){
  const secs=k.sections||[];
  const body=secs.length?secs.map(s=>sectionCard(k,s)).join('')
    :`<div class="callout">No NVIDIA sections for this input — export them with <code>ncu --import &lt;rep&gt; --section &lt;X&gt; --csv</code> (see the README).</div>`;
  return derivedHtml(k)+body+nvlinkCard(k)+catalogCard(k)+metricRefHtml(k);
}

// ---------- metric reference cards (Profiling Guide §2.4 families) ----------
function metricRefHtml(k){
  const fams=k.metric_ref||[];
  if(!fams.length)return '';
  const v=val=>{
    if(val==null)return '—';
    const a=Math.abs(val);
    if(a>0&&a<0.01)return Number(val).toExponential(2);
    if(a>=1e4){const [nv,nu]=smart(val,'');return nv+nu;}
    return fmt(val,Number.isInteger(val)||val>=1000?0:2);
  };
  const cards=fams.map(f=>{
    const rows=f.rows.map(r=>{
      let val;
      if(!r.present)val='<span class="mref-absent">—</span>';
      else if(r.str!=null)val=`<span class="mref-str">${esc(r.str)}</span>`;
      else val=v(r.value)+(r.name.endsWith('_pct')?'%':'');
      return `<tr><td class="mref-name"><code>${esc(r.name)}</code></td><td>${esc(r.desc)}</td><td class="num">${val}</td></tr>`;
    }).join('');
    const note=f.present===0?`<div class="callout" style="margin-bottom:9px"><strong>Not collected in this profile</strong> — none of the ${f.total} metrics in this family were exported by the profile${f.note?' · '+esc(f.note):''}.</div>`
      :(f.note?`<div class="callout" style="margin-bottom:9px">${esc(f.note)}</div>`:'');
    return `<section class="card" id="sec-mr-${esc(f.sid)}" data-sid="mr-${esc(f.sid)}">
      <div class="card-head"><span class="arrow">⌄</span><span class="title">${esc(f.title)}</span>
      <span class="meta">§${esc(f.guide)} · ${f.present}/${f.total} collected<span class="src-tag">NVIDIA</span></span></div>
      <div class="body">
      <div class="callout"><strong>${esc(f.title)}</strong> — section ${esc(f.guide)} of the Profiling Guide. ${esc(f.intro)}</div>
      ${note}
      <table class="data-table"><thead><tr><th>Metric</th><th>Description</th><th style="text-align:right">Value</th></tr></thead>
      <tbody>${rows}</tbody></table></div></section>`;
  }).join('');
  return `<section class="card" id="metric-ref"><div class="card-head"><span class="arrow">⌄</span><span class="title">Metric reference</span>
    <span class="meta">Profiling Guide §2.4 metric families · values from this profile<span class="src-tag">NVIDIA</span></span></div><div class="body">
    <div class="callout"><strong>Reference cards for the metric families NVIDIA documents in the Profiling Guide</strong> — names and descriptions are NVIDIA's own; values are read from this profile's exported metrics. A metric the export does not carry shows '—'; nothing is invented.</div>
    ${cards}</div></section>`;
}

// ---------- doc panels (Profiling Guide synthesis) ----------
function docCard(title,meta,body){
  return `<section class="card"><div class="card-head"><span class="arrow">⌄</span><span class="title">${esc(title)}</span><span class="meta">${esc(meta)}</span></div><div class="body">${body}</div></section>`;
}
function docMetrics(){
  return docCard('Metric Structure','PerfWorks model',`
    <div class="callout"><strong>Nsight Compute metrics answer two questions:</strong> what happened (counters/metrics), and how close the workload got to peak hardware performance (throughput percentages). Peak rates are associated with counters in the metric database.</div>
    <div class="definition"><b>counter</b><span>Raw GPU counter or calculated counter value.</span></div>
    <div class="definition"><b>.sum / .avg / .min / .max</b><span>First-level roll-ups across unit instances.</span></div>
    <div class="definition"><b>.peak_sustained</b><span>Peak sustained rate associated with a counter roll-up.</span></div>
    <div class="definition"><b>.pct_of_peak_sustained</b><span>Percentage of sustained peak rate represented by the metric.</span></div>
    <div class="definition"><b>.per_cycle_active / elapsed</b><span>Rate normalized to active or elapsed cycles.</span></div>
    <div class="definition"><b>burst vs sustained</b><span>Burst is the maximum rate reportable in one cycle; sustained is the maximum rate achievable over an infinitely long period for typical operations.</span></div>`)
  +docCard('Metric Naming Convention','unit__subunit_pipestage_quantity_qualifiers',`
    <div class="code">Unit-Level Counter : unit__(subunit?)_(pipestage?)_quantity_(qualifiers?)
Interface Counter  : unit__(subunit?)_(pipestage?)_(interface)_quantity_(qualifiers?)
Unit Metric        : (counter_name).(rollup_metric)
Sub-Metric         : (counter_name).(rollup_metric).(submetric)

unit       = logical or physical GPU unit
subunit    = subunit / pipeline mode
pipestage  = pipeline stage
quantity   = what is measured
qualifiers  = predicates / filters
interface  = sender2receiver
rollup     = sum | avg | min | max

Example:
sm__inst_executed
sm__inst_executed.sum
sm__inst_executed.avg
l1tex__data_bank_conflicts_pipe_lsu.sum.pct_of_peak_sustained_active</div>`)
  +docCard('Metric Selection Rules','Query / sets / sections',`
    <div class="three">
      <div class="unit"><h4>Default / Basic</h4><p>A relatively small set of high-level utilization, launch and occupancy information.</p></div>
      <div class="unit"><h4>Section Sets</h4><p>Groups of logically associated metrics. More detail generally means more collection overhead.</p></div>
      <div class="unit"><h4>Full</h4><p>The full section set can be requested with <code>--set full</code>.</p></div>
    </div>
    <div style="height:10px"></div>
    <div class="code">ncu --list-sets
ncu --list-sections
ncu --query-metrics
ncu --set full ./application
ncu --section SpeedOfLight --section MemoryWorkloadAnalysis ./application</div>`);
}
function docCollection(){
  return docCard('Metric Collection & Replay','Kernel · Application · Range · Application-Range',`
    <div class="three">
      <div class="unit active"><h4>Kernel Replay</h4><p>Metrics for a kernel instance can be grouped into multiple passes. Memory written by the kernel is saved and restored between passes when needed.</p></div>
      <div class="unit"><h4>Application Replay</h4><p>The application is run multiple times, one pass per run. Requires deterministic kernel activity and launch matching.</p></div>
      <div class="unit"><h4>Range Replay</h4><p>Captures and replays complete CUDA API-call/kernel-launch ranges and supports workloads that must execute concurrently.</p></div>
    </div>
    <div style="height:8px"></div>
    <div class="unit"><h4>Application Range Replay</h4><p>Re-runs the application and collects complete selected ranges without associating metrics with individual kernels.</p></div>
    <div class="callout" style="margin-top:10px"><strong>Why replay exists:</strong> the GPU can collect only a limited number of hardware counters at once, and software-patched counters can have substantial overhead. Nsight Compute therefore groups requested metrics into passes.</div>`)
  +docCard('Collection Overhead, Cache & Clock Control','',`
    <div class="definition"><b>Hardware metrics</b><span>Generally low runtime overhead.</span></div>
    <div class="definition"><b>Launch/device attributes</b><span>Statically available; no kernel runtime overhead for collection.</span></div>
    <div class="definition"><b>Software-patched metrics</b><span>Highest overhead because instructions are modified to execute additional code; collected in separate passes.</span></div>
    <div class="definition"><b>Cache control</b><span>By default, GPU caches are flushed before replay passes to make hardware-counter values more deterministic. <code>--cache-control none</code> can disable this.</span></div>
    <div class="definition"><b>Clock control</b><span>Profiling consistency can depend on GPU clock behavior. Mobile/partitioned environments can have limitations.</span></div>
    <div class="definition"><b>Persistence mode</b><span>Recommended on applicable systems for more consistent behavior around GPU initialization.</span></div>`)
  +docCard('Ranges / NVTX','Range Replay',`
    <div class="callout"><strong>Range markers:</strong> <code>cu(da)ProfilerStart/Stop</code> can define a profiling range; NVTX include expressions can also select ranges. Ranges should be as narrow as practical and must satisfy synchronization and supported-API constraints.</div>
    <div class="code">ncu --replay-mode range --nvtx --nvtx-include "MyRange" ./my_app

# Example range marker:
cuProfilerStart();
run_workload();
cuProfilerStop();</div>`)
  +docCard('Profile Series & Metric Distributor','',`
    <div class="two">
      <div class="callout"><strong>Profile Series:</strong> automatically profiles a kernel repeatedly while changing selected launch parameters. Each parameter combination produces a profile, making it easier to identify the best configuration.</div>
      <div class="callout"><strong>Metric Distributor:</strong> divides metric collection across multiple NCU processes/GPUs to reduce total passes. Participating GPUs must have identical chip architecture; partial reports are merged afterward.</div>
    </div>
    <div class="code">ncu --set full --metric-distribution-groups 4 --communicator none -o report ./app
$NCU_INSTALL_PATH/extras/ReportUtils/ReportMergeTool -i &lt;report directory&gt; -o final_report</div>`);
}
function docReference(){
  return docCard('Special Configurations','MIG · MPS · Graphs · Multi-GPU',`
    <div class="three">
      <div class="unit"><h4>MIG</h4><p>Partitions a GPU into GPU Instances and Compute Instances. Compute Instances have exclusive SM ownership while instances within a GPU Instance share its memory/bandwidth.</p></div>
      <div class="unit"><h4>MPS</h4><p>Nsight Compute supports MPS profiling through CLI. Range replay is recommended where possible; observation-window and client-homogeneity constraints apply.</p></div>
      <div class="unit"><h4>Graphs</h4><p>Graph profiling has metric limitations; some instruction-level source metrics are unavailable when graph profiling is enabled.</p></div>
    </div>`)
  +docCard('Workload / Metric Compatibility','',`
    <table class="data-table"><thead><tr><th>Workload</th><th>Kernel replay</th><th>Application</th><th>Range</th><th>HW/SMSP</th><th>Unit source</th><th>Instruction source</th></tr></thead><tbody>
    <tr><td>Kernel</td><td>Yes</td><td>Yes</td><td>Yes</td><td>Yes</td><td>Yes</td><td>Yes</td></tr>
    <tr><td>Range</td><td>No</td><td>No</td><td>Yes</td><td>Yes</td><td>No</td><td>Yes</td></tr>
    <tr><td>Cmdlist</td><td>Yes</td><td>No</td><td>No</td><td>Yes</td><td>Yes</td><td>Yes</td></tr>
    <tr><td>Graph</td><td>Yes</td><td>No</td><td>No</td><td>Yes</td><td>No</td><td>No</td></tr>
    </tbody></table>`)
  +docCard('FAQ / Troubleshooting','Profiling Guide',`
    <div class="faq"><h3>Why is a metric n/a?</h3><p>It means the metric is not available. Common causes include a typo, missing suffix, old metric name, zero-width Unicode characters, or the metric not existing on the target GPU architecture. Verify names with <code>--query-metrics</code>.</p></div>
    <div class="faq"><h3>Why can a percentage exceed 100%?</h3><p>Metric ranges and precision rules can permit values outside an intuitive logical range; consult the metric's defined peak/rate semantics.</p></div>
    <div class="faq"><h3>What is ERR_NVGPUCTRPERM?</h3><p>The process does not have permission to access NVIDIA GPU performance counters. Depending on the platform, profiling may require elevated permissions or non-admin profiling configuration. On WSL, GPU performance-counter access must be enabled in the NVIDIA Control Panel on the Windows host.</p></div>
    <div class="faq"><h3>What does unsupported GPU mean?</h3><p>The GPU or current GPU configuration is not supported by the Nsight Compute version being used. Check the corresponding release notes and supported-device list.</p></div>
    <div class="faq"><h3>Why do replay passes behave differently?</h3><p>Hardware caches cannot be saved/restored like device memory. Nsight Compute normally flushes GPU caches before replay passes; disabling cache control can make results more representative of a larger application context but less isolated.</p></div>`);
}

// ---------- page assembly ----------
function pageHead(k){
  const d=DATA.meta.device||{};
  const bits=[d.name,d.cc?('CC '+d.cc):null,d.sm_count?fmt(d.sm_count.v,0)+' SMs':null].filter(x=>x);
  return `<div class="page-head">
    <h1>${knameHtml(k.name)}</h1>
    <span class="sub">${esc(bits.join(' · '))}${DATA.meta.input?'<span style="color:var(--dim)"> · ' + esc(DATA.meta.input) + '</span>':''}</span>
    <span class="right">Profiling Guide reference: <span class="source-link" onclick="window.open('https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html','_blank')">official NVIDIA documentation</span></span>
  </div>`;
}
function kernelPage(k){
  return `${pageHead(k)}${kpis(k)}${heroGrid(k)}${tabsHtml()}
    <div class="tab-panel active" id="panel-analysis">${analysisPanel(k)}</div>
    <div class="tab-panel" id="panel-metrics">${docMetrics()}</div>
    <div class="tab-panel" id="panel-collection">${docCollection()}</div>
    <div class="tab-panel" id="panel-reference">${docReference()}</div>
    <div class="footer">Built as a black/white Nsight Compute-style dashboard. Every number is NVIDIA's own exported data from the profile — section rows, rule results, counters, device attributes. The documentation panels are a structured synthesis of NVIDIA's Profiling Guide, not a replacement for the official manual.
      <br><br><span class="source-link" onclick="window.open('https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html','_blank')">Open NVIDIA Nsight Compute Profiling Guide →</span></div>`;
}
function summaryPage(){
  const series=DATA.series||[];
  const multiSeries = series.length > 1;
  const best=series.slice().sort((a,b)=>(a.time_us??Infinity)-(b.time_us??Infinity))[0];
  let rows='';
  series.forEach(r=>{
    const k=DATA.kernels.find(x=>x.key===r.key);
    const v=k&&k.verdict;
    rows+=`<tr>${multiSeries?`<td class="l">${knameHtml(r.name)}</td>`:''}<td class="num">${r.time_us!=null?smart(r.time_us,'µs')[0]+' '+smart(r.time_us,'µs')[1]:'—'}</td>
      <td class="num">${r.pipe_pct!=null?fmt(r.pipe_pct,1):'—'}</td>
      <td class="num">${r.dram_pct!=null?fmt(r.dram_pct,1):'—'}</td>
      <td class="num">${r.occupancy_pct!=null?fmt(r.occupancy_pct,1):'—'}</td>
      <td class="num">${r.stall_cycles!=null?fmt(r.stall_cycles,2):'—'}</td>
      <td>${v?esc(v.name):'—'}</td></tr>`;
  });
  const sevColor=s=>s==='critical'?'var(--bad)':s==='warning'?'var(--warn)':'var(--text)';
  const recs=DATA.kernels.map(k=>k.rules.map(r=>({...r}))).flat()
    .sort(estSort);
  const recHtml=recs.map((r,i)=>`<div class="rec"><div class="recno">${i+1}</div>
      <div class="rectext"><span class="rec-name">${esc(r.name)}</span></div>
      <div class="impact">${r.est?'est. '+esc(r.est):SEVNAME[r.severity]||r.severity}</div></div>
      <div class="recmsg">${esc(r.message)}${focusHtml(r)}</div>`).join('');
  return `<div class="page-head"><h1>Summary</h1>
      <span class="sub">${DATA.meta.input?esc(DATA.meta.input):''}</span></div>
    <div class="kpis" style="grid-template-columns:repeat(3,1fr)">
      <div class="kpi"><div class="value">${DATA.kernels.length}</div><div class="label">Kernels in Report</div></div>
      <div class="kpi"><div class="value" title="${best&&best.name&&multiSeries?esc(best.name):''}">${best&&best.name&&multiSeries?esc(kshort(best.name)):'—'}</div><div class="label">Best kernel</div></div>
      <div class="kpi"><div class="value">${best&&best.time_us!=null?smart(best.time_us,'µs')[0]+' '+smart(best.time_us,'µs')[1]:'—'}</div><div class="label">Best duration</div></div>
    </div>
    <section class="card"><div class="card-head"><span class="arrow">⌄</span><span class="title">Prioritized recommendations (all kernels)</span><span class="meta">NVIDIA rule engine</span></div><div class="body">
      ${recHtml||'<div class="copy">No recommendations.</div>'}</div></section>
    <section class="card"><div class="card-head"><span class="arrow">⌄</span><span class="title">Kernel series</span><span class="meta">all kernels</span></div><div class="body">
      <table class="data-table"><thead><tr>${multiSeries?'<th>Kernel</th>':''}<th style="text-align:right">Duration</th><th style="text-align:right">Tensor pipe %</th><th style="text-align:right">DRAM %</th><th style="text-align:right">Occupancy %</th><th style="text-align:right">Stall/issue</th><th>Verdict</th></tr></thead><tbody>${rows}</tbody></table>
    </div></section>
    <div class="footer">Built as a black/white Nsight Compute-style dashboard. Every number is NVIDIA's own exported data from the profile.
      <br><br><span class="source-link" onclick="window.open('https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html','_blank')">Open NVIDIA Nsight Compute Profiling Guide →</span></div>`;
}

// ---------- sidebar ----------
function applyFilter(){
  const list=DATA.kernels.filter(k=>!searchQ||k.name.toLowerCase().includes(searchQ.toLowerCase()));
  const kn=list.map((k,i)=>{
    const t=k.stats.time_us!=null?fmt(k.stats.time_us/1000,2)+' ms':'';
    return `<div class="kernel" data-key="${esc(k.key)}" data-view="kernel"><span class="r">${i+1}</span><span class="n">${knameHtml(k.name)}</span><span class="t">${esc(t)}</span></div>`;}).join('');
  const k0=DATA.kernels[0];
  const secs=(k0?(k0.sections||[]):[]).map(s=>`<div class="nav-item" data-scroll="sec-${esc(s.sid)}"><span class="nav-icon">▸</span>${esc(s.title)}</div>`).join('');
  const mrs=(k0?(k0.metric_ref||[]):[]).map(f=>`<div class="nav-item" data-scroll="sec-mr-${esc(f.sid)}"><span class="nav-icon">▸</span>${esc(f.title)}<span style="margin-left:auto;color:var(--dim);font-size:8px">${esc(f.guide)}</span></div>`).join('');
  const sb=$('#sidebar');
  if(!sb)return;
  sb.innerHTML='<div class="side-label">REPORT</div>'
    +`<div class="nav-item" data-scroll="top"><span class="nav-icon">▦</span>Overview</div>`
    +`<div class="nav-item" data-view="summary" data-key="__summary__"><span class="nav-icon">▤</span>Summary<span class="badge-count">${DATA.kernels.length}</span></div>`
    +`<div class="nav-item" data-tab="analysis"><span class="nav-icon">◈</span>Analysis</div>`
    +`<div class="nav-item" data-tab="metrics"><span class="nav-icon">◇</span>Metrics</div>`
    +`<div class="nav-item" data-tab="collection"><span class="nav-icon">◫</span>Collection &amp; Replay</div>`
    +`<div class="sep"></div><div class="side-label">KERNEL NAVIGATION</div>`
    +`<div class="side-search">⌕ <input id="ks" placeholder="Search kernels..."></div>`
    +`<div id="kernels">${kn}</div>`
    +`<div class="nav-item" data-scroll="top" style="margin-top:3px"><span class="nav-icon">▦</span>View all ${DATA.kernels.length} kernels</div>`
    +`<div class="sep"></div><div class="side-label">NCU-VIEW CALCULATIONS</div>`
    +`<div class="nav-item" data-scroll="derived"><span class="nav-icon">✦</span>Derived metrics<span class="src-tag ours" style="margin-left:auto">ours</span></div>`
    +`<div class="sep"></div><div class="side-label">METRIC REFERENCE</div>`
    +mrs
    +`<div class="sep"></div><div class="side-label">NSIGHT COMPUTE SECTIONS</div>`
    +`<div class="nav-item" data-scroll="sections"><span class="nav-icon">▤</span>All Sections</div>`
    +secs;
  const ks=$('#ks');
  if(ks)ks.addEventListener('input',e=>{searchQ=ks.value.trim();applyFilter();});
  $$('#sidebar .nav-item,#sidebar .kernel').forEach(el=>el.addEventListener('click',()=>{
    $$('#sidebar .nav-item,#sidebar .kernel').forEach(x=>x.classList.remove('active'));
    el.classList.add('active');
    if(el.dataset.view==='summary'){curView='summary';pushHash('#s');renderMain();}
    else if(el.dataset.tab){curView='kernel';showTab(el.dataset.tab);}
    else if(el.dataset.scroll){
      curView='kernel';showTab('analysis');
      const id=el.dataset.scroll;
      const t=id==='top'?$('#main .page-head'):$('#'+id);
      if(t)t.scrollIntoView({behavior:'smooth',block:'start'});
      if(id==='top'&&window.scrollTo)window.scrollTo({top:0,behavior:'smooth'});
    }
    else{curView='kernel';curKernel=el.dataset.key;pushHash('#k-'+el.dataset.key);renderMain();}
  }));
  markActive();
  const kernelEls=$$('#sidebar .kernel');
  kernelEls.forEach(el=>el.addEventListener('click',()=>{
    kernelEls.forEach(x=>x.classList.remove('active'));el.classList.add('active');
  }));
}
function sidebar(){applyFilter();}
function markActive(){
  $$('#sidebar .nav-item,#sidebar .kernel').forEach(x=>x.classList.remove('active'));
  if(curView==='summary'){const el=$('#sidebar .nav-item[data-view="summary"]');if(el)el.classList.add('active');return;}
  const el=$('#sidebar .kernel[data-key="'+curKernel+'"]');
  if(el)el.classList.add('active');
}

// ---------- interactions ----------
function bindCollapse(){
  $$('.card-head').forEach(h=>h.addEventListener('click',()=>{
    const card=h.parentElement;if(!card)return;
    const sid=card.dataset.sid;
    const on=card.classList.toggle('collapsed');
    if(sid){try{const saved=JSON.parse(store.get('ncu-view-collapsed')||'[]');const set=new Set(saved);
      on?set.add(sid):set.delete(sid);store.set('ncu-view-collapsed',JSON.stringify([...set]));}catch(e){}}
  }));
}
function bindRecs(){
  $$('#main .rec').forEach(el=>el.addEventListener('click',()=>{
    const msg=el.nextElementSibling;
    if(msg&&msg.classList.contains('recmsg'))msg.classList.toggle('open');
  }));
}
function restoreCollapsed(){
  let saved=[];try{saved=JSON.parse(store.get('ncu-view-collapsed')||'[]');}catch(e){}
  saved.forEach(sid=>{const el=$('#sec-'+sid);if(el)el.classList.add('collapsed');});
}
function renderMain(){
  const main=$('#main');
  if(!main)return;
  if(curView==='summary'){main.innerHTML=summaryPage();bindCollapse();bindRecs();markActive();return;}
  const k=DATA.kernels.find(x=>x.key===curKernel);
  if(!k)return;
  main.innerHTML=kernelPage(k);
  bindCollapse();bindRecs();bindTabs();bindScrollLinks();spy();markActive();
}
function bindScrollLinks(){
  $$('#main [data-scroll]').forEach(el=>el.addEventListener('click',()=>{
    const t=$('#'+el.dataset.scroll);if(t)t.scrollIntoView({behavior:'smooth',block:'start'});
  }));
}
function spy(){
  const secs=$$('#main .card[id]');
  if(!secs.length||typeof IntersectionObserver==='undefined')return;
  const io=new IntersectionObserver(es=>{es.forEach(e=>{
    if(!e.isIntersecting)return;
    const sid=e.target.id;
    $$('#sidebar .nav-item[data-scroll]').forEach(x=>x.classList.toggle('active',x.dataset.scroll===sid));});},
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
  if(h.startsWith('#sec-')){curView='kernel';renderMain();showTab('analysis');
    setTimeout(()=>{const t=$('#'+h.slice(1));if(t)t.scrollIntoView({behavior:'smooth'});},30);return;}
  const k=DATA.kernels.find(x=>x.key===h.slice(3));
  if(k){curView='kernel';curKernel=k.key;renderMain();}
}
function keynav(e){
  if(e.target&&(e.target.tagName==='INPUT'||e.target.tagName==='TEXTAREA'))return;
  const idx=DATA.kernels.findIndex(x=>x.key===curKernel);
  if(e.key==='ArrowRight'&&idx<DATA.kernels.length-1){curView='kernel';curKernel=DATA.kernels[idx+1].key;pushHash('#k-'+curKernel);renderMain();}
  else if(e.key==='ArrowLeft'&&idx>0){curView='kernel';curKernel=DATA.kernels[idx-1].key;pushHash('#k-'+curKernel);renderMain();}
  else if(e.key==='s'||e.key==='S'){curView='summary';pushHash('#s');renderMain();}
  else if((e.ctrlKey||e.metaKey)&&e.key.toLowerCase()==='k'){const gs=$('#global');if(gs){e.preventDefault();gs.focus();}}
  else if(e.key==='/'&&e.target!==$('#ks')){const ks=$('#ks');if(ks){ks.focus();e.preventDefault();}}
}
function exportData(){
  try{
    const blob=new Blob([JSON.stringify(DATA,null,1)],{type:'application/json'});
    const a=document.createElement('a');a.href=URL.createObjectURL(blob);
    a.download='ncu-view-report.json';document.body.appendChild(a);a.click();a.remove();
  }catch(e){}
}
function copyKernel(){
  const k=DATA.kernels.find(x=>x.key===curKernel);
  if(!k)return;
  const s=k.stats||{};const v=k.verdict;
  copyText(`kernel: ${k.name}\nduration: ${s.time_us!=null?s.time_us+' µs':''}\nverdict: ${v?v.name:'—'}\ntensor pipe: ${s.pipe_pct!=null?s.pipe_pct+'%':''}\ndram: ${s.dram_pct!=null?s.dram_pct+'%':''}\noccupancy: ${s.occupancy_pct!=null?s.occupancy_pct+'%':''}`);
}

// ---------- search palette ----------
let SEARCH=[];
function buildSearchIndex(){
  SEARCH=[];
  DATA.kernels.forEach(k=>{
    SEARCH.push({kind:'k',label:k.name,
      sub:((k.verdict&&k.verdict.name)||'')+(k.stats&&k.stats.time_us!=null?' · '+fmt(k.stats.time_us,1)+' µs':''),
      key:k.key});
    (k.sections||[]).forEach(s=>{
      SEARCH.push({kind:'s',label:s.title,sub:k.name,key:k.key,sid:s.sid});
      (s.rows||[]).forEach(r=>{
        if(r.label&&String(r.label).trim())
          SEARCH.push({kind:'r',label:String(r.label),sub:s.title+' · '+k.name,key:k.key,sid:s.sid});
      });
    });
    (k.derived||[]).forEach((d,i)=>{
      SEARCH.push({kind:'d',label:d.name,
        sub:(d.desc||'')+(d.group?' · '+d.group:''),key:k.key,tile:i});
    });
    (k.metric_ref||[]).forEach(f=>{
      SEARCH.push({kind:'s',label:f.title+' (metric reference)',
        sub:'§'+f.guide+' · '+f.present+'/'+f.total+' collected · '+k.name,key:k.key,sid:'mr-'+f.sid});
      f.rows.forEach(r=>{
        SEARCH.push({kind:'r',label:r.name,sub:f.title+' · '+k.name,key:k.key,sid:'mr-'+f.sid});
      });
    });
  });
  const tmp=document.createElement('div');
  tmp.innerHTML=summaryPage()+docMetrics()+docCollection()+docReference();
  tmp.querySelectorAll('.definition b,.unit h4,.faq h3').forEach(el=>{
    const label=el.textContent.replace(/\s+/g,' ').trim();
    if(!label)return;
    const sub=el.parentElement?el.parentElement.textContent.replace(/\s+/g,' ').trim().slice(0,110):'';
    SEARCH.push({kind:'c',label,sub});
  });
}
function searchPopHtml(q){
  const groups=[['k','Kernels'],['s','Sections'],['r','Metrics'],['d','Derived'],['c','Concepts']];
  const badge={k:'K',s:'S',r:'M',d:'D',c:'C'};
  let out='';
  groups.forEach(([kind,gname])=>{
    const items=SEARCH.filter(x=>x.kind===kind&&(x.label+' '+(x.sub||'')).toLowerCase().includes(q)).slice(0,6);
    if(!items.length)return;
    out+=`<div class="search-grp">${gname}</div>`+items.map(it=>
      `<div class="search-row" data-kind="${it.kind}" data-key="${esc(it.key||'')}" data-sid="${esc(it.sid||'')}" data-tile="${it.tile!=null?it.tile:''}"><span class="k">${badge[it.kind]}</span><div class="tx"><b>${esc(it.label)}</b>${it.sub?`<i>${esc(it.sub)}</i>`:''}</div></div>`
    ).join('');
  });
  return out||`<div class="search-empty">No matches for “${esc(q)}”</div>`;
}
function searchMark(i){
  const pop=$('#search-pop');
  if(!pop)return;
  const rows=$$('.search-row',pop);
  rows.forEach((r,j)=>r.classList.toggle('active',j===i));
  if(rows[i])rows[i].scrollIntoView({block:'nearest'});
}
function searchMove(d){
  const pop=$('#search-pop');
  if(!pop)return;
  const rows=$$('.search-row',pop);
  if(!rows.length)return;
  const cur=rows.findIndex(r=>r.classList.contains('active'));
  const next=(cur<0?0:(cur+d+rows.length)%rows.length);
  searchMark(next);
}
function searchGo(row){
  const kind=row.dataset.kind,key=row.dataset.key,sid=row.dataset.sid,tile=row.dataset.tile;
  const gs=$('#global');if(gs)gs.value='';
  const pop=$('#search-pop');if(pop)pop.style.display='none';
  const ensureKernel=()=>{
    if(curKernel!==key){curView='kernel';curKernel=key;if(key)pushHash('#k-'+key);renderMain();}
    if(curView==='summary'){curView='kernel';if(key)pushHash('#k-'+key);renderMain();}
    showTab('analysis');
  };
  const gotoSection=()=>{
    ensureKernel();
    const el=document.getElementById('sec-'+sid);
    if(el){el.classList.remove('collapsed');flash(el);el.scrollIntoView({behavior:'smooth',block:'start'});}
  };
  if(kind==='k'){ensureKernel();const t=$('#main .page-head');if(t){flash(t);t.scrollIntoView({behavior:'smooth',block:'start'});}}
  else if(kind==='s'||kind==='r')gotoSection();
  else if(kind==='d'){
    ensureKernel();
    const card=document.getElementById('derived');
    if(card)card.classList.remove('collapsed');
    const tiles=$$('#derived .dmetric');
    const el=tiles[parseInt(tile,10)];
    if(el){el.scrollIntoView({behavior:'smooth',block:'center'});flash(el);}
    else if(card){card.scrollIntoView({behavior:'smooth',block:'start'});flash(card);}
  }
  else if(kind==='c'){
    ensureKernel();
    const label=row.querySelector('b')?row.querySelector('b').textContent:'';
    const el=$$('#main .definition b,#main .unit h4,#main .faq h3').find(x=>x.textContent.trim()===label);
    if(el){
      const panel=el.closest('.tab-panel');
      if(panel)showTab(panel.id.replace('panel-',''));
      const p=el.parentElement;
      flash(p);p.scrollIntoView({behavior:'smooth',block:'center'});
    }
  }
}
function flash(el){
  el.style.outline='1px solid var(--accent)';
  setTimeout(()=>{el.style.outline='';},1100);
}
function bindSearch(){
  const gs=$('#global');if(!gs)return;
  const pop=$('#search-pop');
  let timer=null;
  const openPop=()=>{if(pop)pop.style.display='block';};
  const render=()=>{
    const q=gs.value.toLowerCase().trim();
    if(!q){if(pop)pop.style.display='none';return;}
    if(pop){pop.innerHTML=searchPopHtml(q);openPop();searchMark(0);}
  };
  gs.addEventListener('input',()=>{clearTimeout(timer);timer=setTimeout(render,120);});
  gs.addEventListener('focus',render);
  gs.addEventListener('keydown',e=>{
    if(e.key==='ArrowDown'||e.key==='ArrowUp'){
      if(pop&&pop.style.display==='block'){e.preventDefault();searchMove(e.key==='ArrowDown'?1:-1);}
    }
    else if(e.key==='Enter'){
      e.preventDefault();
      const rows=pop?$$('.search-row',pop):[];
      const act=rows.find(r=>r.classList.contains('active'))||rows[0];
      if(act)searchGo(act);
    }
    else if(e.key==='Escape'){if(pop)pop.style.display='none';gs.blur();}
  });
  if(pop)pop.addEventListener('click',e=>{
    const row=e.target.closest?e.target.closest('.search-row'):null;
    if(row)searchGo(row);
  });
  document.addEventListener('click',e=>{
    if(!(e.target.closest&&e.target.closest('.global-search'))){if(pop)pop.style.display='none';}
  });
}
document.addEventListener('DOMContentLoaded',()=>{
  themeInit();
  const count=$('#app-count');if(count)count.textContent=DATA.kernels.length+' kernels';
  const tty=$('#tty-status');
  if(tty){
    const d=DATA.meta.device||{};
    const secCount=DATA.kernels.reduce((n,k)=>n+(k.sections||[]).length,0);
    const parts=[d.name?`<b>${esc(d.name)}</b>`:null,
      d.cc?`CC ${esc(d.cc)}`:`cc ${esc(d.cc||'?')}`,
      `${DATA.kernels.length} kernel${DATA.kernels.length===1?'':'s'}`,
      `${secCount} sections`].filter(x=>x);
    tty.innerHTML='[ '+parts.join(' │ ')+' ]';
  }
  const themeBtn=$('#theme-toggle');if(themeBtn)themeBtn.addEventListener('click',()=>{
    setTheme(document.documentElement.getAttribute('data-theme')==='light'?'dark':'light');});
  const copyBtn=$('#copy-btn');if(copyBtn)copyBtn.addEventListener('click',copyKernel);
  const exportBtn=$('#export-btn');if(exportBtn)exportBtn.addEventListener('click',exportData);
  bindSearch();
  if(typeof window!=='undefined')window.addEventListener('hashchange',applyHash);
  sidebar();renderMain();restoreCollapsed();applyHash();
  buildSearchIndex();
  document.addEventListener('keydown',keynav);
});
"""


def render_html(report: dict) -> str:
    payload = __import__("json").dumps(report, sort_keys=True)
    js = ICONS_JS.replace("__DATA__", payload)
    title = "ncu-view"
    if report.get("meta", {}).get("input"):
        title += " — " + report["meta"]["input"]
    return f"""<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<style>{REFERENCE_CSS}</style>
</head>
<body>

<header>
  <div class="brand"><div class="eye"></div><div class="logo">NVIDIA</div><div class="vline"></div><div class="brand-title" id="app-title">Nsight Compute — Profiling Dashboard</div></div>
  <div class="tty-status" id="tty-status"></div>
  <div class="header-spacer"></div>
  <div class="global-search">⌕ <input id="global" placeholder="Search sections, metrics, concepts..."><kbd>⌘K</kbd><div class="search-pop" id="search-pop"></div></div>
  <button class="hbtn" id="copy-btn">▣ Copy</button><button class="hbtn" id="export-btn">⇩ Export</button><button class="hbtn" id="theme-toggle">☼</button>
</header>

<aside id="sidebar"></aside>

<main>
<div class="content" id="main"></div>
</main>

<script>{js}</script>
</body>
</html>"""
