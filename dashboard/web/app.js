/* Paper Book — plain JS, no framework, no dependencies.
   One poll of /api/status (tiny, cached server-side) every 30 s in market
   hours and 5 min otherwise; page data is re-fetched only when the status
   version changes. Everything renders from strings into #page. */
'use strict';

// ── helpers ──────────────────────────────────────────────────────────────────
const $ = (s, r = document) => r.querySelector(s);
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const TZ = 'Asia/Kolkata';
const store = {
  get(k, d) { try { const v = localStorage.getItem(k); return v == null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch { /* private mode */ } },
};

// Formatters (from the design handoff): Indian grouping, true minus, ▲/▼ never colour alone.
const g = (n, dp) => Math.abs(n).toLocaleString('en-IN', { minimumFractionDigits: dp, maximumFractionDigits: dp });
const sgn = (n, dp) => { const r = +(+n).toFixed(dp); return r > 0 ? '+' : r < 0 ? '−' : ''; };
const rs = (n, dp = 2) => (n < 0 && +Math.abs(n).toFixed(dp) > 0 ? '−' : '') + '₹' + g(n, dp);
const srs = (n, dp = 2) => sgn(n, dp) + '₹' + g(n, dp);
const spct = (f, dp = 2) => sgn(f * 100, dp) + g(f * 100, dp) + '%';
const arr = (f, dp = 2) => { const r = +(f * 100).toFixed(dp); return (r > 0 ? '▲ ' : r < 0 ? '▼ ' : '') + g(f * 100, dp) + '%'; };
const tone = n => n > 0.004 ? 'up' : n < -0.004 ? 'down' : 'flat';
const toDate = s => s && s.length === 10 ? new Date(s + 'T00:00:00+05:30') : new Date(s);
const hm = s => toDate(s).toLocaleTimeString('en-GB', { timeZone: TZ, hour: '2-digit', minute: '2-digit' });
const hms = s => toDate(s).toLocaleTimeString('en-GB', { timeZone: TZ, hour: '2-digit', minute: '2-digit', second: '2-digit' });
const dm = s => toDate(s).toLocaleDateString('en-GB', { timeZone: TZ, day: 'numeric', month: 'short' });
const wdm = s => toDate(s).toLocaleDateString('en-GB', { timeZone: TZ, weekday: 'short', day: 'numeric', month: 'short' });
const istDay = d => d.toLocaleDateString('en-CA', { timeZone: TZ });   // YYYY-MM-DD
const minutesIST = s => { const [h, m] = hm(s).split(':').map(Number); return h * 60 + m; };
const ago = s => { const m = Math.round((Date.now() - toDate(s)) / 60000); return m < 1 ? 'just now' : m < 60 ? `${m} min ago` : m < 1440 ? `${Math.round(m / 60)} h ago` : `${Math.round(m / 1440)} d ago`; };
const short = t => String(t || '').replace(/\.NS$/, '');
const titleCase = s => s ? s[0].toUpperCase() + s.slice(1) : s;

async function getJSON(url) {
  const r = await fetch(url, { cache: 'no-store' });
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  return r.json();
}
async function postJSON(url, body) {
  const r = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json', 'X-Paper-Book': '1' }, body: JSON.stringify(body || {}) });
  return r.json();
}

// ── state ────────────────────────────────────────────────────────────────────
const ui = store.get('pb-ui', { sk: 'value', sd: -1, ps: 'value', range: '1D', open: {}, overlays: false, doc: null });
const saveUI = () => store.set('pb-ui', ui);
let STATUS = null, LOGIN_URL = null, FLASH = null, OFFLINE = false;
const DATA = {}, VER = {};

const PAGES = {
  portfolio: { title: 'Portfolio', api: '/api/portfolio', render: pagePortfolio },
  experiment: { title: 'Experiment', api: '/api/experiment', render: pageExperiment },
  health: { title: 'System health', api: '/api/health', render: pageHealth },
  research: { title: 'Research log', api: '/api/research', render: pageResearch },
  warmup: { title: 'Warm-up', api: '/api/warmup', render: pageWarmup },
  kite: { title: 'Kite login', api: null, render: pageKite },
};
const TABS = ['portfolio', 'experiment', 'health', 'research', 'warmup'];
const BNAV = [['portfolio', 'Portfolio'], ['kite', 'Kite'], ['experiment', 'Experiment'], ['health', 'Health'], ['research', 'Research']];

function route() {
  const h = location.hash.replace(/^#/, '');
  const [name, qs] = h.split('?');
  return { name: PAGES[name] ? name : 'portfolio', params: new URLSearchParams(qs || '') };
}

// ── theme ────────────────────────────────────────────────────────────────────
function applyTheme() {
  const t = store.get('pb-theme', 'auto');
  if (t === 'auto') delete document.documentElement.dataset.pt; else document.documentElement.dataset.pt = t;
  return t;
}
applyTheme();

// ── chrome: tabs, chips, bottom nav ─────────────────────────────────────────
function renderChrome() {
  const r = route().name;
  $('#tabs').innerHTML = TABS.map(k => `<a href="#${k}" class="${k === r ? 'on' : ''}">${PAGES[k].title}</a>`).join('');
  $('#bnav').innerHTML = BNAV.map(([k, l]) => `<a href="#${k}" class="${k === r ? 'on' : ''}"><i></i>${l}</a>`).join('');
  const s = STATUS;
  const theme = store.get('pb-theme', 'auto');
  const themeIcon = { auto: '◐', light: '☀', dark: '☾' }[theme];
  if (!s) { $('#chips').innerHTML = `<span class="chip"><span class="dot ${OFFLINE ? 'fail' : ''}"></span>${OFFLINE ? 'Server unreachable' : 'Connecting…'}</span>`; return; }
  const ph = { open: ['ok', 'Market open'], pre_open: ['warn', 'Pre-open'], closed: ['', 'Market closed'], weekend: ['', 'Weekend'] }[s.phase] || ['', s.phase];
  const k = s.kite || {};
  const kd = k.state === 'valid' ? 'ok' : k.state === 'error' ? 'warn' : 'fail';
  const kt = k.state === 'valid' ? `Kite connected<small>· until ${hm(k.expires_at)}</small>` : k.state === 'error' ? 'Kite: could not verify' : 'Kite login needed';
  const pd = { OK: 'ok', WARN: 'warn', STALE: 'warn', FAIL: 'fail' }[s.pipeline.status] || '';
  $('#chips').innerHTML = `
    <span class="chip" title="IST clock · NSE 09:15–15:30 (holidays not known)"><span class="dot ${ph[0]}"></span>${ph[1]}</span>
    <a class="chip" href="#kite" title="Kite access token · dashboard/kite_auth.py"><span class="dot ${kd} ${kd === 'fail' ? 'pulse' : ''}"></span>${kt}</a>
    <a class="chip" href="#health" title="${esc(s.pipeline.reasons.join('; ') || 'all checks OK')}"><span class="dot ${pd}"></span>Pipeline ${esc(s.pipeline.status)}</a>
    <span class="vsep"></span>
    <span class="upd muted">${s.last_mark ? 'Updated ' + hm(s.last_mark) : ''}</span>
    <button class="iconbtn" data-act="theme" title="Theme: ${theme}">${themeIcon}</button>
    <button class="iconbtn" data-act="reload" title="Reload now">↻</button>`;
}

// ── attention banner + browser attention (auth refresh first) ───────────────
function renderAttention() {
  const items = [...(FLASH ? [FLASH] : []), ...((STATUS && STATUS.attention) || [])];
  const box = $('#attn');
  if (!items.length) { box.innerHTML = ''; browserAttention(null); return; }
  const a = items[0];
  const more = items.length > 1 ? `<span class="faint"> · +${items.length - 1} more</span>` : '';
  let acts = '';
  if (a.action === 'kite') acts = `${LOGIN_URL ? `<a class="btn primary" href="${esc(LOGIN_URL)}">Log in to Kite</a>` : ''}<a class="btn" href="#kite">Details</a>`;
  else if (a.action === 'health') acts = '<a class="btn" href="#health">Open System health</a>';
  if (a.level === 'urgent' && 'Notification' in window && Notification.permission === 'default')
    acts += '<button class="btn" data-act="notify">Enable desktop alerts</button>';
  if (a.level === 'ok') acts += '<button class="btn" data-act="dismiss">Dismiss</button>';
  const icon = a.level === 'ok' ? '✓' : '!';
  box.innerHTML = `<div class="attn ${a.level}" role="${a.level === 'urgent' ? 'alert' : 'status'}"><span class="ic">${icon}</span>
    <div><span class="t">${esc(a.title)}.</span> ${esc(a.message)}${more}</div><div class="acts">${acts}</div></div>`;
  browserAttention(items.find(i => i.level === 'urgent') || null);
}

let flashTimer = null;
function favicon(kind) {
  const svg = kind === 'urgent'
    ? "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'><circle cx='8' cy='8' r='7' fill='%23d9453a'/><rect x='7' y='3.5' width='2' height='6' fill='white'/><rect x='7' y='11' width='2' height='2' fill='white'/></svg>"
    : "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'><rect x='1.5' y='1.5' width='13' height='13' rx='3' fill='none' stroke='%23556' stroke-width='1.5'/><path d='M4 12 12 4M4 8 8 4M8 12 12 8' stroke='%23889' stroke-width='1'/></svg>";
  $('#favicon').href = 'data:image/svg+xml,' + svg;
}
function baseTitle() {
  const p = DATA.portfolio;
  const page = PAGES[route().name].title;
  return p && p.deployed ? `${rs(p.snapshot.nav, 0)} · ${page} · Paper Book` : `${page} · Paper Book`;
}
function browserAttention(urgent) {
  clearInterval(flashTimer); flashTimer = null;
  favicon(urgent ? 'urgent' : 'normal');
  if (!urgent) { document.title = baseTitle(); return; }
  const alertT = `⚠ ${urgent.title}`;
  document.title = `${alertT} · Paper Book`;
  // Flash the tab title only while the tab is in the background.
  let on = true;
  flashTimer = setInterval(() => {
    if (!document.hidden) { document.title = `${alertT} · Paper Book`; return; }
    on = !on; document.title = on ? alertT : baseTitle();
  }, 1200);
  // Desktop notification: at most once per item every 30 minutes.
  if ('Notification' in window && Notification.permission === 'granted') {
    const key = 'pb-notified-' + urgent.id;
    if (Date.now() - store.get(key, 0) > 30 * 60 * 1000) {
      store.set(key, Date.now());
      try {
        const n = new Notification(`Paper Book: ${urgent.title}`, { body: urgent.message, tag: urgent.id, requireInteraction: true });
        n.onclick = () => { window.focus(); location.hash = urgent.action === 'kite' ? '#kite' : '#health'; n.close(); };
      } catch { /* some browsers need a service worker; the banner and title still show */ }
    }
  }
}

// ── charts: one small SVG plotter, hover by nearest index ────────────────────
const PLOTS = {};
let plotSeq = 0;
function niceTicks(lo, hi, n = 4) {
  const span = hi - lo || 1, step0 = span / n, mag = 10 ** Math.floor(Math.log10(step0));
  const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(s => s >= step0) || mag * 10;
  const out = []; for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(v);
  return out;
}
/* opts: {n, x(i)->0..1000, series:[{ys,color,width,dash,area,dot}], band:{lo,hi}, base:{y,label},
          yfmt, xlabels:[[pct,text]], tip(i)->html, cls} */
function plot(o) {
  const id = 'p' + (++plotSeq);
  const all = [];
  o.series.forEach(s => s.ys.forEach(v => v != null && all.push(v)));
  if (o.band) { o.band.lo.forEach(v => all.push(v)); o.band.hi.forEach(v => all.push(v)); }
  if (o.base) all.push(o.base.y);
  let lo = Math.min(...all), hi = Math.max(...all);
  const pad = (hi - lo || Math.abs(hi) * 0.002 || 1) * 0.12; lo -= pad; hi += pad;
  const X = o.x || (i => o.n > 1 ? i / (o.n - 1) * 1000 : 500);
  const Y = v => (hi - v) / (hi - lo) * 250;
  const path = ys => ys.map((v, i) => v == null ? '' : `${i && ys[i - 1] != null ? 'L' : 'M'}${X(i).toFixed(1)} ${Y(v).toFixed(1)}`).join(' ');
  let svg = '';
  if (o.band) {
    const up = o.band.hi.map((v, i) => `${i ? 'L' : 'M'}${X(i).toFixed(1)} ${Y(v).toFixed(1)}`).join(' ');
    const dn = o.band.lo.map((v, i) => `L${X(i).toFixed(1)} ${Y(v).toFixed(1)}`).reverse().join(' ');
    svg += `<path d="${up} ${dn} Z" fill="var(--band)"/>`;
  }
  o.series.forEach(s => {
    const d = path(s.ys);
    if (s.area && s.ys.length) svg += `<path d="${d} L${X(s.ys.length - 1).toFixed(1)} 250 L${X(0).toFixed(1)} 250 Z" fill="${s.color}" fill-opacity="0.05"/>`;
    svg += `<path d="${d}" fill="none" stroke="${s.color}" stroke-width="${s.width || 1.75}" ${s.dash ? `stroke-dasharray="${s.dash}"` : ''} vector-effect="non-scaling-stroke" stroke-linejoin="round"/>`;
  });
  const ticks = niceTicks(lo + pad * 0.5, hi - pad * 0.5).map(v => `<div class="gl" style="top:${(Y(v) / 2.5).toFixed(2)}%"><span>${esc(o.yfmt(v))}</span></div>`).join('');
  const base = o.base ? `<div class="base" style="top:${(Y(o.base.y) / 2.5).toFixed(2)}%">${o.base.label ? `<span>${esc(o.base.label)}</span>` : ''}</div>` : '';
  const last = o.series[0] && o.series[0].dot ? o.series[0].ys.length - 1 : -1;
  const dot = last >= 0 ? `<div class="now" style="left:${(X(last) / 10).toFixed(2)}%;top:${(Y(o.series[0].ys[last]) / 2.5).toFixed(2)}%"></div>` : '';
  PLOTS[id] = { n: o.n, X, tip: o.tip };
  const xl = (o.xlabels || []).map(([p, t]) => `<span style="${p >= 99 ? 'right:0' : `left:${p}%`}">${esc(t)}</span>`).join('');
  return `<div class="plot ${o.cls || ''}" data-plot="${id}" style="${o.height ? `height:${o.height}px` : ''}">${ticks}${base}
    <svg viewBox="0 0 1000 250" preserveAspectRatio="none">${svg}</svg>${dot}<div class="vline" hidden></div><div class="tip" hidden></div></div>
    <div class="xax ${o.cls || ''}">${xl}</div>`;
}
document.addEventListener('mousemove', e => {
  const el = e.target.closest && e.target.closest('.plot[data-plot]');
  document.querySelectorAll('.plot .tip:not([hidden])').forEach(t => { if (!el || !el.contains(t)) { t.hidden = true; t.previousElementSibling.hidden = true; } });
  if (!el) return;
  const p = PLOTS[el.dataset.plot]; if (!p || !p.tip || !p.n) return;
  const r = el.getBoundingClientRect(), fx = (e.clientX - r.left) / r.width * 1000;
  let best = 0, bd = Infinity;
  for (let i = 0; i < p.n; i++) { const d = Math.abs(p.X(i) - fx); if (d < bd) { bd = d; best = i; } }
  const tip = el.querySelector('.tip'), vl = el.querySelector('.vline');
  const left = p.X(best) / 10;
  vl.hidden = false; vl.style.left = left + '%';
  tip.hidden = false; tip.innerHTML = p.tip(best);
  tip.style.left = left > 60 ? '' : `calc(${left}% + 12px)`;
  tip.style.right = left > 60 ? `calc(${100 - left}% + 12px)` : '';
  tip.style.top = '8px';
});

// ── Portfolio ────────────────────────────────────────────────────────────────
const KEYS = { symbol: 'symbol', sector: 'sector', qty: 'qty', avg: 'avg_price', ltp: 'ltp', d1pct: 'day_chg_pct', d1rs: 'day_chg_rs', value: 'value', cost: 'cost', pnlrs: 'pnl_rs', pnlpct: 'pnl_pct', weight: 'weight', stop: 'stop_gap' };
const PSORT = [['value', 'Value'], ['d1pct', '1D %'], ['pnlpct', 'P&L %'], ['symbol', 'A–Z']];
function sorted(rows, k, d) {
  const f = KEYS[k];
  return [...rows].sort((a, b) => { const x = a[f], y = b[f]; if (x == null) return 1; if (y == null) return -1; return (typeof x === 'string' ? x.localeCompare(y) : x - y) * d; });
}

function intraday(p) {
  const snap = p.snapshot, cap = snap.capital;
  const marks = p.marks || [];
  const lastDay = marks.length ? marks[marks.length - 1].ts.slice(0, 10) : null;
  const days = [...new Set(marks.map(m => m.ts.slice(0, 10)))];
  const range = ui.range;
  const enough = { '1D': true, '1W': days.length > 1, '1M': days.length > 5, All: days.length > 1 };
  const seg = ['1D', '1W', '1M', 'All'].map(r => `<button data-range="${r}" class="${range === r && enough[r] ? 'on' : ''}" ${enough[r] ? '' : `disabled title="History starts ${esc(dm(p.deployed_on || lastDay))}"`}>${r}</button>`).join('');
  let pts, x, xlabels, title = 'Value through the day';
  const eff = enough[range] ? range : '1D';
  if (eff === '1D') {
    pts = marks.filter(m => m.ts.slice(0, 10) === lastDay).map(m => ({ ts: m.ts, nav: m.nav, pnl: m.pnl_rs }));
    const fillDay = (p.trade_days || []).find(t => t.date === lastDay && t.kind === 'first purchase');
    if (fillDay) pts.unshift({ ts: fillDay.ts, nav: cap - fillDay.charges, pnl: -fillDay.charges, fill: true });
    const mins = pts.map(q => minutesIST(q.ts));
    x = i => Math.max(0, Math.min(1000, (mins[i] - 555) / 375 * 1000));
    xlabels = [[0, '09:15'], [28, '11:00'], [60, '13:00'], [100, '15:30']];
  } else {
    const since = eff === '1W' ? days.slice(-5)[0] : eff === '1M' ? days.slice(-21)[0] : days[0];
    pts = marks.filter(m => m.ts.slice(0, 10) >= since).map(m => ({ ts: m.ts, nav: m.nav, pnl: m.pnl_rs }));
    x = null; title = 'Value over time';
    xlabels = [[0, dm(pts[0].ts)], [100, dm(pts[pts.length - 1].ts)]];
  }
  if (!pts.length) return `<div class="card chartcard"><div class="hd">${title}<div class="seg">${seg}</div></div><div class="empty">No price updates yet today.</div></div>`;
  const navs = pts.map(q => q.nav);
  const lowI = navs.indexOf(Math.min(...navs));
  const nxt = STATUS && STATUS.next_event && STATUS.next_event.what === 'price update' ? ` · next mark ${hm(STATUS.next_event.at)}` : '';
  return `<div class="card chartcard"><div class="hd">${title}<div class="seg">${seg}</div></div>
    ${plot({ n: pts.length, x, series: [{ ys: navs, color: 'var(--ink)', area: true, dot: true }], base: { y: cap, label: 'Start ' + rs(cap, 0) },
      yfmt: v => rs(v, 0), xlabels,
      tip: i => `<b>${pts[i].fill ? '09:16 fill' : (eff === '1D' ? hm(pts[i].ts) : wdm(pts[i].ts) + ' ' + hm(pts[i].ts))}</b> · ${rs(pts[i].nav)} <span class="${tone(pts[i].pnl)}">${srs(pts[i].pnl, 0)}</span>` })}
    <div class="chartft"><span>Low ${rs(navs[lowI], 0)} at ${pts[lowI].fill ? '09:16' : hm(pts[lowI].ts)} · ${pts.length} marks</span><span class="mono" style="font-size:11px">live/marks.csv${nxt}</span></div></div>`;
}

function pagePortfolio(p) {
  if (!p.deployed) {
    return `<section class="hero"><div><div class="lbl">Portfolio value</div><div class="navbig">₹1,00,000<span>.00</span></div>
      <div class="note"><span class="i">i</span><span>Not invested yet. ₹1,00,000 of simulated cash is invested by the algorithm at 09:16 IST on the next trading day, after the Kite login.</span></div></div>
      <div class="card chartcard"><div class="empty">Holdings, sectors and the day's chart appear after the first purchase.</div></div></section>`;
  }
  const s = p.snapshot, st = p.state || {};
  const [nI, nD] = rs(s.nav).split('.');
  const H = s.holdings.map(h => ({ ...h, stop_gap: h.stop_level ? h.ltp / h.stop_level - 1 : null }));
  const today = istDay(new Date());
  const markDay = s.ts.slice(0, 10);
  const boughtToday = p.deployed_on === markDay;
  const invested = H.reduce((a, h) => a + h.cost, 0);
  const sells = (p.trade_days || []).reduce((a, t) => a + t.sells, 0);
  const pending = st.pending_stops || [];
  const cool = Object.entries(st.cooldown || {});
  const phase = STATUS ? STATUS.phase : 'closed';
  const closeMarkDone = s.close_mark || minutesIST(s.ts) >= 15 * 60 + 35 && markDay === today;
  const closeTxt = markDay === today && !closeMarkDone && (phase === 'open' || phase === 'pre_open') ? '15:35 today' : 'next session, 15:35';

  // desktop holdings table
  const ind = k => ui.sk === k ? (ui.sd < 0 ? ' ↓' : ' ↑') : '';
  const th = (k, label, cls = '') => `<th data-sort="${k}" class="${cls}">${cls === 'l' ? label + ind(k) : ind(k).trim() + ' ' + label}</th>`;
  const rows = sorted(H, ui.sk, ui.sd).map(h => `<tr>
    <td class="l sym">${esc(h.symbol)}</td><td class="l sec" title="${esc(h.sector)}">${esc(h.sector)}</td>
    <td>${h.qty}</td><td class="muted">${rs(h.avg_price)}</td><td>${rs(h.ltp)}</td>
    <td class="${tone(h.day_chg_rs)}">${arr(h.day_chg_pct)}</td><td class="${tone(h.day_chg_rs)}">${srs(h.day_chg_rs)}</td>
    <td style="font-weight:500">${rs(h.value)}</td><td class="muted">${rs(h.cost)}</td>
    <td class="${tone(h.pnl_rs)}">${srs(h.pnl_rs)}</td><td class="${tone(h.pnl_rs)}">${arr(h.pnl_pct)}</td>
    <td class="l" title="Cap is 10% per stock"><div class="wb"><span>${(h.weight * 100).toFixed(1)}%</span><span class="bar"><i style="width:${Math.min(100, h.weight * 1000).toFixed(1)}%"></i></span></div></td>
    <td class="${h.stop_level ? '' : 'faint'}" title="${h.stop_level ? 'Sold at the next 09:16 if a close falls below this' : 'Stop levels are set at the close mark, 15:35'}">${h.stop_level ? `${rs(h.stop_level)} <span class="faint">${(h.stop_gap * 100).toFixed(1)}% away</span>` : '— at 15:35'}</td>
    <td class="muted">${h.opened ? dm(h.opened) : ''}</td></tr>`).join('');

  // phone holdings list
  const prow = sorted(H, ui.ps, ui.ps === 'symbol' ? 1 : -1).map(h => `<div class="row">
    <div style="min-width:0"><div style="font-weight:500">${esc(h.symbol)}</div><div class="sub">${h.qty} × ${rs(h.ltp)} · ${esc(h.sector)}</div></div>
    <div class="r ${tone(h.day_chg_rs)}" style="font-size:12.5px">${arr(h.day_chg_pct)}</div>
    <div class="r"><div style="font-weight:500">${rs(h.value, 0)}</div><div class="${tone(h.pnl_rs)}" style="font-size:11.5px">${srs(h.pnl_rs)}</div></div></div>`).join('');

  // sectors
  const maxV = Math.max(...H.map(h => h.value));
  const sect = Object.values(H.reduce((m, h) => { (m[h.sector] = m[h.sector] || { name: h.sector, list: [] }).list.push(h); return m; }, {}))
    .map(x => ({ ...x, v: x.list.reduce((a, h) => a + h.value, 0), p: x.list.reduce((a, h) => a + h.pnl_rs, 0), d: x.list.reduce((a, h) => a + h.day_chg_rs, 0) }))
    .sort((a, b) => b.v - a.v);
  const wscale = Math.max(0.2, ...sect.map(x => x.v / s.nav));
  const secRows = sect.map(x => `<tr><td class="l" style="max-width:190px;overflow:hidden;text-overflow:ellipsis" title="${esc(x.name)}">${esc(x.name)}</td><td class="muted">${x.list.length}</td>
    <td class="l"><div class="wb"><span>${(x.v / s.nav * 100).toFixed(1)}%</span><span class="bar" style="width:80px;height:6px"><i style="width:${(x.v / s.nav / wscale * 100).toFixed(1)}%;background:var(--ink2)"></i></span></div></td>
    <td>${rs(x.v, 0)}</td><td class="${tone(x.d)}">${srs(x.d)}</td><td class="${tone(x.p)}">${srs(x.p)}</td></tr>`).join('');
  const bars = sect.map(x => `<div class="sgh"><b>${esc(x.name)}</b><span class="faint">${(x.v / s.nav * 100).toFixed(1)}% · <span class="${tone(x.p)}">${srs(x.p)}</span></span></div>` +
    x.list.sort((a, b) => b.value - a.value).map(h => `<div class="srow" title="${esc(h.symbol)} · P&L ${srs(h.pnl_rs)} (${arr(h.pnl_pct)}) · 1D ${arr(h.day_chg_pct)}">
      <span class="muted">${esc(h.symbol)}</span><span class="tr"><i style="width:${(h.value / maxV * 100).toFixed(1)}%;background:var(--${h.pnl_rs >= 0 ? 'up' : 'down'})"></i></span>
      <span class="r val">${rs(h.value, 0)}</span><span class="r ${tone(h.day_chg_rs)}">${arr(h.day_chg_pct)}</span></div>`).join('')).join('');

  // trades
  const kindLabel = k => ({ 'first purchase': 'First purchase', rebalance: 'Monthly rebalance', volstop: 'Stop-loss sales' }[k] || titleCase(k));
  const trades = (p.trade_days || []).map(t => {
    const open = !!ui.open[t.date];
    const fills = open ? `<table class="t compact fills"><thead><tr><th class="l" style="padding-left:58px">Time</th><th class="l">Side</th><th class="l">Stock</th><th class="l desk-only">Sector</th><th>Qty</th><th>Price</th><th>Value</th><th>Charges</th><th class="desk-only">Realised</th></tr></thead><tbody>` +
      t.fills.map(f => `<tr><td class="l muted" style="padding-left:58px">${f.ts ? hms(f.ts) : ''}</td><td class="l"><span class="tag ${f.side === 'SELL' ? 'sell' : ''}">${esc(f.side)}</span></td>
        <td class="l sym">${esc(short(f.ticker))}</td><td class="l sec desk-only">${esc(f.sector)}</td><td>${f.qty}</td><td>${rs(f.price)}</td><td>${rs(f.value)}</td><td class="muted">${rs(f.charges)}</td>
        <td class="desk-only ${f.realised_rs != null ? tone(f.realised_rs) : 'faint'}">${f.realised_rs != null ? srs(f.realised_rs) : '—'}</td></tr>`).join('') + '</tbody></table>' : '';
    return `<button class="trow" data-trade="${esc(t.date)}"><span class="faint">${open ? '▾' : '▸'}</span><span style="font-weight:500">${wdm(t.date)}${t.ts ? ', ' + hm(t.ts) : ''}</span>
      <span class="muted">${kindLabel(t.kind)}</span><span class="xs"><span class="muted">Buys</span> ${t.buys} <span class="muted">· Sells</span> ${t.sells}</span>
      <span class="xs"><span class="muted">${t.sold ? 'Bought / sold' : 'Bought'}</span> ${rs(t.bought, 0)}${t.sold ? ' / ' + rs(t.sold, 0) : ''}</span><span class="xs"><span class="muted">Charges</span> ${rs(t.charges)}</span></button>${fills}`;
  }).join('');

  return `
  <section class="hero">
    <div style="display:flex;flex-direction:column">
      <div class="lbl" title="live/holdings_latest.json → nav">Portfolio value · ${dm(s.ts)}, marked ${hm(s.ts)} IST</div>
      <div class="navbig">${nI}<span>.${nD}</span></div>
      <div class="kpi2">
        <div title="holdings_latest.json → day_chg_rs, day_chg_pct"><div class="lbl">Today</div><div class="v">${srs(s.day_chg_rs)}</div>
          <div class="s"><b class="${tone(s.day_chg_rs)}">${arr(s.day_chg_pct)}</b><i>${boughtToday ? 'vs 09:16 fill prices' : "vs yesterday's close"}</i></div></div>
        <div title="holdings_latest.json → pnl_rs, pnl_pct"><div class="lbl">Total P&amp;L</div><div class="v">${srs(s.pnl_rs)}</div>
          <div class="s"><b class="${tone(s.pnl_rs)}">${arr(s.pnl_pct)}</b><i>on ${rs(s.capital, 0)}, after charges</i></div></div>
      </div>
      <div class="note"><span class="i">i</span><span>Session ${p.sessions} of a ${p.evidence_months}-month test. A day's move, up or down, says nothing about the algorithm yet. <a href="#experiment">How it compares to random picks →</a></span></div>
    </div>
    ${intraday(p)}
  </section>

  <section class="card metrics">
    <div title="Σ holdings.cost"><div class="k">Invested</div><div class="v">${rs(invested)}</div><div class="s">${H.length} stocks, incl. charges</div></div>
    <div title="holdings_latest.json → cash"><div class="k">Cash</div><div class="v">${rs(s.cash)}</div><div class="s">${(s.cash / s.nav * 100).toFixed(1)}% of value</div></div>
    <div title="holdings_latest.json → realised_rs"><div class="k">Realised P&amp;L</div><div class="v ${tone(s.realised_rs || 0)}">${srs(s.realised_rs || 0)}</div><div class="s">${sells ? `${sells} sale${sells > 1 ? 's' : ''}` : 'no sales yet'}</div></div>
    <div title="holdings_latest.json → charges_rs"><div class="k">Charges paid</div><div class="v">${rs(s.charges_rs || 0)}</div><div class="s">STT, stamp, exchange, GST, DP</div></div>
    <div><div class="k">Starting capital</div><div class="v">${rs(s.capital, 0)}</div><div class="s">invested ${p.deployed_on ? dm(p.deployed_on) : ''} 09:16</div></div>
    <div title="holdings_latest.json → ts"><div class="k">Last marked</div><div class="v">${hm(s.ts)} IST</div><div class="s">${s.prices === 'live' ? 'live Kite quotes · ' + ago(s.ts) : 'rehearsal prices'}</div></div>
  </section>

  <section class="next">
    <span>What happens next</span>
    <span><span class="muted">Close mark</span> ${closeTxt}<span class="faint"> · stop levels are set</span></span>
    <span title="state.json → last_rebalance; next = first session of next month"><span class="muted">Next rebalance</span> ${wdm(p.next_rebalance)}, 09:16<span class="faint"> · up to 30% turnover</span></span>
    <span title="state.json → pending_stops"><span class="muted">Stops to sell at next open</span> ${pending.length ? `<span class="alert">${pending.map(short).map(esc).join(', ')}</span>` : 'none'}</span>
    <span title="state.json → cooldown"><span class="muted">Re-entry bans</span> ${cool.length ? cool.map(([t, n]) => `${esc(short(t))} (${n})`).join(', ') : 'none'}</span>
    <details><summary>How the rules work</summary><div class="pop">Rebalance: on the first session of each month the book re-selects its top 30 stocks, changing at most 30% of holdings.<br><br>Volatility stop: if a stock closes below its stop level, it is sold at the next 09:16 and can't be bought again for 21 sessions. The stop sits between 5% and 30% below the buy price, set by the stock's own 20-day volatility.</div></details>
  </section>
  ${s.unpriced && s.unpriced.length ? `<div class="callout warn">No live quote for ${s.unpriced.map(short).map(esc).join(', ')}; valued at the purchase price.</div>` : ''}

  <section class="sec">
    <div class="sech"><h2>Holdings</h2><span class="n">${H.length}</span><span class="src">live/holdings_latest.json · ${hm(s.ts)}</span></div>
    <div class="card tblwrap desk-only"><table class="t"><thead><tr>
      ${th('symbol', 'Stock', 'l')}${th('sector', 'Sector', 'l')}${th('qty', 'Qty')}${th('avg', 'Avg buy')}${th('ltp', 'LTP')}${th('d1pct', '1D %')}${th('d1rs', '1D ₹')}${th('value', 'Value')}${th('cost', 'Invested')}${th('pnlrs', 'P&amp;L ₹')}${th('pnlpct', 'P&amp;L %')}${th('weight', 'Weight', 'l')}${th('stop', 'Stop level')}<th>Opened</th>
    </tr></thead><tbody>${rows}</tbody></table></div>
    <div class="tblnote desk-only"><span>▲ ▼ and ± mark direction; colour is a second cue, never the only one.</span><span>Click any column to sort.</span></div>
    <div class="phone-only"><div class="chipsort">${PSORT.map(([k, l]) => `<button data-psort="${k}" class="${ui.ps === k ? 'on' : ''}">${l}</button>`).join('')}</div>
      <div class="card plist">${prow}</div></div>
  </section>

  <section class="sec">
    <div class="sech"><h2>By sector</h2><span class="n">${sect.length} sectors</span><span class="src">NSE industry names</span></div>
    <div class="sectors">
      <div class="card desk-only" style="overflow:hidden"><table class="t compact"><thead><tr><th class="l">Sector</th><th>#</th><th class="l">Weight</th><th>Value</th><th>1D ₹</th><th>P&amp;L ₹</th></tr></thead><tbody>${secRows}</tbody></table></div>
      <div class="card barlist"><div class="hd"><b>Each stock, grouped by sector · bar = value</b><span class="legend"><span><i style="background:var(--up)"></i>▲ in profit</span><span><i style="background:var(--down)"></i>▼ in loss</span></span></div>
        <div class="body">${bars}</div></div>
    </div>
  </section>

  <section class="sec">
    <div class="sech"><h2>Trades</h2><span class="src">live/trades.jsonl · live/ledger.jsonl</span></div>
    <div class="card" style="overflow:hidden">${trades || '<div class="empty">No trades yet.</div>'}</div>
  </section>

  <footer class="honest"><p>Early P&amp;L is noise. The algorithm has not shown that it picks better than random stock selection; the pre-registered test needs ${p.evidence_months} months of record, and only rank 1 of 21 at that point counts. <a href="#experiment">Open the experiment →</a></p>
    <p>Paper portfolio · simulated fills at live Kite quotes · no orders placed</p></footer>`;
}

// ── Experiment ───────────────────────────────────────────────────────────────
const BOOKNAME = { signal: 'Algorithm (volatility stop)', equal_weight: 'Equal-weight' };
function kpi(k, v, s, cls = '') { return `<div><div class="k">${k}</div><div class="v ${cls}">${v}</div><div class="s">${s || ''}</div></div>`; }
function pageExperiment(d) {
  const head = `<div class="page-head"><h1>Experiment</h1><p>The same algorithm replayed at NSE's official prices against 20 books that pick stocks at random with identical rules, and an equal-weight book. Recorded once a day after the close. Pre-registered in <span class="mono">audit/P2_PAPER_PERMUTATION_TEST.md</span>.</p></div>`;
  if (d.error) {
    return head + `<div class="callout"><span>ⓘ</span><div><b>No sessions recorded yet.</b> ${esc(d.error)}.<br>The record starts with the 30 Sep session. A session's close is recorded once the next session exists, so the first entry arrives with the 21:00 run on the following trading day.</div></div>
      <div class="callout"><span>ⓘ</span><div>${esc(d.restart_note)}</div></div>` + adaptiveSection(d.adaptive);
  }
  const h = d.headline;
  const nb = h.n_null + 1;
  const pk = x => x ? `${srs(x.rupees, 0)} <span class="${tone(x.rupees)}" style="font-size:13px">${arr(x.pct)}</span>` : '—';
  const monthsPct = Math.min(100, h.months_elapsed / d.evidence_months * 100);
  // chart
  const dates = [...new Set([...d.band.map(b => b[0]), ...d.lines.flatMap(l => l.points.map(q => q[0]))])].sort();
  const at = (pts) => { const m = new Map(pts); return dates.map(x => m.has(x) ? m.get(x) : null); };
  const bandM = new Map(d.band.map(b => [b[0], b]));
  const lines = d.lines.filter(l => l.kind !== 'overlay' || ui.overlays);
  const colors = { signal: 'var(--c1)', equal_weight: 'var(--c2)', overlay: 'var(--ink3)' };
  const series = lines.map(l => ({ ys: at(l.points), color: colors[l.kind], width: l.kind === 'signal' ? 2.5 : 1.75, dash: l.kind === 'overlay' ? '4 3' : null, name: BOOKNAME[l.kind] || l.book }));
  if (d.band.length) series.push({ ys: dates.map(x => bandM.has(x) ? bandM.get(x)[2] : null), color: 'var(--ink3)', dash: '2 3', width: 1.25, name: 'Random median' });
  const chart = dates.length ? plot({
    n: dates.length, series, band: d.band.length ? { lo: dates.map(x => (bandM.get(x) || [0, 0])[1]), hi: dates.map(x => (bandM.get(x) || [0, 0, 0, 0])[3]) } : null,
    base: { y: 0, label: '' }, yfmt: v => srs(v, 0), height: 300,
    xlabels: [[0, dm(dates[0])], [100, dm(dates[dates.length - 1])]],
    tip: i => `<b>${wdm(dates[i])}</b><br>` + series.map(s => s.ys[i] == null ? '' : `${esc(s.name)}: ${srs(s.ys[i], 0)}`).filter(Boolean).join('<br>'),
  }) : '<div class="empty">No sessions yet.</div>';
  const legend = `<div class="lg">${series.map(s => `<span><i style="background:${s.color}"></i>${esc(s.name)}</span>`).join('')}${d.band.length ? '<span><i style="background:var(--band);height:10px"></i>Random books, 10th–90th percentile</span>' : ''}
    <label style="margin-left:auto"><input type="checkbox" data-act="overlays" ${ui.overlays ? 'checked' : ''}> Show the other three stop variants</label></div>`;
  const integ = [
    d.bad_lines ? `<div class="callout warn">${d.bad_lines} unparseable line(s) in record.jsonl were skipped.</div>` : '',
    d.missing_pairs.length ? `<div class="callout warn">${d.missing_pairs.length} (session, book) pair(s) missing from the record; each counts as a zero return. First: ${esc(JSON.stringify(d.missing_pairs.slice(0, 3)))}</div>` : '',
  ].join('');
  const cols = d.sessions.length ? Object.keys(d.sessions[0]) : [];
  const table = d.sessions.length ? `<div class="card tblwrap" style="max-height:420px"><table class="t compact"><thead><tr>${cols.map((c, i) => `<th class="${i ? '' : 'l'}">${esc(c)}</th>`).join('')}</tr></thead><tbody>${d.sessions.map(r => `<tr>${cols.map((c, i) => `<td class="${i ? '' : 'l'}">${typeof r[c] === 'number' ? (Math.abs(r[c]) < 1 ? spct(r[c]) : g(r[c], 2)) : esc(r[c])}</td>`).join('')}</tr>`).join('')}</tbody></table></div>` : '';
  return head + `
  <section class="card kpis">
    ${kpi('Algorithm book', pk(h.signal), 'on ₹1,00,000, since ' + esc(dm(h.first_date)))}
    ${kpi('Equal-weight', pk(h.equal_weight), 'every eligible stock, equal amounts')}
    ${kpi(`Random books, median of ${h.n_null}`, pk(h.null_median), 'same rules, random rankings')}
    ${kpi('Random, 10th–90th pct', h.null_p10 ? `${srs(h.null_p10.rupees, 0)} … ${srs(h.null_p90.rupees, 0)}` : '—', 'the range luck alone produces')}
  </section>
  <div class="callout warn"><span>!</span><div><b>Rank ${h.rank ?? '—'} of ${nb} — not evidence yet.</b> ${h.sessions} session(s) recorded, ${esc(dm(h.first_date))} to ${esc(dm(h.last_date))}. The pre-registered test reads the rank only at ${d.evidence_months} months, and success is exactly rank 1 of ${nb} then; nothing else counts. Until then the rank is shown only to check the plumbing.
    <div class="progress" title="${h.months_elapsed.toFixed(2)} of ${d.evidence_months} months"><i style="width:${monthsPct.toFixed(2)}%"></i></div><div class="faint" style="font-size:12px;margin-top:4px">${h.months_elapsed.toFixed(2)} of ${d.evidence_months} months · ≈ ${h.trading_months.toFixed(2)} trading months</div></div></div>
  ${integ}
  <section class="sec"><div class="sech"><h2>Cumulative P&amp;L</h2><span class="src">audit/paper/record.jsonl</span></div>
    <div class="card big">${chart}${legend}</div></section>
  <section class="sec"><div class="sech"><h2>Per session</h2><span class="n">${d.sessions.length}</span><span class="src">daily log returns as first recorded</span></div>${table}</section>
  <div class="callout" style="margin-top:28px"><span>ⓘ</span><div>${esc(d.restart_note)}</div></div>` + adaptiveSection(d.adaptive);
}

// P3: the adaptive shadow book (audit/P3_ADAPTIVE_SHADOW.md)
function adaptiveSection(a) {
  if (!a) return '';
  const refits = (a.refits || []).slice().reverse().map(r => `<tr><td class="l sym">${esc(dm(r.fit_date))}</td>
    <td class="l">${pill(r.status === 'OK' ? 'OK' : 'FAIL')}</td>
    <td class="l muted">${r.window ? `train ${esc(r.window.train_start)} → ${esc(r.window.train_end)} · val → ${esc(r.window.val_end)}` : esc(r.error || '')}</td>
    <td>${r.wall_s != null ? g(r.wall_s / 60, 1) + ' min' : '—'}</td><td class="l">${esc(r.gate_verdict || '—')}</td></tr>`).join('');
  const intro = `<div class="sech" style="margin-top:56px"><h2>Adaptive shadow book</h2><span class="n">P3</span><span class="src">audit/P3_ADAPTIVE_SHADOW.md</span></div>
    <p class="muted" style="margin:0 0 14px;max-width:860px;line-height:1.55">The same rules, except the model is refitted on the newest data before every rebalance and the book rebalances twice a month (first session on or after the 1st and the 16th). Replayed end-of-day like the test above, from ${esc(dm(a.start))}. The live ₹1,00,000 portfolio stays on the frozen model. Read once, on ${esc(dm(a.read_at))} ${esc(a.read_at.slice(0, 4))}; until then it is plumbing, not evidence.</p>`;
  const refitTable = `<div class="card tblwrap" style="max-height:300px"><table class="t compact"><thead><tr><th class="l">Data through</th><th class="l">Status</th><th class="l">Window</th><th>Took</th><th class="l">Gate (recorded, not used)</th></tr></thead><tbody>${refits || '<tr><td class="l faint" colspan="5">No refit yet. The first runs on data through 30 Sep.</td></tr>'}</tbody></table></div>`;
  if (!a.headline) return intro + refitTable + `<div class="callout"><span>ⓘ</span><div>No sessions recorded yet. The first rebalance is ${esc(wdm(a.start))} at the open; its close is recorded by the following night's run.</div></div>`;
  const h = a.headline;
  const pk = x => `${srs(x.rupees, 0)} <span class="${tone(x.rupees)}" style="font-size:13px">${arr(x.pct)}</span>`;
  const dates = [...new Set([...a.band.map(b => b[0]), ...a.lines.flatMap(l => l.points.map(q => q[0]))])].sort();
  const at = pts => { const m = new Map(pts); return dates.map(x => m.has(x) ? m.get(x) : null); };
  const bm = new Map(a.band.map(b => [b[0], b]));
  const names = { adaptive: 'Adaptive book', frozen: 'Frozen algorithm (P2)', equal_weight: 'Equal-weight, twice monthly' };
  const colors = { adaptive: 'var(--c1)', frozen: 'var(--ink)', equal_weight: 'var(--c2)' };
  const series = a.lines.map(l => ({ ys: at(l.points), color: colors[l.kind], width: l.kind === 'adaptive' ? 2.5 : 1.5, dash: l.kind === 'frozen' ? '5 3' : null, name: names[l.kind] }));
  const chart = dates.length ? plot({ n: dates.length, series, band: a.band.length ? { lo: dates.map(x => (bm.get(x) || [0, 0])[1]), hi: dates.map(x => (bm.get(x) || [0, 0, 0, 0])[3]) } : null,
    base: { y: 0 }, yfmt: v => srs(v, 0), height: 280, xlabels: [[0, dm(dates[0])], [100, dm(dates[dates.length - 1])]],
    tip: i => `<b>${wdm(dates[i])}</b><br>` + series.map(s => s.ys[i] == null ? '' : `${esc(s.name)}: ${srs(s.ys[i], 0)}`).filter(Boolean).join('<br>') }) : '';
  return intro + `<section class="card kpis" style="margin-top:0">
      ${kpi('Adaptive book', pk(h.adaptive), 'since ' + esc(dm(h.first_date)))}
      ${kpi('Frozen algorithm, same days', pk(h.frozen), 'the P2 signal book')}
      ${kpi(`Its random books, median of ${h.n_null}`, pk(h.null_median), 'twice-monthly, same rules')}
      ${kpi('Rank vs its random books', `${h.rank ?? '—'} / ${h.n_null + 1}`, 'not evidence until the read')}
    </section>
    <div class="card big" style="margin-top:16px">${chart}<div class="lg">${series.map(s => `<span><i style="background:${s.color}"></i>${esc(s.name)}</span>`).join('')}${a.band.length ? '<span><i style="background:var(--band);height:10px"></i>Its random books, 10th–90th</span>' : ''}</div></div>
    <div class="sech" style="margin-top:24px"><h2 style="font-size:15px">Refits</h2><span class="n">${(a.refits || []).length}</span><span class="src">audit/paper/adaptive/refits.jsonl</span></div>${refitTable}`;
}

// ── System health ────────────────────────────────────────────────────────────
const pill = s => `<span class="pill ${esc(s)}">${esc(s)}</span>`;
function pageHealth(d) {
  const pr = d.probe;
  const pipe = d.pipeline;
  const probeTxt = pr.can ? 'Probes NSE (about 12 small requests). At most once per 10 minutes.' : (pr.why === 'rate limit' && pr.age_s != null ? `Next allowed in ${Math.ceil((pr.min_interval_s - pr.age_s) / 60)} min.` : esc(pr.why));
  const live = d.live.map(c => `<div>${pill(c.status)}<span class="nm">${esc(c.check)}</span><span class="dt">${esc(c.detail)}</span><span></span></div>`).join('');
  const rep = d.report;
  const checks = rep ? rep.checks.map(c => `<div>${pill(c.status || '?')}<span class="nm">${esc(c.check)}</span><span class="dt">${esc(c.detail || '')}${c.probes ? `<br><span class="faint">${c.probes.map(p => `${esc(dm(p.date))}: ${p.bhavcopy}/${p.index}`).join(' · ')}</span>` : ''}${c.missing && c.missing.length ? `<br><span class="faint">published but unrecorded: ${esc(c.missing.join(', '))}</span>` : ''}</span><span class="when faint mono" style="font-size:11px">${esc(c._source)} · ${esc(String(c._checked_at || '').slice(0, 16).replace('T', ' '))}</span></div>`).join('') : `<div class="empty">${esc(d.report_error)}</div>`;
  const runs = [...d.runs].reverse().map((b, i) => {
    const steps = b.lines.map(ln => { const m = ln.match(/^(\S+Z)\s+(.*)$/); if (!m) return null; const ok = /\bOK\b|DONE|nothing to record|no new session/.test(m[2]); const bad = /FAIL/.test(m[2]); return `<div><span class="muted mono" style="font-size:12px">${hm(m[1])}</span><span class="d ${bad ? 'fail' : ok ? 'ok' : ''}"></span><span>${esc(m[2])}</span></div>`; }).filter(Boolean).join('');
    return `<details class="run" ${i === 0 ? 'open' : ''}><summary>${pill(/FAIL/.test(b.outcome) ? 'FAIL' : 'OK')}<b style="font-weight:500">${esc(wdm(b.started))} ${hm(b.started)} IST</b><span class="muted">${esc(b.outcome)}</span><span class="src">logs/paper/${esc(b.file)}</span></summary><div class="steps">${steps || '<pre class="log">' + esc(b.lines.join('\n')) + '</pre>'}</div></details>`;
  }).join('');
  let mem = '<div class="empty">No memory samples.</div>';
  if (d.memory && d.memory.rows.length) {
    const rows = d.memory.rows, cols = d.memory.columns;
    const clr = ['var(--c1)', 'var(--c2)'];
    mem = plot({ n: rows.length, series: cols.map((c, j) => ({ ys: rows.map(r => r[j + 1]), color: clr[j], width: 1.5 })), yfmt: v => g(v, 0) + ' MB', height: 220,
      xlabels: [[0, hm(rows[0][0])], [100, hm(rows[rows.length - 1][0])]], tip: i => `<b>${hm(rows[i][0])}</b><br>` + cols.map((c, j) => `${c}: ${g(rows[i][j + 1], 0)} MB`).join('<br>') }) +
      `<div class="lg">${cols.map((c, j) => `<span><i style="background:${clr[j]}"></i>${esc(c)}</span>`).join('')}<span class="src">${esc(d.memory.source)}</span></div>`;
  }
  const det = d.determinism;
  return `<div class="page-head"><h1>System health</h1><p>Whether the data pipeline, the schedule and the record are working. This is the plumbing, not the strategy.</p></div>
  <section class="card" style="padding:20px 22px;margin-top:20px;display:flex;gap:20px;align-items:center;flex-wrap:wrap">
    <div style="font-size:18px;font-weight:500;display:flex;gap:12px;align-items:center">Data pipeline ${pill(pipe.status)}</div>
    <div class="muted" style="flex:1;min-width:240px">${pipe.reasons.length ? pipe.reasons.map(esc).join('<br>') : 'All checks OK.'}</div>
    <div style="text-align:right"><button class="btn" data-act="rerun" ${pr.can ? '' : 'disabled'}>Re-run health check</button><div class="faint" style="font-size:12px;margin-top:6px;max-width:300px">${probeTxt}</div></div>
  </section>
  <div class="grid2 sec">
    <section><div class="sech"><h2>Live checks</h2><span class="src">recomputed from logs, no network</span></div><div class="card checks">${live}</div></section>
    <section><div class="sech"><h2>Schedule</h2><span class="src">docker/paper.crontab · IST</span></div>
      <div class="card"><table class="t compact wrap"><tbody>${d.schedule.map(r => `<tr><td style="width:40%"><b style="font-weight:500">${esc(r[0])}</b><br><span class="faint">${esc(r[1])}</span></td><td>${esc(r[2])}</td></tr>`).join('')}</tbody></table></div></section>
  </div>
  <section class="sec"><div class="sech"><h2>Health report</h2>${rep ? pill(rep.overall) : ''}<span class="src">${rep ? 'newest ' + esc(String(rep.checked_at).slice(0, 16).replace('T', ' ')) + ' UTC · ' + rep.age_hours.toFixed(1) + ' h ago' : ''}</span></div>
    ${rep && rep.stale ? `<div class="callout warn" style="margin:0 0 12px">Not re-checked for ${rep.age_hours.toFixed(0)} h.</div>` : ''}
    <div class="card checks">${checks}</div></section>
  <section class="sec"><div class="sech"><h2>Recent runs</h2><span class="src">logs/paper/*.status</span></div><div class="card" style="overflow:hidden">${runs || '<div class="empty">No runs logged.</div>'}</div></section>
  <div class="grid2 sec">
    <section><div class="sech"><h2>Memory during the latest run</h2></div><div class="card big">${mem}</div></section>
    <section><div class="sech"><h2>Determinism</h2><span class="src">${det ? esc(det.source) + '/summary.json' : ''}</span></div>
      ${det ? `<div class="card kpis" style="margin-top:0;grid-template-columns:1fr 1fr">${kpi('Compared against', esc(det.previous))}${kpi('Overlap days', esc(det.overlap_days))}${kpi('Max abs relative diff', Number(det.max_abs_rel_diff || 0).toExponential(2))}${kpi('Books diverged', esc(det.books_diverged))}</div>` : '<div class="card empty">No replay summary yet. It appears after the first recorded session.</div>'}</section>
  </div>`;
}

// ── Research log ─────────────────────────────────────────────────────────────
function md(src) {
  const inline = s => esc(s).replace(/`([^`]+)`/g, '<code>$1</code>').replace(/\*\*([^*]+)\*\*/g, '<b>$1</b>').replace(/(^|[^*])\*([^*\s][^*]*)\*/g, '$1<i>$2</i>')
    .replace(/\[([^\]]+)\]\((https?:[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>').replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, '$1');
  const L = src.split('\n'); let out = '', i = 0;
  while (i < L.length) {
    const l = L[i];
    if (/^```/.test(l)) { const b = []; i++; while (i < L.length && !/^```/.test(L[i])) b.push(L[i++]); i++; out += `<pre><code>${esc(b.join('\n'))}</code></pre>`; continue; }
    if (/^#{1,6}\s/.test(l)) { const n = l.match(/^#+/)[0].length; out += `<h${Math.min(n, 3)}>${inline(l.replace(/^#+\s*/, ''))}</h${Math.min(n, 3)}>`; i++; continue; }
    if (/^\s*\|/.test(l) && i + 1 < L.length && /^\s*\|?\s*:?-{2,}/.test(L[i + 1])) {
      const cells = r => r.trim().replace(/^\||\|$/g, '').split('|').map(c => c.trim());
      const hd = cells(l); i += 2; let body = '';
      while (i < L.length && /^\s*\|/.test(L[i])) { body += '<tr>' + cells(L[i]).map(c => `<td>${inline(c)}</td>`).join('') + '</tr>'; i++; }
      out += `<table><thead><tr>${hd.map(c => `<th>${inline(c)}</th>`).join('')}</tr></thead><tbody>${body}</tbody></table>`; continue;
    }
    if (/^>\s?/.test(l)) { const b = []; while (i < L.length && /^>\s?/.test(L[i])) b.push(L[i++].replace(/^>\s?/, '')); out += `<blockquote>${inline(b.join(' '))}</blockquote>`; continue; }
    if (/^\s*([-*]|\d+\.)\s/.test(l)) { const ord = /^\s*\d+\./.test(l); const b = []; while (i < L.length && /^\s*([-*]|\d+\.)\s/.test(L[i])) { let item = L[i++].replace(/^\s*([-*]|\d+\.)\s/, ''); while (i < L.length && /^\s{2,}\S/.test(L[i]) && !/^\s*([-*]|\d+\.)\s/.test(L[i])) item += ' ' + L[i++].trim(); b.push(`<li>${inline(item)}</li>`); } out += ord ? `<ol>${b.join('')}</ol>` : `<ul>${b.join('')}</ul>`; continue; }
    if (/^---+\s*$/.test(l)) { out += '<hr>'; i++; continue; }
    if (!l.trim()) { i++; continue; }
    const b = []; while (i < L.length && L[i].trim() && !/^(#{1,6}\s|```|>|\s*\||\s*([-*]|\d+\.)\s|---+\s*$)/.test(L[i])) b.push(L[i++]);
    if (!b.length) b.push(L[i++]);
    out += `<p>${inline(b.join(' '))}</p>`;
  }
  return out;
}
function pageResearch(d) {
  const verdict = v => ({ PASS: 'PASS', FAIL: 'FAIL' }[v] || 'INFO');
  const ex = d.experiments.map(e => `<tr><td class="l sym">${esc(e.id)}</td><td class="l" style="max-width:320px">${esc(e.question)}</td><td class="l muted" style="max-width:360px">${esc(e.criterion)}</td>
    <td class="l">${pill(verdict(e.verdict))}<div class="muted" style="margin-top:4px">${esc(e.result)}</div>${e.note ? `<div class="faint" style="margin-top:4px">${esc(e.note)}</div>` : ''}</td>
    <td class="l"><a data-doc="${esc(e.file)}">${esc(String(e.file).replace(/^audit\//, ''))}</a></td></tr>`).join('');
  const gcols = d.gates.length ? ['file', 'k', 'horizon', 'n_windows', 'mean', 't', 't_crit', 'rank', 'verdict'].filter(c => c in d.gates[0]) : [];
  const gates = d.gates.map(r => `<tr>${gcols.map((c, i) => `<td class="${i === 0 || c === 'verdict' ? 'l' : ''}">${c === 'file' ? esc(String(r[c]).replace('audit/topk_gate/', '')) : typeof r[c] === 'number' ? (Number.isInteger(r[c]) ? r[c] : r[c].toFixed(4)) : esc(r[c])}</td>`).join('')}</tr>`).join('');
  const docSel = ui.doc && d.docs.includes(ui.doc) ? ui.doc : null;
  return `<div class="page-head"><h1>Research log</h1><p>Every experiment, its pre-registered criterion and its result, read from the audit files at load time. Failures are first-class: most results here are FAILs, and that is a legitimate outcome. Nothing here is a "winner".</p></div>
  <section class="sec" style="margin-top:24px"><div class="sech"><h2>Experiments</h2><span class="n">${d.experiments.length}</span><span class="src">audit/*.md</span></div>
    <div class="card tblwrap"><table class="t wrap"><thead><tr><th class="l">ID</th><th class="l">Question</th><th class="l">Criterion</th><th class="l">Result</th><th class="l">File</th></tr></thead><tbody>${ex}</tbody></table></div></section>
  <section class="sec"><div class="sech"><h2>Top-K gate results</h2><span class="n">${d.gates.length}</span><span class="src">audit/topk_gate/*.json</span></div>
    <div class="callout" style="margin:0 0 12px">Backtest gates over walk-forward windows, not the forward paper record. "Rank 1/21" with FAIL means it beat the random books but not the absolute bar. Short-leg information refers to names to avoid, which a long-only book cannot trade.</div>
    <div class="card tblwrap"><table class="t compact"><thead><tr>${gcols.map((c, i) => `<th class="${i === 0 || c === 'verdict' ? 'l' : ''}">${esc(c)}</th>`).join('')}</tr></thead><tbody>${gates}</tbody></table></div></section>
  <section class="sec"><div class="sech"><h2>Documents</h2><select data-act="doc" style="margin-left:12px"><option value="">Choose a document…</option>${d.docs.map(x => `<option ${x === docSel ? 'selected' : ''}>${esc(x)}</option>`).join('')}</select></div>
    <div class="card doc" id="docbody">${docSel ? '<div class="faint">Loading…</div>' : '<div class="faint">Pick a document to read it here.</div>'}</div></section>`;
}
async function loadDoc(name) {
  const box = $('#docbody'); if (!box) return;
  if (!name) { box.innerHTML = '<div class="faint">Pick a document to read it here.</div>'; return; }
  try { const r = await fetch('/api/doc?name=' + encodeURIComponent(name)); box.innerHTML = r.ok ? md(await r.text()) : '<div class="faint">Not found.</div>'; }
  catch (e) { box.innerHTML = `<div class="faint">${esc(e.message)}</div>`; }
}

// ── Warm-up ──────────────────────────────────────────────────────────────────
function pageWarmup(d) {
  const head = `<div class="page-head"><h1>Warm-up history</h1><p>The replay before the test clock started. <b>Context, not the test:</b> none of it counts toward the pre-registered statistic.</p></div>`;
  if (d.error) return head + `<div class="callout"><span>ⓘ</span><div>${esc(d.error)}. The replay files appear after the first recorded session.</div></div>`;
  const dates = [...new Set([...d.band.map(b => b[0]), ...d.lines.flatMap(l => l.points.map(q => q[0]))])].sort();
  const at = pts => { const m = new Map(pts); return dates.map(x => m.has(x) ? m.get(x) : null); };
  const bm = new Map(d.band.map(b => [b[0], b]));
  const colors = { signal: 'var(--c1)', equal_weight: 'var(--c2)' };
  const series = d.lines.map(l => ({ ys: at(l.points), color: colors[l.kind], width: l.kind === 'signal' ? 2.5 : 1.75, name: BOOKNAME[l.kind] }));
  if (d.band.length) series.push({ ys: dates.map(x => (bm.get(x) || [])[2] ?? null), color: 'var(--ink3)', dash: '2 3', width: 1.25, name: 'Random median' });
  const chart = dates.length ? plot({ n: dates.length, series, band: d.band.length ? { lo: dates.map(x => (bm.get(x) || [0, 0])[1]), hi: dates.map(x => (bm.get(x) || [0, 0, 0, 0])[3]) } : null,
    base: { y: 0 }, yfmt: v => srs(v, 0), height: 320, xlabels: [[0, dm(dates[0])], [100, dm(dates[dates.length - 1])]],
    tip: i => `<b>${wdm(dates[i])}</b><br>` + series.map(s => s.ys[i] == null ? '' : `${esc(s.name)}: ${srs(s.ys[i], 0)}`).filter(Boolean).join('<br>') }) : '<div class="empty">No NAV files.</div>';
  const w = d.warmup;
  return head + `<section class="sec" style="margin-top:24px"><div class="sech"><h2>P&amp;L on ${rs(d.notional, 0)} notional</h2><span class="src">${esc(d.source)}</span></div>
    <div class="card big">${chart}<div class="lg">${series.map(s => `<span><i style="background:${s.color}"></i>${esc(s.name)}</span>`).join('')}${d.band.length ? '<span><i style="background:var(--band);height:10px"></i>Random 10th–90th</span>' : ''}</div></div></section>
    ${w ? `<section class="card kpis" style="grid-template-columns:repeat(3,1fr)">${kpi('Algorithm, cumulative log return', (w.signal_cum >= 0 ? '+' : '−') + Math.abs(w.signal_cum).toFixed(4))}${kpi('Random books, mean', (w.null_mean >= 0 ? '+' : '−') + Math.abs(w.null_mean).toFixed(4))}${kpi('Warm-up rank', `${w.rank} of ${w.n}`, 'not forward evidence')}</section>` : ''}
    ${d.note ? `<div class="callout">${esc(d.note)}</div>` : ''}`;
}

// ── Kite login ───────────────────────────────────────────────────────────────
function pageKite(_, params) {
  const k = (STATUS && STATUS.kite) || { state: 'missing' };
  const lbl = { valid: 'Connected', expired: 'Expired', missing: 'Not logged in', error: 'Could not verify' }[k.state];
  const dot = { valid: 'ok', error: 'warn' }[k.state] || 'fail';
  const ok = params.get('ok'), err = params.get('err');
  const flash = ok ? `<div class="callout ok"><span>✓</span><div><b>Kite connected.</b> The token is valid until ${esc(wdm(ok))} ${esc(hm(ok))} IST.</div></div>` : err ? `<div class="callout fail"><span>!</span><div><b>Login failed.</b> ${esc(err)}</div></div>` : '';
  return `<div class="page-head"><h1>Kite login</h1><p>The live book reads prices from Zerodha Kite at 09:16 and through the day. Kite access tokens expire around 06:00 IST every morning, so log in once each trading day before 09:16. Only the login and profile endpoints are used; no orders are ever placed.</p></div>
  ${flash}
  <div class="kite">
    <section class="card big">
      <div class="kstate"><span class="dot ${dot} ${dot === 'fail' ? 'pulse' : ''}"></span>${lbl}</div>
      <div class="kgrid">
        <div><div class="k">Logged in</div><div class="v">${k.minted_at ? esc(wdm(k.minted_at) + ' ' + hm(k.minted_at)) : '—'}</div></div>
        <div><div class="k">Expires (approx.)</div><div class="v">${k.expires_at ? esc(wdm(k.expires_at) + ' ' + hm(k.expires_at)) : '—'}</div></div>
        <div><div class="k">Account</div><div class="v">${esc(k.broker || '—')}</div></div>
        <div><div class="k">Last check</div><div class="v">${k.error ? esc(k.error) : k.state === 'valid' ? 'profile OK' : '—'}</div></div>
      </div>
      <div style="margin-top:22px;display:flex;gap:10px;flex-wrap:wrap">
        ${LOGIN_URL ? `<a class="btn primary" href="${esc(LOGIN_URL)}">Log in to Kite</a>` : '<span class="callout fail" style="margin:0">KITE_API_KEY is not configured.</span>'}
        <button class="btn" data-act="kitecheck">Check now</button>
      </div>
      <p class="faint" style="font-size:12.5px;margin:14px 0 0;line-height:1.55">Kite sends you back here (redirect URL <span class="mono">http://127.0.0.1:8501/</span>) and the token is saved automatically.</p>
    </section>
    <section class="card big">
      <div style="font-weight:500;margin-bottom:10px">Or paste the redirect URL</div>
      <p class="muted" style="margin:0 0 12px;font-size:13px">If Kite sends you somewhere else, copy that page's address, or just the <span class="mono">request_token</span>, into this box.</p>
      <form data-form="paste" style="display:flex;flex-direction:column;gap:10px">
        <input type="password" name="text" autocomplete="off" placeholder="http://…?status=success&amp;request_token=…">
        <button class="btn" type="submit">Connect</button>
        <div id="pastemsg"></div>
      </form>
      <ol class="steps2" style="margin-top:18px;font-size:13px"><li>Click <b>Log in to Kite</b> and sign in on Zerodha's page.</li><li>You return here with a green "Kite connected".</li><li>The 09:16 trade and every price update use the saved token.</li></ol>
      <p class="faint" style="font-size:12px;line-height:1.55">The token is stored in <span class="mono">secrets/kite/access_token.json</span> (owner-only, gitignored) and is never shown.</p>
    </section>
  </div>`;
}

// ── render + poll loop ───────────────────────────────────────────────────────
function renderPage() {
  const r = route(), p = PAGES[r.name];
  const d = p.api ? DATA[r.name] : null;
  if (p.api && !d) { $('#page').innerHTML = '<div class="loading">Loading…</div>'; return; }
  const y = window.scrollY;
  try { $('#page').innerHTML = p.render(d || {}, r.params); }
  catch (e) { $('#page').innerHTML = `<div class="callout fail"><span>!</span><div>This page could not render: ${esc(e.message)}</div></div>`; console.error(e); }
  window.scrollTo(0, y);
  if (r.name === 'research' && ui.doc) loadDoc(ui.doc);
  document.title = baseTitle();
}

let pollTimer = null;
async function tick(forcePage) {
  clearTimeout(pollTimer);
  try {
    STATUS = await getJSON('/api/status'); OFFLINE = false;
    if (!LOGIN_URL) { try { LOGIN_URL = (await getJSON('/api/kite/login-url')).url; } catch { /* retry next tick */ } }
    const r = route(), p = PAGES[r.name];
    if (p.api && (forcePage || !DATA[r.name] || VER[r.name] !== STATUS.version)) {
      DATA[r.name] = await getJSON(p.api); VER[r.name] = STATUS.version; renderPage();
    } else if (!p.api || r.name === 'portfolio') renderPage();       // kite page reads STATUS; portfolio shows "x min ago"
    if (r.name !== 'portfolio' && !DATA.portfolio) { getJSON('/api/portfolio').then(v => { DATA.portfolio = v; VER.portfolio = STATUS.version; }).catch(() => {}); }
  } catch (e) { OFFLINE = true; console.warn(e); }
  renderChrome(); renderAttention();
  const s = (STATUS && STATUS.poll_s) || 60;
  pollTimer = setTimeout(tick, (OFFLINE ? 15 : s) * 1000);
}

// ── events (delegated) ───────────────────────────────────────────────────────
window.addEventListener('hashchange', () => {
  const r = route();
  if (r.name === 'kite' && (r.params.get('ok') || r.params.get('err'))) {
    if (r.params.get('ok')) FLASH = { id: 'kite-ok', level: 'ok', title: 'Kite connected', message: `Token valid until ${hm(r.params.get('ok'))} IST.`, action: null };
  }
  renderChrome(); renderPage(); tick();
});
document.addEventListener('visibilitychange', () => { if (!document.hidden) tick(); });
// A login completed in one tab refreshes every other open dashboard tab at once.
window.addEventListener('storage', e => { if (e.key === 'pb-kite-login') { delete VER[route().name]; tick(true); } });
const announceLogin = () => store.set('pb-kite-login', Date.now());
document.addEventListener('click', async e => {
  const t = e.target.closest('[data-sort],[data-psort],[data-range],[data-trade],[data-act],[data-doc]');
  if (!t) return;
  if (t.dataset.sort) { const k = t.dataset.sort; ui.sd = ui.sk === k ? -ui.sd : (k === 'symbol' || k === 'sector' ? 1 : -1); ui.sk = k; saveUI(); renderPage(); }
  else if (t.dataset.psort) { ui.ps = t.dataset.psort; saveUI(); renderPage(); }
  else if (t.dataset.range) { ui.range = t.dataset.range; saveUI(); renderPage(); }
  else if (t.dataset.trade) { ui.open[t.dataset.trade] = !ui.open[t.dataset.trade]; saveUI(); renderPage(); }
  else if (t.dataset.doc) { ui.doc = t.dataset.doc; saveUI(); location.hash = '#research'; renderPage(); $('#docbody') && $('#docbody').scrollIntoView({ behavior: 'smooth' }); }
  else if (t.dataset.act === 'reload') { Object.keys(VER).forEach(k => delete VER[k]); tick(true); }
  else if (t.dataset.act === 'theme') { const n = { auto: 'light', light: 'dark', dark: 'auto' }[store.get('pb-theme', 'auto')]; store.set('pb-theme', n); applyTheme(); renderChrome(); }
  else if (t.dataset.act === 'notify') { try { await Notification.requestPermission(); } catch { /* ignore */ } renderAttention(); }
  else if (t.dataset.act === 'dismiss') { FLASH = null; if (location.hash.includes('?')) history.replaceState(null, '', '#' + route().name); renderAttention(); renderPage(); }
  else if (t.dataset.act === 'kitecheck') { t.disabled = true; t.textContent = 'Checking…'; await tick(true); }
  else if (t.dataset.act === 'rerun') {
    t.disabled = true; t.textContent = 'Running… (up to a few minutes)';
    const r = await postJSON('/api/health/rerun');
    if (!r.ok) alert('Health check: ' + (r.error || 'failed'));
    delete VER.health; tick(true);
  }
});
document.addEventListener('change', e => {
  const t = e.target;
  if (t.dataset.act === 'overlays') { ui.overlays = t.checked; saveUI(); renderPage(); }
  if (t.dataset.act === 'doc') { ui.doc = t.value || null; saveUI(); loadDoc(ui.doc); }
});
document.addEventListener('submit', async e => {
  const f = e.target.closest('[data-form="paste"]'); if (!f) return;
  e.preventDefault();
  const msg = $('#pastemsg'); msg.innerHTML = '<span class="faint">Connecting…</span>';
  const r = await postJSON('/api/kite/token', { text: f.text.value });
  f.text.value = '';
  msg.innerHTML = r.ok ? `<div class="callout ok" style="margin:0"><span>✓</span><div>Kite connected. Valid until ${esc(hm(r.expires_at))} IST.</div></div>` : `<div class="callout fail" style="margin:0"><span>!</span><div>${esc(r.error)}</div></div>`;
  if (r.ok) { announceLogin(); tick(true); }
});

// boot
if (route().name === 'kite' && route().params.get('ok')) announceLogin();
if (route().name === 'kite' && route().params.get('ok')) FLASH = { id: 'kite-ok', level: 'ok', title: 'Kite connected', message: `Token valid until ${hm(route().params.get('ok'))} IST.`, action: null };
renderChrome();
tick(true);
