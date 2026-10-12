# ruff: noqa: E501, E702, E731  (layout-dense SVG generator)
"""Generate architecture_diagrams.html (diagram-design skill, default skin)."""
import math
from html import escape as E

PAPER, PAPER2, INK, MUTED, SOFT = "#f5f5f5", "#ececec", "#2d3142", "#4f5d75", "#5a6580"
ACCENT, ACCENT_T, LINK = "#bf4520", "rgba(191,69,32,0.08)", "#2e5aa8"
SANS, MONO, SERIF = "'Geist', system-ui, sans-serif", "'Geist Mono', ui-monospace, Menlo, monospace", "'Instrument Serif', 'Noto Serif', serif"

KIND = {  # fill, stroke, tag stroke, tag text
    "focal": (ACCENT_T, ACCENT, "rgba(191,69,32,0.50)", ACCENT),
    "backend": ("#ffffff", INK, "rgba(45,49,66,0.40)", INK),
    "store": ("rgba(45,49,66,0.05)", MUTED, "rgba(79,93,117,0.50)", MUTED),
    "external": ("rgba(45,49,66,0.03)", "rgba(45,49,66,0.30)", "rgba(45,49,66,0.22)", SOFT),
    "input": ("rgba(79,93,117,0.10)", SOFT, "rgba(90,101,128,0.40)", SOFT),
    "optional": ("rgba(45,49,66,0.02)", "rgba(45,49,66,0.20)", "rgba(45,49,66,0.22)", SOFT),
}


def r4(v):
    return int(math.ceil(v / 4.0) * 4)


def tagw(t):
    return r4(len(t) * 7 * 0.62 + 10)


def node(x, y, w, h, kind, tag, name, subs=(), dashed=False, name_dy=None):
    fill, stroke, ts, tt = KIND[kind]
    tw = tagw(tag)
    cx = x + w / 2
    ny = y + (name_dy if name_dy is not None else (h / 2 + 4 - 7 * len(subs) + (3 if len(subs) == 2 else 0)))
    da = ' stroke-dasharray="4,3"' if dashed or kind == "optional" else ""
    out = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{PAPER}"/>',
           f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{fill}" stroke="{stroke}" stroke-width="1"{da}/>',
           f'<rect x="{x+8}" y="{y+6}" width="{tw}" height="12" rx="2" fill="none" stroke="{ts}" stroke-width="0.8"/>',
           f'<text x="{x+8+tw/2}" y="{y+15}" fill="{tt}" font-size="7" font-family="{MONO}" text-anchor="middle" letter-spacing="0.08em">{E(tag)}</text>',
           f'<text x="{cx}" y="{ny}" fill="{INK}" font-size="12" font-weight="600" font-family="{SANS}" text-anchor="middle">{E(name)}</text>']
    for i, s in enumerate(subs):
        out.append(f'<text x="{cx}" y="{ny+16+i*13}" fill="{MUTED}" font-size="9" font-family="{MONO}" text-anchor="middle">{E(s)}</text>')
    return "\n".join(out)


def mk(color):
    return {MUTED: "arrow", ACCENT: "arrow-accent", LINK: "arrow-link"}[color]


def path(d, color=MUTED, dash=False, w=1.2):
    da = ' stroke-dasharray="5,4"' if dash else ""
    return f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{w}"{da} marker-end="url(#{mk(color)})"/>'


def label(cx, top, text, color=SOFT, anchor="middle"):
    """Mask rect top at `top`, 12 tall; text centred on cx (or starting at cx)."""
    w = r4(len(text) * 8 * 0.62 + 10)
    x = cx - w / 2 if anchor == "middle" else cx
    tx = cx if anchor == "middle" else cx + w / 2
    return (f'<rect x="{x}" y="{top}" width="{w}" height="12" rx="2" fill="{PAPER}"/>'
            f'<text x="{tx}" y="{top+9}" fill="{color}" font-size="8" font-family="{MONO}" text-anchor="middle" letter-spacing="0.06em">{E(text)}</text>')


def legend(y, items, vw):
    out = [f'<line x1="30" y1="{y-8}" x2="{vw-30}" y2="{y-8}" stroke="rgba(45,49,66,0.10)" stroke-width="0.8"/>',
           f'<text x="30" y="{y+12}" fill="{MUTED}" font-size="8" font-family="{MONO}" letter-spacing="0.14em">LEGEND</text>']
    x = 110
    for kind, text in items:
        if kind.startswith("arrow"):
            col = {"arrow": MUTED, "arrow-link": LINK, "arrow-accent": ACCENT, "arrow-dash": MUTED}[kind]
            da = ' stroke-dasharray="5,4"' if kind == "arrow-dash" else ""
            m = "arrow" if kind == "arrow-dash" else kind
            out.append(f'<line x1="{x}" y1="{y+8}" x2="{x+18}" y2="{y+8}" stroke="{col}" stroke-width="1.2"{da} marker-end="url(#{m})"/>')
        elif kind == "dot-accent":
            out.append(f'<circle cx="{x+8}" cy="{y+8}" r="5" fill="{ACCENT}"/>')
        elif kind == "dot":
            out.append(f'<circle cx="{x+8}" cy="{y+8}" r="4" fill="{INK}"/>')
        elif kind == "diamond":
            out.append(f'<polygon points="{x+8},{y+1} {x+16},{y+8} {x+8},{y+15} {x},{y+8}" fill="#ffffff" stroke="{INK}" stroke-width="1"/>')
        elif kind == "zone":
            out.append(f'<rect x="{x}" y="{y+2}" width="18" height="12" rx="2" fill="rgba(45,49,66,0.02)" stroke="rgba(45,49,66,0.20)" stroke-dasharray="4,4"/>')
        else:
            f, s, _, _ = KIND[kind]
            da = ' stroke-dasharray="4,3"' if kind == "optional" else ""
            out.append(f'<rect x="{x}" y="{y+2}" width="18" height="12" rx="2" fill="{f}" stroke="{s}" stroke-width="1"{da}/>')
        out.append(f'<text x="{x+26}" y="{y+12}" fill="{MUTED}" font-size="9" font-family="{SANS}">{E(text)}</text>')
        x += 26 + r4(len(text) * 9 * 0.6) + 28
    return "\n".join(out)


def svg(slug, vw, vh, title, desc, body):
    return f'''<div class="diagram-container"><svg viewBox="0 0 {vw} {vh}" style="min-width:{vw}px" xmlns="http://www.w3.org/2000/svg" role="img" aria-labelledby="{slug}-title {slug}-desc">
<title id="{slug}-title">{E(title)}</title>
<desc id="{slug}-desc">{E(desc)}</desc>
<defs>
<marker id="{slug}-arrow" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto"><polygon points="0 0, 8 3, 0 6" fill="{MUTED}"/></marker>
<marker id="{slug}-arrow-accent" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto"><polygon points="0 0, 8 3, 0 6" fill="{ACCENT}"/></marker>
<marker id="{slug}-arrow-link" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto"><polygon points="0 0, 8 3, 0 6" fill="{LINK}"/></marker>
</defs>
<rect width="100%" height="100%" fill="{PAPER}"/>
{body.replace("url(#arrow", f"url(#{slug}-arrow")}
</svg></div>'''


def callout(tx, ty, lines, lx, ly, anchor="start", color=INK, lead="rgba(45,49,66,0.40)", d=None):
    out = [f'<text x="{tx}" y="{ty+i*17}" fill="{color}" font-size="14" font-style="italic" font-family="{SERIF}" text-anchor="{anchor}">{E(t)}</text>' for i, t in enumerate(lines)]
    out.append(f'<path d="{d}" fill="none" stroke="{lead}" stroke-width="1" stroke-dasharray="4,3"/>')
    out.append(f'<circle cx="{lx}" cy="{ly}" r="2" fill="{color}"/>')
    return "\n".join(out)


# ─────────────────────────── 1. System overview ────────────────────────────
def d1():
    W, H = 1280, 500
    C = [30, 290, 550, 810, 1070]       # column x, box w 180, gap 80
    bw, bh = 180, 80
    r1, r2 = 72, 272
    y1, y2 = r1 + 40, r2 + 40
    b = []
    # zones
    b.append('<rect x="270" y="40" width="1000" height="132" rx="8" fill="rgba(45,49,66,0.02)" stroke="rgba(45,49,66,0.10)" stroke-width="0.8"/>')
    b.append(f'<rect x="280" y="44" width="196" height="12" rx="2" fill="{PAPER}"/><text x="378" y="53" fill="{SOFT}" font-size="7" font-family="{MONO}" text-anchor="middle" letter-spacing="0.14em">NIGHTLY · 21:00 IST · DOCKER</text>')
    b.append('<rect x="530" y="240" width="480" height="132" rx="8" fill="rgba(45,49,66,0.02)" stroke="rgba(45,49,66,0.10)" stroke-width="0.8"/>')
    b.append(f'<rect x="540" y="244" width="200" height="12" rx="2" fill="{PAPER}"/><text x="640" y="253" fill="{SOFT}" font-size="7" font-family="{MONO}" text-anchor="middle" letter-spacing="0.14em">MARKET HOURS · 09:16–15:35</text>')
    # arrows (row 1)
    b.append(path(f"M {C[0]+bw},{y1} H {C[1]-1}"));             b.append(label((C[0]+bw+C[1])/2, y1-20, "DAILY FILES"))
    b.append(path(f"M {C[1]+bw},{y1} H {C[2]-1}"));             b.append(label((C[1]+bw+C[2])/2, y1-20, "FEATURES"))
    b.append(path(f"M {C[2]+bw},{y1} H {C[3]-1}"));             b.append(label((C[2]+bw+C[3])/2, y1-20, "SCORES"))
    b.append(path(f"M {C[3]+bw},{y1} H {C[4]-1}"));             b.append(label((C[3]+bw+C[4])/2, y1-20, "TARGETS"))
    # verticals
    xv = C[3] + bw / 2
    b.append(path(f"M {xv},{r1+bh} V {r2-1}"));                b.append(label(xv + 8, (r1+bh+r2)/2 - 6, "09:16 TRADE", anchor="start"))
    xr = C[4] + bw / 2
    b.append(path(f"M {xr},{r1+bh} V {r2-1}"));                b.append(label(xr + 8, (r1+bh+r2)/2 - 6, "APPEND", anchor="start"))
    # row 2
    b.append(path(f"M {C[1]+bw},{y2} H {C[2]-1}", LINK));      b.append(label((C[1]+bw+C[2])/2, y2-20, "DAILY LOGIN", LINK))
    b.append(path(f"M {C[2]+bw},{y2} H {C[3]-1}", LINK));      b.append(label((C[2]+bw+C[3])/2, y2-20, "LIVE QUOTES", LINK))
    b.append(path(f"M {C[3]+bw},{y2} H {C[4]-1}"));             b.append(label((C[3]+bw+C[4])/2, y2-20, "FILLS, MARKS"))
    # records -> dashboard, under row 2
    yb = r2 + bh + 48
    xd = C[1] + bw / 2
    b.append(path(f"M {xr},{r2+bh} V {yb-8} Q {xr},{yb} {xr-8},{yb} H {xd+8} Q {xd},{yb} {xd},{yb-8} V {r2+bh+1}", MUTED, dash=True, w=1))
    b.append(label((C[2]+C[3]+bw)/2, yb - 20, "READ-ONLY"))
    # nodes
    b.append(node(C[0], r1, bw, bh, "external", "EXT", "NSE archives", ["bhavcopy · index · CA", "no API key needed"]))
    b.append(node(C[1], r1, bw, bh, "store", "PANEL", "Feature panel", ["504 PIT names", "15 price features"]))
    b.append(node(C[2], r1, bw, bh, "focal", "MODEL", "Signal model", ["TCN · r̂ 5d / 20d", "frozen r4_pit_long"]))
    b.append(node(C[3], r1, bw, bh, "backend", "RULES", "Allocator + vol stop", ["top-30 · inverse-vol", "caps · band · budget"]))
    b.append(node(C[4], r1, bw, bh, "backend", "ENV", "End-of-day replay", ["P2: 85 books", "P3: adaptive shadow"]))
    b.append(node(C[1], r2, bw, bh, "input", "UI", "Paper Book dashboard", ["server.py · :8501", "stdlib + plain JS"]))
    b.append(node(C[2], r2, bw, bh, "external", "EXT", "Kite Connect", ["read-only: quote", "token expires 06:00"]))
    b.append(node(C[3], r2, bw, bh, "focal", "BOOK", "Live paper book", ["₹1,00,000 · 29 stocks", "live_paper.py"]))
    b.append(node(C[4], r2, bw, bh, "store", "STORE", "Audit records", ["record.jsonl · ledger", "marks · MLflow"]))
    b.append(legend(H - 44, [("focal", "Focal"), ("backend", "Logic"), ("store", "Stored data"), ("external", "External service"),
                             ("input", "User interface"), ("arrow", "Data flow"), ("arrow-link", "Kite API call"), ("arrow-dash", "Passive read")], W))
    return svg("overview", W, H, "trading-bot-01 system overview",
               "NSE daily files become a feature panel, a frozen neural signal ranks stocks, a rule-based allocator turns ranks into target weights, and those targets drive both a nightly replay against random books and a live paper portfolio priced by Kite, all read by a local dashboard.", "\n".join(b))


# ─────────────────────────── 2. The 09:16 decision ─────────────────────────
def d2():
    W, H = 1000, 600
    cx = 400
    b = []

    def diamond(cy, l1, l2, focal=False):
        f, s = (ACCENT_T, ACCENT) if focal else ("#ffffff", INK)
        pts = f"{cx},{cy-44} {cx+92},{cy} {cx},{cy+44} {cx-92},{cy}"
        return (f'<polygon points="{pts}" fill="{PAPER}"/><polygon points="{pts}" fill="{f}" stroke="{s}" stroke-width="1"/>'
                f'<text x="{cx}" y="{cy-2}" fill="{INK}" font-size="12" font-weight="600" font-family="{SANS}" text-anchor="middle">{E(l1)}</text>'
                f'<text x="{cx}" y="{cy+13}" fill="{INK}" font-size="12" font-weight="600" font-family="{SANS}" text-anchor="middle">{E(l2)}</text>')

    def oval(y, w, t, s):
        return (f'<rect x="{cx-w/2}" y="{y}" width="{w}" height="48" rx="20" fill="{PAPER}"/>'
                f'<rect x="{cx-w/2}" y="{y}" width="{w}" height="48" rx="20" fill="rgba(79,93,117,0.10)" stroke="{SOFT}" stroke-width="1"/>'
                f'<text x="{cx}" y="{y+21}" fill="{INK}" font-size="12" font-weight="600" font-family="{SANS}" text-anchor="middle">{E(t)}</text>'
                f'<text x="{cx}" y="{y+36}" fill="{MUTED}" font-size="9" font-family="{MONO}" text-anchor="middle">{E(s)}</text>')

    D = [144, 272, 400]
    ax, aw, ah = 600, 240, 72
    # arrows
    b.append(path(f"M {cx},{72} V {D[0]-45}"))
    b.append(path(f"M {cx},{D[0]+44} V {D[1]-45}")); b.append(label(cx + 8, (D[0]+44+D[1]-44)/2 - 6, "YES", anchor="start"))
    b.append(path(f"M {cx},{D[1]+44} V {D[2]-45}")); b.append(label(cx + 8, (D[1]+44+D[2]-44)/2 - 6, "NO", anchor="start"))
    b.append(path(f"M {cx},{D[2]+44} V {508-1}"));   b.append(label(cx + 8, (D[2]+44+508)/2 - 6, "NO", anchor="start"))
    for cy, t, col in ((D[0], "NO", MUTED), (D[1], "YES", ACCENT), (D[2], "YES", MUTED)):
        b.append(path(f"M {cx+92},{cy} H {ax-1}", col))
        b.append(label((cx + 92 + ax) / 2, cy - 20, t))
    # nodes
    b.append(oval(24, 200, "09:16 IST, weekday", "scripts/live_paper.py trade"))
    b.append(diamond(D[0], "Book already", "funded?"))
    b.append(diamond(D[1], "First session", "of the month?", focal=True))
    b.append(diamond(D[2], "Stop-loss sales", "pending?"))
    b.append(node(ax, D[0] - ah / 2, aw, ah, "backend", "ONCE", "First funding", name_dy=34, subs=["invest ₹1,00,000 in full", "whole shares, top-up leftovers"]))
    b.append(node(ax, D[1] - ah / 2, aw, ah, "backend", "MONTHLY", "Rebalance to new top-30", name_dy=34, subs=["≤30% turnover · 1% band", "stopped names stay out"]))
    b.append(node(ax, D[2] - ah / 2, aw, ah, "backend", "STOP", "Sell stopped stocks", name_dy=34, subs=["full exit at the open", "21-session re-entry ban"]))
    b.append(oval(508, 200, "Hold", "nothing traded today"))
    # callouts (left margin)
    b.append(callout(40, 40, ["Needs today's Kite token;", "without it the run refuses", "and writes nothing."], 300, 48,
                     d="M 214 46 Q 260 46 298 48"))
    b.append(callout(40, 388, ["Stops are set at the", "15:35 close: a close", "below entry by more", "than 1×vol×√20 (5–30%)."], 308, 400,
                     d="M 214 394 Q 260 400 306 400"))
    b.append(legend(H - 30, [("input", "Start / end"), ("diamond", "Decision"), ("backend", "Action"), ("focal", "Most consequential")], W))
    return svg("decision", W, H, "The 09:16 trade decision",
               "Each weekday at 09:16 the live book checks, in order, whether it has been funded, whether today is the first session of a month, and whether any stop-loss sales are pending, and either funds, rebalances, sells stopped stocks or holds.", "\n".join(b))


# ─────────────────────────── 3. A trading day ──────────────────────────────
def d3():
    W, H = 1280, 330
    x0, x1, t0, t1 = 60, 1220, 6.0, 22.0
    X = lambda h: x0 + (h - t0) / (t1 - t0) * (x1 - x0)
    by = 170
    b = [f'<line x1="{x0}" y1="{by}" x2="{x1}" y2="{by}" stroke="{INK}" stroke-width="1"/>']
    for hh in range(6, 23, 2):
        x = X(hh)
        b.append(f'<line x1="{x:.1f}" y1="{by-4}" x2="{x:.1f}" y2="{by+4}" stroke="{INK}" stroke-width="1"/>')
        b.append(f'<text x="{x:.1f}" y="{by+18}" fill="{SOFT}" font-size="8" font-family="{MONO}" text-anchor="middle" letter-spacing="0.06em">{hh:02d}:00</text>')
    # market-hours band and marks range
    b.append(f'<rect x="{X(9.25):.1f}" y="{by-10}" width="{X(15.5)-X(9.25):.1f}" height="20" fill="rgba(45,49,66,0.05)"/>')
    for m in range(0, 26):
        h = 9.5 + m * 0.25
        if h > 15.75:
            break
        b.append(f'<line x1="{X(h):.1f}" y1="{by-6}" x2="{X(h):.1f}" y2="{by}" stroke="{MUTED}" stroke-width="0.8"/>')
    b.append(f'<text x="{(X(9.25)+X(15.5))/2:.1f}" y="{by+34}" fill="{MUTED}" font-size="8" font-family="{MONO}" text-anchor="middle" letter-spacing="0.06em">NSE OPEN 09:15–15:30 · PRICE UPDATE EVERY 15 MIN</text>')

    def ev(h, lines, tier, major=False, anchor="middle"):
        x = X(h)
        r, col = (6, ACCENT) if major else (4, INK)
        ty = {1: 52, 2: 100, -1: 236, -2: 284}[tier]
        y_line_end = ty + 6 + 13 * (len(lines) - 1) if tier > 0 else ty - 14
        out = [f'<line x1="{x:.1f}" y1="{by}" x2="{x:.1f}" y2="{y_line_end if tier>0 else ty-14}" stroke="rgba(45,49,66,0.30)" stroke-width="1"/>' if tier > 0 else
               f'<line x1="{x:.1f}" y1="{by}" x2="{x:.1f}" y2="{ty-14}" stroke="rgba(45,49,66,0.30)" stroke-width="1"/>',
               f'<circle cx="{x:.1f}" cy="{by}" r="{r}" fill="{col}"/>']
        if tier > 0:
            out[0] = f'<line x1="{x:.1f}" y1="{ty+8+13*(len(lines)-1)}" x2="{x:.1f}" y2="{by}" stroke="rgba(45,49,66,0.30)" stroke-width="1"/>'
        tx = x + (6 if anchor == "start" else -6 if anchor == "end" else 0)
        for i, t in enumerate(lines):
            if i == 0:
                out.append(f'<text x="{tx:.1f}" y="{ty+i*13}" fill="{ACCENT if major else INK}" font-size="12" font-weight="600" font-family="{SANS}" text-anchor="{anchor}">{E(t)}</text>')
            else:
                out.append(f'<text x="{tx:.1f}" y="{ty+i*13}" fill="{MUTED}" font-size="9" font-family="{MONO}" text-anchor="{anchor}">{E(t)}</text>')
        return "\n".join(out)

    b.append(ev(6.0, ["06:00 token expires", "Kite needs a new login"], 2, anchor="start"))
    b.append(ev(7.5, ["07:30 catch-up run", "+ 07:50 health check"], -1, anchor="start"))
    b.append(ev(8.5, ["Log in to Kite", "dashboard · before 09:16"], 1, anchor="end"))
    b.append(ev(9 + 16 / 60, ["09:16 trade", "fund · rebalance · stops · hold"], 2, major=True, anchor="start"))
    b.append(ev(15 + 35 / 60, ["15:35 close mark", "sets stop levels"], -2, anchor="end"))
    b.append(ev(21.0, ["21:00 nightly run", "data · signal · 85 books · record"], 1, major=True, anchor="end"))
    b.append(ev(21 + 25 / 60, ["21:25 host refit", "eve of 1st / 16th only"], -1, anchor="end"))
    b.append(ev(21 + 40 / 60, ["21:40 adaptive", "P3 replay"], -2, anchor="end"))
    return svg("day", W, H, "One trading day, IST",
               "A weekday from the 06:00 Kite token expiry through the morning catch-up and login, the 09:16 trade, price updates every 15 minutes, the 15:35 stop check, and the 21:00 nightly run, refit and adaptive replay.", "\n".join(b))


# ─────────────────────────── 4. Model lineage ──────────────────────────────
def d4():
    W, H = 1280, 470
    b = []
    root = (460, 32, 240, 56)
    br = [(40, 152, 240, 64), (400, 152, 240, 64), (880, 152, 240, 64)]
    lw, lh, ly = 200, 92, 288
    lx = [60, 300, 540, 780, 1020]
    rc = root[0] + root[2] / 2
    bc = [x + w / 2 for x, _, w, _ in br]
    lc = [x + lw / 2 for x in lx]
    bus1 = 120
    b.append(f'<path d="M {rc},{root[1]+root[3]} V {bus1}" fill="none" stroke="{MUTED}" stroke-width="1"/>')
    b.append(f'<path d="M {bc[0]},{bus1} H {bc[2]}" fill="none" stroke="{MUTED}" stroke-width="1"/>')
    for c in bc:
        b.append(f'<path d="M {c},{bus1} V {151}" fill="none" stroke="{MUTED}" stroke-width="1" marker-end="url(#arrow)"/>')
    bus2 = 252
    groups = [(0, [0]), (1, [1, 2]), (2, [3, 4])]
    for bi, leaves in groups:
        b.append(f'<path d="M {bc[bi]},{216} V {bus2}" fill="none" stroke="{MUTED}" stroke-width="1"/>')
        if len(leaves) > 1:
            b.append(f'<path d="M {lc[leaves[0]]},{bus2} H {lc[leaves[-1]]}" fill="none" stroke="{MUTED}" stroke-width="1"/>')
        for li in leaves:
            b.append(f'<path d="M {lc[li]},{bus2} V {ly-1}" fill="none" stroke="{MUTED}" stroke-width="1" marker-end="url(#arrow)"/>')
    b.append(node(*root, "input", "ROOT", "Decision layers", ["what chooses the stocks"]))
    b.append(node(*br[0], "backend", "M5–M7", "End-to-end RL", ["PPO picks weights directly"]))
    b.append(node(*br[1], "backend", "R4", "Supervised signal", ["predicts 5d / 20d returns"]))
    b.append(node(*br[2], "backend", "R5 / R6", "Allocator layer", ["turns scores into weights"]))
    leaves = [
        ("optional", "RETIRED", "PPO actor-critic", ["MLP · GNN · FiLM · aux head", "M5 gate never passed"]),
        ("focal", "IN USE", "TCN r4_pit_long", ["363k params · frozen", "rank IC pass; top-30 = random"]),
        ("store", "SHADOW", "Adaptive refits (P3)", ["same TCN, refit 1st & 16th", "read once, Oct 2028"]),
        ("backend", "IN USE", "Deterministic allocator", ["no learned parameters", "R5 gate FAIL on PIT"]),
        ("optional", "REJECTED", "PPO over allocator", ["17-dim Beta policy", "won in-sample, lost OOS"]),
    ]
    for x, (k, tag, name, subs) in zip(lx, leaves):
        b.append(node(x, ly, lw, lh, k, tag, name, subs))
    b.append(legend(H - 40, [("focal", "In use: drives the live book"), ("backend", "In use / branch"), ("store", "Shadow experiment"),
                             ("optional", "Built, not used")], W))
    return svg("lineage", W, H, "Models and RL agents: what was built, what runs",
               "Three decision layers were built: an end-to-end PPO agent that never passed its gate, a supervised TCN signal now in use with an adaptive refitting shadow, and an allocator layer where fixed rules are in use and a PPO tuner was rejected out of sample.", "\n".join(b))


# ─────────────────────────── 5. Deployment ─────────────────────────────────
def d5():
    W, H = 1280, 560
    b = []

    def zone(x, y, w, h, text, lw):
        return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="rgba(45,49,66,0.02)" stroke="rgba(45,49,66,0.20)" stroke-width="1" stroke-dasharray="4,4"/>'
                f'<rect x="{x+12}" y="{y-6}" width="{lw}" height="12" rx="2" fill="{PAPER}"/>'
                f'<text x="{x+16}" y="{y+3}" fill="{SOFT}" font-size="8" font-family="{MONO}" letter-spacing="0.14em">{E(text)}</text>')

    def chip(x, y, w, name, ver):
        return (f'<rect x="{x}" y="{y}" width="{w}" height="24" rx="4" fill="rgba(45,49,66,0.05)" stroke="{MUTED}" stroke-width="0.8"/>'
                f'<text x="{x+8}" y="{y+16}" fill="{INK}" font-size="12" font-family="{SANS}">{E(name)}</text>'
                f'<text x="{x+w-8}" y="{y+16}" fill="{MUTED}" font-size="9" font-family="{MONO}" text-anchor="end">{E(ver)}</text>')

    b.append(zone(30, 40, 860, 440, "MAC MINI M4 · macOS · FILEVAULT ON", 296))
    b.append(zone(300, 80, 560, 380, "DOCKER DESKTOP VM · 9 GB / 6 CPU", 268))
    b.append(zone(940, 40, 310, 440, "INTERNET", 84))
    # nodes geometry
    host = (60, 120, 200, 140)
    paper = (340, 120, 240, 172)
    dash = (340, 340, 240, 100)
    mlf = (660, 120, 180, 100)
    nse = (980, 120, 240, 76)
    kite = (980, 340, 240, 76)
    # paths
    b.append(path(f"M {paper[0]+paper[2]},{150} H {mlf[0]-1}"));                         b.append(label(620, 130, "HTTP:5000"))
    b.append(path(f"M {paper[0]+paper[2]},{250} H {900-8} Q 900,250 900,{242} V {170} Q 900,158 908,158 H {nse[0]-1}", LINK))
    b.append(label(940, 138, "HTTPS:443", LINK))
    b.append(path(f"M {paper[0]+paper[2]},{276} H {920-8} Q 920,276 920,{284} V {362} Q 920,370 928,370 H {kite[0]-1}", LINK))
    b.append(label(940, 350, "HTTPS:443", LINK))
    b.append(path(f"M {dash[0]+dash[2]},{400} H {kite[0]-1}", LINK));                   b.append(label(780, 380, "LOGIN EXCHANGE", LINK))
    b.append(path(f"M {paper[0]+120},{paper[1]+paper[3]} V {dash[1]-1}", MUTED, dash=True, w=1))
    b.append(label(paper[0] + 128, 310, "FILES · RO", anchor="start"))
    b.append(path(f"M {host[0]+host[2]},{190} H {paper[0]-1}", MUTED, dash=True, w=1))
    b.append(label(300, 170, "REFITS.JSONL"))
    # nodes
    b.append(node(*host, "optional", "LAUNCHD", "Host refit job", ["Apple GPU (MPS)", "not loaded yet"], name_dy=40))
    b.append(chip(host[0] + 12, host[1] + 92, 176, "refit_adaptive", "py 3.12"))
    b.append(node(*paper, "focal", "CONTAINER", "paper", ["supercronic schedule"], name_dy=40))
    b.append(chip(paper[0] + 12, paper[1] + 72, 216, "supercronic", "v0.2.33"))
    b.append(chip(paper[0] + 12, paper[1] + 104, 216, "kiteconnect", "5.2.1"))
    b.append(chip(paper[0] + 12, paper[1] + 136, 216, "python-slim", "3.12"))
    b.append(node(*dash, "backend", "CONTAINER", "dashboard", ["127.0.0.1:8501 · 20 MB"], name_dy=40))
    b.append(chip(dash[0] + 12, dash[1] + 66, 216, "server.py", "stdlib 3.12"))
    b.append(node(*mlf, "backend", "CONTAINER", "mlflow", ["127.0.0.1:5555"], name_dy=40))
    b.append(chip(mlf[0] + 12, mlf[1] + 64, 156, "mlflow", "3.11.1"))
    b.append(node(*nse, "external", "MANAGED", "NSE archives", ["nsearchives.nseindia.com", "bhavcopy, index, CA"]))
    b.append(node(*kite, "external", "MANAGED", "Kite Connect", ["api.kite.trade", "paid subscription"]))
    b.append(callout(60, 340, ["After any restart nothing", "runs until someone logs in:", "the 2–11 Oct outage."], 340, 290,
                     d="M 180 372 Q 260 372 338 292", color=ACCENT, lead="rgba(191,69,32,0.50)"))
    b.append(legend(H - 40, [("focal", "Single point of failure"), ("backend", "Container"), ("optional", "Inactive"), ("external", "External service"),
                             ("zone", "Boundary"), ("arrow-link", "Crosses to internet"), ("arrow-dash", "Shared files")], W))
    return svg("deploy", W, H, "Where trading-bot-01 runs",
               "Everything runs on one Mac mini: three containers in the Docker VM, with the paper container scheduling every job and calling NSE and Kite over HTTPS, and an inactive host job for GPU refits.", "\n".join(b))


CARDS = {
"overview": [("coral", "Signal model", ["TCN encoder over 60 days × 15 features for each of 504 stocks", "Outputs predicted 5-day and 20-day returns per stock", "Frozen: last trained on data to Jan 2022; refit only on 1 April", "Ranks stocks well overall, but its top 30 do no better than random picks"]),
             ("ink", "Allocator and risk", ["Top 30 by predicted 20-day return, weighted by 1/volatility", "Caps: 10% per stock, 25% per sector; 1% no-trade band", "30% turnover per rebalance (lifted for the first funding)", "Volatility stop: exit below entry by 1×vol×√20, 5–30%"]),
             ("muted", "Two ways it is run", ["Live book: ₹1L, filled at the 09:16 Kite quote, marked every 15 min", "Experiment (P2): same rules replayed on NSE closes vs 20 random books", "Adaptive shadow (P3): refit + rebalance twice a month, vs its own randoms", "No real orders anywhere; Kite is read-only"])],
"lineage": [("muted", "End-to-end PPO (M5–M7)", ["Actor-critic reads the whole market tensor and outputs weights", "Encoders: MLP, GNN; FiLM regime conditioning; auxiliary return head", "Validation Sharpe was anti-correlated with test (−0.86)", "Its M5 gate never passed, so nothing built on it is used"]),
            ("coral", "Supervised signal (R4)", ["TCN trained walk-forward, 13 windows, 2005–2024", "Rank IC gate passed: 20d +0.028, t 4.19", "But top-30 ranks 12th of 21 against random books", "Information sits in the bottom tail a long-only book can't use"]),
            ("ink", "PPO over the allocator (R6)", ["Steps once a month; observes 42 allocator-state numbers", "Acts on 17 dims (K, budget, cash floor, 14 sector tilts)", "+1.5 CAGR points in sample, −2 to −3 points out of sample", "Diagnosis: 17 parameters vs ~4 independent 2-year periods"])],
}


def cards(key):
    out = ['<div class="cards">']
    for dot, h, items in CARDS[key]:
        out.append(f'<div class="card"><div class="card-header"><span class="card-dot {dot}"></span><h3>{E(h)}</h3></div><ul>'
                   + "".join(f"<li>{E(i)}</li>" for i in items) + "</ul></div>")
    out.append("</div>")
    return "\n".join(out)


SECTIONS = [
    ("01 · System", "How a day's data becomes a portfolio", "Nightly NSE files feed a frozen neural signal and a rule-based allocator. The same targets drive the live ₹1L book at 09:16 and an end-of-day replay against random books.", d1, "overview"),
    ("02 · Decision", "What happens at 09:16", "One run per weekday, idempotent: retries at 09:20 and 09:30 change nothing once a day's trade is written.", d2, None),
    ("03 · Schedule", "One trading day, IST", "Spacing is to scale. The 21:25 refit only fires on the evening before a 1st or 16th rebalance.", d3, None),
    ("04 · Models", "Every model and RL agent, and its status", "Reinforcement learning was built twice and neither version earned its place. The supervised signal is what trades today.", d4, "lineage"),
    ("05 · Deployment", "Where it runs", "One Mac mini, one Docker VM. Postgres also runs but nothing in the daily loop writes to it, so it is left off.", d5, None),
]

CSS = """*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{--paper:#f5f5f5;--ink:#2d3142;--muted:#4f5d75;--soft:#5a6580;--rule:rgba(45,49,66,0.12);--accent:#bf4520;--link:#2e5aa8;
--sans:'Geist',system-ui,sans-serif;--serif:'Instrument Serif','Noto Serif',serif;--mono:'Geist Mono',ui-monospace,Menlo,monospace}
body{font-family:var(--sans);background:var(--paper);color:var(--ink);padding:3rem 1rem}
.container{max-width:1300px;margin:0 auto}
.header{margin-bottom:2.5rem}
.eyebrow{font-family:var(--mono);font-size:.66rem;font-weight:500;letter-spacing:.18em;text-transform:uppercase;color:var(--muted);margin-bottom:.75rem}
h1{font-family:var(--serif);font-size:clamp(1.75rem,3vw + 1rem,2.6rem);font-weight:400;letter-spacing:-.02em;line-height:1.1;margin-bottom:.6rem}
h2{font-family:var(--serif);font-size:1.75rem;font-weight:400;letter-spacing:-.01em;line-height:1.15;margin-bottom:.4rem}
.subtitle{font-size:1rem;line-height:1.55;color:var(--muted);max-width:70ch}
section{margin-top:3.5rem;padding-top:2rem;border-top:1px solid rgba(45,49,66,0.10)}
section .subtitle{margin-bottom:1.5rem}
.diagram-container{width:100%;overflow-x:auto}
svg{width:100%;display:block}
@media print{.diagram-container{overflow-x:visible}svg{min-width:0!important}}
.cards{display:grid;grid-template-columns:1.1fr 1fr .9fr;gap:1rem;margin-top:1.5rem}
@media (max-width:900px){.cards{grid-template-columns:1fr}}
.card{background:#fff;border-radius:6px;border:1px solid var(--rule);padding:1.25rem}
.card-header{display:flex;align-items:center;gap:.6rem;margin-bottom:.875rem;padding-bottom:.875rem;border-bottom:1px solid rgba(45,49,66,0.08)}
.card-dot{width:7px;height:7px;border-radius:50%;flex-shrink:0}
.card-dot.ink{background:var(--ink)}.card-dot.muted{background:var(--muted)}.card-dot.coral{background:var(--accent)}
.card h3{font-size:.875rem;font-weight:600}
.card ul{list-style:none;color:var(--muted);font-size:.8125rem;line-height:1.55}
.card li{margin-bottom:.3rem;padding-left:.875rem;position:relative}
.card li::before{content:'—';position:absolute;left:0;color:rgba(45,49,66,0.25);font-size:.75rem}
.footer{margin-top:3rem;padding-top:1.5rem;border-top:1px solid rgba(45,49,66,0.10);font-family:var(--mono);font-size:.72rem;letter-spacing:.06em;color:var(--soft);display:flex;justify-content:space-between;flex-wrap:wrap;gap:.5rem}"""

FONTS = '<link href="https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Geist:wght@400;500;600&family=Geist+Mono:wght@400;500;600&family=Noto+Serif:ital@0;1&display=swap" rel="stylesheet">'

parts = []
for eb, h, sub, fn, ck in SECTIONS:
    parts.append(f'<section><p class="eyebrow">{E(eb)}</p><h2>{E(h)}</h2><p class="subtitle">{E(sub)}</p>{fn()}{cards(ck) if ck else ""}</section>')

html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>trading-bot-01 Architecture</title>{FONTS}<style>{CSS}</style></head>
<body><div class="container">
<div class="header"><p class="eyebrow">Architecture · 12 Oct 2026</p><h1>trading-bot-01</h1>
<p class="subtitle">A daily, long-only paper-trading system for NSE stocks: what feeds it, how it decides, which models and RL agents were tried, and where it runs. Paper only; no real orders are ever placed.</p></div>
{''.join(parts)}
<div class="footer"><span>trading-bot-01 · architecture · drawn from the repository on 12 Oct 2026</span><span>sources: ARCHITECTURE.md · PROGRESS.md · audit/P2, P3 · docker/paper.crontab</span></div>
</div></body></html>"""

out = "/Users/jash/storage/trading-bot-01/architecture_diagrams.html"
open(out, "w").write(html)
print("wrote", out, len(html), "bytes")
