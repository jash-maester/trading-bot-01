"""Forward paper-trading dashboard (audit/P2). PAPER ONLY: no orders, no money.

    streamlit run dashboard/app.py          # from the repo root
    docker compose -f docker/docker-compose.yml up -d dashboard   # http://localhost:8501

Read-only over the repo. The one exception is the "re-run health check" button,
which runs scripts/paper_healthcheck.py with PAPER_HEALTH_OUT pointed at
logs/dashboard/health.json (the only writable mount in the container).
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import plotly.graph_objects as go
import polars as pl
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kite_auth as KA  # noqa: E402

import data as D  # noqa: E402

ROOT = Path(os.environ.get("DASHBOARD_ROOT", Path(__file__).resolve().parent.parent))
PAPER = ROOT / "audit/paper"
LOGS = ROOT / "logs/paper"
DASH_HEALTH = ROOT / os.environ.get("DASHBOARD_HEALTH_OUT", "logs/dashboard/health.json")
# The rate limit keys on the ATTEMPT, stamped before any probe, not on the
# health.json the script writes last: a check that dies mid-probe still counts.
PROBE_STAMP = DASH_HEALTH.parent / "probe_stamp"
PROBE_LOCK = DASH_HEALTH.parent / "probe.lock"
PROBE_MIN_INTERVAL_S = 600
# worst case 6 weekdays x 2 feeds x 20 s urlopen timeout = 240 s of probing,
# plus the store reads (scripts/paper_healthcheck.py:77-83)
PROBE_TIMEOUT_S = 330
PROBE_LOCK_TTL_S = PROBE_TIMEOUT_S + 60

# reference categorical palette (dataviz skill), fixed by entity, never by rank
C_SIGNAL, C_EW = "#2a78d6", "#eb6834"
C_OVER = {
    "allocator_k30_b0.01_rnone_monthly_20d": "#1baf7a",
    "allocator_k30_b0.01_rstop10_monthly_20d": "#4a3aa7",
    "allocator_k30_b0.01_rstop15_monthly_20d": "#e87ba4",
}
C_BAND, C_MED = "rgba(128,128,128,0.18)", "#8a8984"
BADGE = {
    "OK": ("#008300", "OK"),
    "WARN": ("#c98500", "WARN"),
    "FAIL": ("#e34948", "FAIL"),
    "STALE": ("#c98500", "STALE"),
}

st.set_page_config(page_title="Paper Test Monitor", layout="wide")


def rs(x: float) -> str:
    return f"{'-' if x < 0 else '+'}Rs {abs(x):,.0f}"


def rs_n(x: float) -> str:
    """Rupee P&L on the notional: never a real-money figure."""
    return f"{rs(x)} (notional)"


def pct(x: float) -> str:
    return f"{x * 100:+.2f}%"


def badge(status: str) -> str:
    col, lab = BADGE.get(status, ("#8a8984", status))
    return (
        f"<span style='background:{col};color:#fff;padding:2px 10px;border-radius:4px;"
        f"font-weight:600'>{lab}</span>"
    )


def short(book: str) -> str:
    return (
        book.replace("allocator_k30_b0.01_r", "signal/")
        .replace("_monthly_20d", "")
        .replace("null_signal_k30_b0.01_r", "random/")
        .replace("equal_weight_monthly", "equal-weight")
    )


def layout(fig: go.Figure, ytitle: str) -> go.Figure:
    fig.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=30, b=10),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.08),
        yaxis_title=ytitle,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.2)", zerolinecolor="rgba(128,128,128,0.5)")
    return fig


def paper_banner() -> None:
    st.markdown(
        "<div style='border:2px solid #e34948;border-radius:6px;"
        "padding:8px 14px;margin-bottom:8px'>"
        "<b>PAPER ONLY.</b> Rs 1,00,000 of simulated cash (deployed 2026-09-30) on NSE "
        "point-in-time data. No orders are "
        "placed, no real money is at risk. Pre-registered in "
        "<code>audit/P2_PAPER_PERMUTATION_TEST.md</code>.</div>",
        unsafe_allow_html=True,
    )


# ── cached loads (ttl so a fresh run shows up without a restart) ──────────────
@st.cache_data(ttl=60)
def _record():
    r = D.load_record(PAPER / "record.jsonl")
    return r.value or [], r.error, r.extra


@st.cache_data(ttl=60)
def _latest():
    ld = D.latest_dir(PAPER)
    if not ld.ok:
        return None, None, ld.error, None
    s = D.read_json(ld.value / "summary.json")
    return str(ld.value), (s.value if s.ok else None), s.error, ld.extra.get("note")


def _health():
    return D.merge_health(PAPER / "health.json", DASH_HEALTH)


def _live(rec):
    return D.live_checks(LOGS, rec)


@st.cache_data(ttl=60)
def _warmup(path: str, mtime: float):  # mtime only keys the cache
    return D.warmup_navs(Path(path))


def _nav_mtime(path: str) -> float:
    try:
        return max((f.stat().st_mtime for f in Path(path).glob("nav_*.parquet")), default=0.0)
    except OSError:
        return 0.0


def pipeline_status(h, live: list[dict]) -> tuple[str, list[str]]:
    """One status for the overview: worst of the merged health report, its
    staleness, and the probe-free live checks. Returns (status, reasons)."""
    reasons, sts = [], []
    if h.ok:
        sts.append(h.value["overall"])
        if h.value["overall"] != "OK":
            bad = [c["check"] for c in h.value["checks"] if c.get("status") != "OK"]
            reasons.append(f"health check {h.value['overall']}: {', '.join(bad)}")
        if h.value["stale"]:
            sts.append("WARN")
            reasons.append(f"health not re-checked for {h.value['age_hours']:.0f} h")
        if h.value["missing"]:
            sts.append("WARN")
            reasons.append("checks never reported: " + ", ".join(h.value["missing"]))
    else:
        sts.append("WARN")
        reasons.append(h.error)
    for c in live:
        sts.append(c["status"])
        if c["status"] != "OK":
            reasons.append(f"{c['check']} {c['status']}: {c['detail']}")
    st_ = D.worst(sts)
    if st_ == "WARN" and h.ok and h.value["stale"] and len(reasons) == 1:
        st_ = "STALE"
    return st_, reasons


def go_health() -> None:
    st.session_state["nav"] = "Health & ops"


# ── pages ────────────────────────────────────────────────────────────────────
def page_overview() -> None:
    st.title("Forward paper test — overview")
    rec, err, extra = _record()
    h = _health()
    live = _live(rec)
    status, reasons = pipeline_status(h, live)
    if status == "FAIL":
        st.error(
            "**Data pipeline FAIL.** " + "; ".join(reasons) + ". See Health & ops.",
        )
    elif status in ("WARN", "STALE"):
        st.warning("**Data pipeline " + status + ".** " + "; ".join(reasons) + ".")
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"**Data pipeline health** {badge(status)}", unsafe_allow_html=True)
    if h.ok:
        srcs = ", ".join(
            Path(x).relative_to(ROOT).as_posix() if x.startswith(str(ROOT)) else x
            for x in h.value["sources"]
        )
        c1.caption(
            f"health report {h.value['checked_at']} ({h.value['age_hours']:.1f} h ago) from "
            f"{srcs}; run/record liveness recomputed from logs on every load (no NSE call). "
            "This is the data pipeline, not the strategy."
        )
    else:
        c1.caption(str(h.error))
    c1.button("Open Health & ops", on_click=go_health)
    if err:
        st.info(f"Forward record: {err}")
        return
    hd = D.headline(rec)
    c2.metric("Sessions recorded", hd["sessions"], help=f"{hd['first_date']} .. {hd['last_date']}")
    c3.metric("Data date (last recorded)", hd["last_date"] or "—")
    c4.metric("Last record write (UTC)", (hd["last_run_ts"] or "—").replace("T", " "))
    if extra.get("bad_lines"):
        st.warning(f"{extra['bad_lines']} unparseable line(s) in record.jsonl were skipped.")
    _, summ, _, _ = _latest()
    fm = D.freeze_mismatch(summ)
    if fm:
        st.error(f"Freeze-date cross-check: {fm}.")
    miss = D.missing_pairs(rec)
    if miss:
        st.warning(
            f"{len(miss)} (session, book) pair(s) missing from the record among the signal, "
            f"equal-weight and random books; each counts as a 0 return (as in "
            f"scripts/paper_record.py), which shifts cumulative figures and the rank. "
            f"First: {miss[:3]}"
        )

    st.subheader(f"Forward P&L since the record (re)started on {hd['first_date']}")
    st.caption(
        f"Sum of recorded daily log returns for {hd['sessions']} session(s), "
        f"{hd['first_date']} .. {hd['last_date']}, on Rs 1,00,000 of simulated cash "
        "deployed 2026-09-30. "
        f"Scoring filter: sessions after the {D.FREEZE_DATE} freeze. {D.RESTART_NOTE} "
        "Source: audit/paper/record.jsonl."
    )
    m = st.columns(4)
    m[0].metric(
        "Signal book (volstop), notional", rs_n(hd["signal"]["rupees"]), pct(hd["signal"]["pct"])
    )
    m[1].metric(
        "Equal-weight, notional", rs_n(hd["equal_weight"]["rupees"]), pct(hd["equal_weight"]["pct"])
    )
    if "null_median" in hd:
        m[2].metric(
            f"Random books, median of {hd['n_null']}, notional",
            rs_n(hd["null_median"]["rupees"]),
            pct(hd["null_median"]["pct"]),
        )
        m[3].metric(
            "Random books, 10th–90th pct, notional",
            f"{rs(hd['null_p10']['rupees'])} .. {rs(hd['null_p90']['rupees'])}",
        )
    st.markdown("---")
    r1, r2 = st.columns([1, 3])
    n_books = hd["n_null"] + 1
    r1.metric(f"Signal rank among {n_books}", f"{hd['rank']} / {n_books}" if hd["rank"] else "—")
    r2.warning(
        f"**Do not read this rank.** P2 forbids treating it as evidence before "
        f"{D.EVIDENCE_MONTHS} months of forward record. Record so far: **{hd['sessions']} "
        f"session(s)**, {hd['first_date']} .. {hd['last_date']} "
        f"(**{hd['months_elapsed']:.2f} calendar months**, ≈ {hd['trading_months']:.2f} "
        f"trading months). At this length the rank is noise; it is shown only so the "
        f"plumbing can be checked (1 = best, {n_books} = worst). "
        f"**P2 success = rank 1 of {n_books} at {D.EVIDENCE_MONTHS} months; nothing else "
        f"counts** (audit/P2_PAPER_PERMUTATION_TEST.md:122)."
    )


def page_pnl() -> None:
    st.title("Forward P&L")
    rec, err, _ = _record()
    if err:
        st.info(f"Forward record: {err}")
        return
    ser = D.cumulative_series(rec)
    if not ser.height:
        st.info("No scored sessions yet.")
        return
    band = D.null_band(ser)
    fig = go.Figure()
    if band.height:
        d = band["date"].to_list()
        fig.add_trace(
            go.Scatter(
                x=d, y=band["p90"].to_list(), line=dict(width=0), showlegend=False, hoverinfo="skip"
            )
        )
        fig.add_trace(
            go.Scatter(
                x=d,
                y=band["p10"].to_list(),
                fill="tonexty",
                fillcolor=C_BAND,
                line=dict(width=0),
                name="random books 10th–90th",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=d,
                y=band["p50"].to_list(),
                name="random median",
                line=dict(color=C_MED, width=2, dash="dot"),
            )
        )
    show_over = st.checkbox("Show the other three signal overlays (secondary)", value=False)
    for b, g in ser.group_by("book", maintain_order=True):
        b = b[0]
        kind = g["kind"][0]
        if kind == "signal":
            sty = dict(color=C_SIGNAL, width=3)
        elif kind == "equal_weight":
            sty = dict(color=C_EW, width=2)
        elif kind == "overlay" and show_over:
            sty = dict(color=C_OVER.get(b, C_MED), width=1.5, dash="dash")
        else:
            continue
        fig.add_trace(
            go.Scatter(
                x=g["date"].to_list(),
                y=g["rupees"].to_list(),
                name=short(b),
                mode="lines+markers",
                marker=dict(size=8),
                line=sty,
            )
        )
    n = ser["date"].n_unique()
    first, last = ser["date"].min(), ser["date"].max()
    st.warning(
        f"{n} session(s), {first}..{last}. P2 reads nothing before {D.EVIDENCE_MONTHS} "
        f"months; this is plumbing, not a result. {D.RESTART_NOTE}"
    )
    st.plotly_chart(layout(fig, f"cumulative P&L, Rs notional ({n} session(s))"), width="stretch")
    st.subheader("Per session")
    tab = D.session_table(rec)
    if tab.height:
        st.dataframe(tab.to_pandas(), width="stretch", hide_index=True)
        st.caption("Daily log returns as frozen by the first run that recorded each date.")


def page_warmup() -> None:
    st.title("Warm-up — before the test clock")
    st.warning(
        "This is the 2024-08-29 .. 2026-09-09 replay that precedes the freeze. It is "
        "**context, not the test**: none of it counts toward the P2 statistic."
    )
    path, summ, err, note = _latest()
    if path is None:
        st.info(err)
        return
    if note:
        st.caption(note)
    w = _warmup(path, _nav_mtime(path))
    if not w.height:
        st.info(f"No NAV files readable in {path}.")
        return
    nulls = w.filter(pl.col("kind") == "null")
    fig = go.Figure()
    if nulls.height:
        q = (
            nulls.group_by("date")
            .agg(
                pl.col("growth").quantile(0.1, interpolation="linear").alias("p10"),
                pl.col("growth").quantile(0.9, interpolation="linear").alias("p90"),
                pl.col("growth").quantile(0.5, interpolation="linear").alias("p50"),
            )
            .sort("date")
        )
        d = q["date"].to_list()
        fig.add_trace(
            go.Scatter(
                x=d,
                y=((q["p90"] - 1) * D.NOTIONAL).to_list(),
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=d,
                y=((q["p10"] - 1) * D.NOTIONAL).to_list(),
                fill="tonexty",
                fillcolor=C_BAND,
                line=dict(width=0),
                name="random 10th–90th",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=d,
                y=((q["p50"] - 1) * D.NOTIONAL).to_list(),
                name="random median",
                line=dict(color=C_MED, width=2, dash="dot"),
            )
        )
    for kind, col, wd in (("equal_weight", C_EW, 2), ("signal", C_SIGNAL, 3)):
        g = w.filter(pl.col("kind") == kind)
        if g.height:
            fig.add_trace(
                go.Scatter(
                    x=g["date"].to_list(),
                    y=((g["growth"] - 1) * D.NOTIONAL).to_list(),
                    name=short(g["book"][0]),
                    line=dict(color=col, width=wd),
                )
            )
    st.plotly_chart(layout(fig, "P&L on Rs 10,00,000 notional (warm-up replay)"), width="stretch")
    st.caption("Random band: 10th/50th/90th percentile, linear interpolation (same as P&L page).")
    wu = (summ or {}).get("warmup")
    if wu:
        c = st.columns(2)
        c[0].metric("Warm-up signal cum (log)", f"{wu['signal_cum']:+.4f}")
        c[1].metric("Warm-up random mean (log)", f"{wu['null_mean']:+.4f}")
        st.caption(
            f"Warm-up rank {wu['rank']} of {wu['n']}. Replay before the test clock (P2 "
            f"§Warm-up baseline). Not forward evidence, and not comparable with the forward "
            f"rank as a trend; P2 reads rank only at {D.EVIDENCE_MONTHS} months of forward "
            f"record. From {path}/summary.json, span {wu['span'][0]} .. {wu['span'][1]}."
        )


def page_health() -> None:
    st.title("Health & operations")
    rec, _, _ = _record()
    h = _health()
    live = _live(rec)
    status, reasons = pipeline_status(h, live)
    col1, col2 = st.columns([3, 1])
    now = datetime.now(UTC)
    can, age, why = D.probe_gate(
        PROBE_STAMP, PROBE_LOCK, now, PROBE_MIN_INTERVAL_S, PROBE_LOCK_TTL_S
    )
    with col2:
        if st.button(
            "Re-run health check",
            disabled=not can,
            help="Probes NSE (~12 small requests). At most once per 10 minutes, one at a time.",
        ):
            if not D.acquire_probe(PROBE_STAMP, PROBE_LOCK, datetime.now(UTC), PROBE_LOCK_TTL_S):
                st.warning("another session is already running the health check")
            else:
                with st.spinner("probing NSE and reading the data store..."):
                    env = {**os.environ, "PAPER_HEALTH_OUT": str(DASH_HEALTH)}
                    try:
                        r = subprocess.run(
                            [sys.executable, "scripts/paper_healthcheck.py"],
                            cwd=ROOT,
                            env=env,
                            capture_output=True,
                            text=True,
                            timeout=PROBE_TIMEOUT_S,
                        )
                        if r.returncode != 0:
                            st.error(f"health check exited {r.returncode}: {r.stderr[-800:]}")
                    except Exception as e:  # noqa: BLE001
                        st.error(f"health check could not run: {e}")
                    finally:
                        D.release_probe(PROBE_LOCK)
                st.rerun()
        if not can:
            if why == "rate limit" and age is not None:
                st.caption(
                    f"last on-demand attempt {age / 60:.0f} min ago; next allowed in "
                    f"{(PROBE_MIN_INTERVAL_S - age) / 60:.0f} min"
                )
            else:
                st.caption(why)
    with col1:
        st.markdown(f"Data pipeline {badge(status)}", unsafe_allow_html=True)
        for r_ in reasons:
            st.caption(r_)
    st.subheader("Live (recomputed from logs on every load; no NSE call)")
    for c in live:
        st.markdown(
            f"{badge(c['status'])} &nbsp; **{c['check']}** — {c['detail']}",
            unsafe_allow_html=True,
        )
    st.subheader("Health report (scripts/paper_healthcheck.py)")
    if not h.ok:
        st.info(h.error)
    else:
        hv = h.value
        st.markdown(
            f"Merged overall {badge(hv['overall'])} &nbsp; newest report `{hv['checked_at']}` "
            f"({hv['age_hours']:.1f} h ago)",
            unsafe_allow_html=True,
        )
        if hv["stale"]:
            st.warning(
                f"Health not re-checked for {hv['age_hours']:.0f} h. Nothing runs "
                "scripts/paper_healthcheck.py on a schedule; the live checks above are the "
                "only continuously current signal."
            )
        st.caption(
            "Each check is taken from whichever file ran it most recently (host "
            "audit/paper/health.json or this page's logs/dashboard/health.json). The "
            "'scheduler' check (paper container state) runs only on the host, so a "
            "button press here never replaces it."
        )
        if hv["missing"]:
            st.warning("Never reported by any file: " + ", ".join(hv["missing"]))
        for c in hv["checks"]:
            src = c["_source"].removeprefix(str(ROOT) + "/")
            st.markdown(
                f"{badge(c.get('status', '?'))} &nbsp; **{c.get('check')}** — "
                f"{c.get('detail', '')} &nbsp; <small>({src}, {c['_checked_at']})</small>",
                unsafe_allow_html=True,
            )
            if c.get("check") == "nse_access" and c.get("probes"):
                st.dataframe(
                    pl.DataFrame(c["probes"]).with_columns(pl.all().cast(pl.Utf8)).to_pandas(),
                    hide_index=True,
                )
                st.caption(
                    "200 = served (ranged GET), 404 = not published (normal before ~18:30 IST "
                    "or on a holiday), anything else = blocked / erroring."
                )
            if c.get("missing"):
                st.caption("published but unrecorded: " + ", ".join(c["missing"]))

    st.subheader("Last run blocks (logs/paper/*.status)")
    blocks = D.status_blocks(LOGS, n=5)
    if not blocks:
        st.info(f"no status files in {LOGS}")
    for b in reversed(blocks):
        with st.expander(
            f"{b['started']}  —  {b['outcome']}  ({b['file']})", expanded=b is blocks[-1]
        ):
            st.code("\n".join(b["lines"]) or "(empty)")

    st.subheader("Memory during the latest run")
    mem = D.memory_samples(LOGS)
    if not mem.ok:
        st.info(mem.error)
    else:
        m = mem.value
        fig = go.Figure()
        for colname, colr in (("used_mb", C_SIGNAL), ("avail_mb", C_EW)):
            if colname in m.columns:
                fig.add_trace(
                    go.Scatter(
                        x=m["ts"].to_list(),
                        y=m[colname].to_list(),
                        name=colname,
                        line=dict(color=colr, width=2),
                    )
                )
        st.plotly_chart(layout(fig, "MB"), width="stretch")
        st.caption(f"{mem.source} — {m.height} samples")

    st.subheader("Determinism")
    path, summ, err, note = _latest()
    if summ and summ.get("determinism"):
        d = summ["determinism"]
        c = st.columns(4)
        c[0].metric("Compared against", str(d.get("previous")))
        c[1].metric("Overlap days", d.get("overlap_days"))
        c[2].metric("Max abs rel diff", f"{d.get('max_abs_rel_diff', 0):.2e}")
        c[3].metric("Books diverged", d.get("books_diverged"))
        st.caption(f"{path}/summary.json" + (f" — {note}" if note else ""))
    else:
        st.info(err or "no determinism block in summary.json")


def page_research() -> None:
    st.title("Research & reasoning")
    st.caption(
        "Criterion and result are extracted from the ledger files at load time; open the "
        "file for the full reasoning. Nothing here is a 'winner' — see CLAUDE.md rule 2."
    )
    rows = D.experiments(ROOT)
    st.dataframe(
        pl.DataFrame(rows).to_pandas(),
        width="stretch",
        hide_index=True,
        column_config={
            "criterion": st.column_config.TextColumn(width="large"),
            "result": st.column_config.TextColumn(width="medium"),
        },
    )
    st.subheader("Top-K gate results (audit/topk_gate/*.json)")
    g = D.gate_jsons(ROOT)
    if g.height:
        st.caption(
            "Backtest gates over 13 walk-forward windows, not the forward paper record. "
            "'rank 1/21' with FAIL means it beat the random screens but not the absolute bar. "
            "SHORT-LEG INFORMATION refers to names to avoid, not something the long-only "
            "paper book can trade."
        )
        st.dataframe(g.to_pandas(), width="stretch", hide_index=True)
    else:
        st.info("no gate JSONs found")
    st.subheader("Documents")
    docs = [d for d in D.DOCS if (ROOT / d).exists()]
    if not docs:
        st.info("no ledger documents found")
        return
    pick = st.selectbox("Document", docs)
    try:
        st.markdown((ROOT / pick).read_text())
    except OSError as e:
        st.error(f"cannot read {pick}: {e}")



def _inr(x: float, signed: bool = False) -> str:
    s = f"{abs(x):,.0f}"
    sign = ("+" if x > 0 else "-" if x < 0 else "") if signed else ("-" if x < 0 else "")
    return f"{sign}Rs {s}"


def page_live() -> None:
    import plotly.express as px

    st.header("Live book — Rs 1,00,000 paper portfolio")
    st.caption(
        "Simulated orders at live Kite prices right after the 09:15 open (09:16 IST), "
        "chosen by the same algorithm and sizing code as the end-of-day paper record. "
        "No real money and no orders are ever placed. Marks every 15 minutes in market "
        "hours. Source: audit/paper/live/ (scripts/live_paper.py)."
    )
    snap = D.load_live_snapshot(ROOT)
    state = D.load_live_state(ROOT)
    if snap is None or state is None:
        st.info("Not deployed yet. Rs 1,00,000 cash is scheduled to be invested at "
                "09:16 IST on 2026-09-30, after the Kite login. This page fills in "
                "from the first mark (~09:30).")
        return
    if snap.get("prices") != "live":
        st.warning(f"These marks use '{snap.get('prices')}' prices (a rehearsal), not live quotes.")
    dep = state.get("deployed", {})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Capital", _inr(snap["capital"]))
    c2.metric("Invested (cost)", _inr(sum(h["cost"] for h in snap["holdings"])),
              f"{len(snap['holdings'])} stocks")
    c3.metric("Cash", _inr(snap["cash"]))
    c4.metric("Current value", _inr(snap["nav"]))
    c5, c6, c7 = st.columns(3)
    c5.metric("Today's P&L", _inr(snap["day_chg_rs"], True), f"{snap['day_chg_pct']:+.2%}")
    c6.metric("Total P&L", _inr(snap["pnl_rs"], True), f"{snap['pnl_pct']:+.2%}")
    c7.metric("Last marked", snap["ts"].replace("T", " ")[:16] + " IST")
    if snap.get("unpriced"):
        st.warning(f"No live quote for {', '.join(snap['unpriced'])}; valued at purchase price.")
    st.caption("Total P&L includes buying charges (STT, stamp, exchange, GST). "
               "Selling charges and tax apply only when a position is sold.")

    st.subheader("Trading and risk")
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Realised P&L (sold)", _inr(snap.get("realised_rs", 0.0), True),
              help="Proceeds minus selling charges minus the average cost of the shares sold.")
    r2.metric("Charges paid to date", _inr(snap.get("charges_rs", 0.0)))
    r3.metric("Last rebalance", state.get("last_rebalance", dep.get("trade_date", "—")),
              help="Monthly: the first session of each month at 09:16, turnover budget 30%.")
    pend = state.get("pending_stops") or []
    r4.metric("Stops pending for next open", len(pend), ", ".join(D.short(t) for t in pend) or None)
    cool = state.get("cooldown") or {}
    if cool:
        st.caption("Barred from re-entry after a stop (sessions left): "
                   + ", ".join(f"{D.short(t)} {n}" for t, n in sorted(cool.items())))
    st.caption("Volatility stop: a stock closing below its stop level is sold at the next "
               "09:16 and cannot be re-bought for 21 sessions. Stop levels are set at the "
               "15:35 close mark.")

    st.subheader("Holdings")
    h = pl.DataFrame(snap["holdings"])
    nav = float(snap["nav"]) or 1.0
    view = h.select(
        pl.col("symbol").alias("Stock"), pl.col("sector").alias("Sector"),
        pl.col("qty").alias("Qty"), pl.col("avg_price").alias("Avg buy (Rs)"),
        pl.col("cost").alias("Invested (Rs)"), pl.col("ltp").alias("LTP (Rs)"),
        (pl.col("day_chg_pct") * 100).round(2).alias("1D %"),
        pl.col("day_chg_rs").round(0).alias("1D Rs"),
        pl.col("value").round(0).alias("Value (Rs)"),
        pl.col("pnl_rs").round(0).alias("Total P&L (Rs)"),
        (pl.col("pnl_pct") * 100).round(2).alias("Total P&L %"),
        (pl.col("value") / nav * 100).round(1).alias("Weight %"),
        (pl.col("stop_level") if "stop_level" in h.columns
         else pl.lit(None, pl.Float64)).alias("Stop level (Rs)"),
    ).sort("Value (Rs)", descending=True)
    st.dataframe(view.to_pandas(), hide_index=True, width="stretch", height=420)

    st.subheader("By sector")
    sec = D.sector_summary(snap)
    left, right = st.columns([3, 2])
    with left:
        tm = px.treemap(h.to_pandas(), path=["sector", "symbol"], values="value",
                        color="pnl_pct", color_continuous_scale="RdYlGn",
                        color_continuous_midpoint=0.0,
                        hover_data={"qty": True, "ltp": ":.2f", "pnl_rs": ":,.0f"})
        tm.update_layout(margin=dict(t=10, l=0, r=0, b=0), height=460,
                         coloraxis_colorbar=dict(title="P&L %", tickformat=".1%"))
        st.plotly_chart(tm, width="stretch")
        st.caption("Box size = current value; colour = total P&L %. Click a sector to zoom in.")
    with right:
        sv = sec.select(pl.col("sector").alias("Sector"), pl.col("stocks").alias("Stocks"),
                        (pl.col("weight") * 100).round(1).alias("Weight %"),
                        pl.col("value").round(0).alias("Value (Rs)"),
                        pl.col("pnl_rs").round(0).alias("P&L (Rs)"),
                        pl.col("day_chg_rs").round(0).alias("1D Rs"))
        st.dataframe(sv.to_pandas(), hide_index=True, width="stretch", height=460)
    rows = h.sort(["sector", "value"], descending=[False, True]).to_pandas()
    bar = go.Figure(go.Bar(y=[f"{r.symbol}  ·  {r.sector}" for r in rows.itertuples()],
                           x=rows["value"], orientation="h",
                           marker_color=["#2e7d32" if v >= 0 else "#c62828"
                                         for v in rows["pnl_rs"]],
                           customdata=rows[["pnl_rs", "day_chg_pct"]],
                           hovertemplate="%{y}<br>value Rs %{x:,.0f}"
                                         "<br>P&L Rs %{customdata[0]:,.0f}"
                                         "<br>1D %{customdata[1]:.2%}<extra></extra>"))
    bar.update_layout(height=max(420, 28 * len(rows)), margin=dict(l=10, r=10, t=10, b=10),
                      xaxis_title="Current value (Rs); green = in profit, red = in loss",
                      yaxis=dict(autorange="reversed"))
    with st.container(height=520):
        st.plotly_chart(bar, width="stretch")
    st.caption("Scroll the chart above: every stock, grouped by sector.")

    marks = D.load_live_marks(ROOT)
    if marks.height >= 2:
        st.subheader("Portfolio value through the day")
        f = go.Figure(go.Scatter(x=marks["ts"].to_list(), y=marks["nav"].to_list(),
                                 mode="lines+markers", name="Value"))
        f.add_hline(y=float(snap["capital"]), line_dash="dot",
                    annotation_text="Rs 1,00,000 invested")
        f.update_layout(height=320, yaxis_title="Rs", margin=dict(t=10, b=10),
                        xaxis=dict(rangeslider=dict(visible=True)))
        st.plotly_chart(f, width="stretch")
    sm = D.load_live_stock_marks(ROOT)
    if sm.height and sm["ts"].n_unique() >= 2:
        st.subheader("Per-stock P&L over time")
        pick = st.multiselect("Stocks", sorted(sm["symbol"].unique().to_list()),
                              default=sorted(sm["symbol"].unique().to_list())[:5])
        if pick:
            g = px.line(sm.filter(pl.col("symbol").is_in(pick)).to_pandas(),
                        x="ts", y="pnl_rs", color="symbol", markers=True)
            g.update_layout(height=360, yaxis_title="Total P&L (Rs)", margin=dict(t=10, b=10))
            st.plotly_chart(g, width="stretch")

    st.subheader("Fills")
    led = D.load_live_ledger(ROOT)
    if led.height:
        st.dataframe(led.select([c for c in ("ts", "kind", "side", "ticker", "sector", "qty",
                                             "price", "value", "charges", "realised_rs",
                                             "target_weight", "quote_ts")
                                 if c in led.columns]).to_pandas(),
                     hide_index=True, width="stretch")
    st.caption("A reminder that matters: after 3 sessions of the previous record the algorithm "
               "ranked 17th of 21 against random stock picks. Nothing so far shows it beats "
               "random selection; read early P&L as noise, not skill (audit/P2).")


# ── Kite daily login ─────────────────────────────────────────────────────────

@st.cache_data(ttl=60, show_spinner=False)
def _kite_status(_stamp: float) -> dict:
    """Token status, re-checked at most once a minute (and whenever the file changes)."""
    return KA.check(ROOT)


def kite_status() -> dict:
    p = KA.token_path(ROOT)
    try:
        stamp = p.stat().st_mtime
    except OSError:
        stamp = 0.0
    return _kite_status(stamp)


def _handle_kite_callback() -> None:
    """Kite redirects here with ?request_token=...: exchange it once, then clean the URL."""
    qp = st.query_params
    if "request_token" not in qp and "status" not in qp:
        return
    raw = "&".join(f"{k}={v}" for k, v in qp.items())
    rt = KA.parse_request_token(raw)
    st.query_params.clear()                      # a request token is single-use
    st.session_state["nav"] = "Kite login"
    if rt is None:
        st.session_state["kite_msg"] = ("error", "Kite login was not successful "
                                        f"(status={qp.get('status', '?')}). Try again.")
        return
    if st.session_state.get("kite_rt_done") == rt:
        return
    st.session_state["kite_rt_done"] = rt
    try:
        meta = KA.exchange(ROOT, rt)
        st.session_state["kite_msg"] = ("success", f"Kite connected. Token valid until "
                                        f"{meta['expires_at'][:16].replace('T', ' ')} IST.")
        _kite_status.clear()
    except RuntimeError as e:
        st.session_state["kite_msg"] = ("error", str(e))


def page_kite() -> None:
    st.header("Kite login")
    st.caption(
        "The live book needs a Kite access token to read prices at 09:16 and through the "
        "day. Tokens expire around 06:00 IST every morning. Log in once each trading day "
        "before 09:16. Only login and profile endpoints are used; no orders are ever placed."
    )
    msg = st.session_state.pop("kite_msg", None)
    if msg:
        (st.success if msg[0] == "success" else st.error)(msg[1])
    s = kite_status()
    state = s["state"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Token", {"valid": "Valid", "expired": "Expired", "missing": "Not set",
                        "error": "Check failed"}[state])
    c2.metric("Minted", (s.get("minted_at") or "—")[:16].replace("T", " "))
    c3.metric("Expires (approx.)", (s.get("expires_at") or "—")[:16].replace("T", " "))
    if state == "error":
        st.warning(f"Could not verify the token: {s.get('error')}. Kite may be unreachable.")

    key, secret = KA.credentials(ROOT)
    if not key or not secret:
        st.error("KITE_API_KEY / KITE_API_SECRET are not in .env, so login cannot work.")
        return
    st.subheader("1. Log in")
    st.link_button("Log in to Kite", KA.login_url(key), type="primary")
    st.caption(
        "Kite sends you back to the redirect URL registered for this app. If that URL is "
        "this dashboard, http://127.0.0.1:8501/, the token is captured automatically. "
        "Set it once at developers.kite.trade → My apps → Redirect URL."
    )
    st.subheader("2. Or paste the redirect URL")
    with st.form("kite_paste", clear_on_submit=True):
        txt = st.text_input("Redirect URL or request_token",
                            placeholder="http://127.0.0.1:5000/kite/callback?...&request_token=...",
                            type="password")
        if st.form_submit_button("Connect"):
            rt = KA.parse_request_token(txt)
            if rt is None:
                st.error("No request_token found (or the login status was not success).")
            else:
                try:
                    meta = KA.exchange(ROOT, rt)
                    _kite_status.clear()
                    st.success(f"Kite connected. Token valid until "
                               f"{meta['expires_at'][:16].replace('T', ' ')} IST.")
                except RuntimeError as e:
                    st.error(str(e))
    st.caption("The token is stored in secrets/kite/access_token.json (owner-only, "
               "gitignored) and never shown. The 09:16 trade and the marks read it from there.")


PAGES = {
    "Live book": page_live,
    "Overview": page_overview,
    "P&L": page_pnl,
    "Warm-up context": page_warmup,
    "Health & ops": page_health,
    "Research & reasoning": page_research,
    "Kite login": page_kite,
}

_handle_kite_callback()

with st.sidebar:
    st.markdown("### Paper test monitor")
    choice = st.radio("Section", list(PAGES), label_visibility="collapsed", key="nav")
    st.caption("PAPER ONLY · no orders · read-only view of the repo")
    _ks = kite_status()["state"]
    if _ks == "valid":
        st.success("Kite: connected", icon="✅")
    elif _ks == "error":
        st.warning("Kite: could not verify", icon="⚠️")
    else:
        st.error("Kite: login needed", icon="🔑")
    if st.button("Reload data"):
        st.cache_data.clear()

paper_banner()  # every page: PAPER ONLY / notional
try:
    PAGES[choice]()
except Exception as e:  # noqa: BLE001 -- a message, never a stack trace
    st.error(f"This section could not render: {type(e).__name__}: {e}")
