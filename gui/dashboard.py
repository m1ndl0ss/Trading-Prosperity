"""
Read-only visualization: local Prosperity CSVs + offline strategy replay.
Strategies are auto-loaded from `trading_algos/{ROUND}/` matching the selected
data folder (see `gui/algorithm_discovery.py`).

Run from `Trading-Prosperity` root (needs `replay_engine` + `datamodel` on PYTHONPATH,
e.g. parent `IMC Prosperity` folder, or copy those modules here):

  .venv\\Scripts\\python.exe -m streamlit run gui/dashboard.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_IMC_HOME = _ROOT.parent.parent
for _p in (_ROOT, _IMC_HOME):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from gui.algorithm_discovery import discover_strategies, find_strategy, trading_algorithms_fingerprint
from gui.strategy_registry import get_strategy_entry, goal_hint, list_all_strategies
from replay_engine import market_frame, read_prices_csv, read_trades_csv, replay_summary, replay_to_dataframe

ROOT = _ROOT
EMERALDS = "EMERALDS"

_DAY_RE = re.compile(r"day_(-?\d+)\.csv$", re.IGNORECASE)


def _scan_data_rounds() -> list[str]:
    """Return sorted list of subdirectory names under Data/."""
    data_dir = ROOT / "Data"
    if not data_dir.is_dir():
        return []
    return sorted(x.name for x in data_dir.iterdir() if x.is_dir())


def _build_day_map(round_name: str) -> dict[str, tuple[str, str]]:
    """
    Return {day_label: (prices_path, trades_path)} for every day that has
    both a prices_*.csv and a matching trades_*.csv in Data/{round_name}/.
    Sorted numerically by day number.
    """
    folder = ROOT / "Data" / round_name
    result: dict[int, tuple[str, str]] = {}
    for pf in folder.glob("prices_*.csv"):
        m = _DAY_RE.search(pf.name)
        if not m:
            continue
        day_num = int(m.group(1))
        trades_name = pf.name.replace("prices_", "trades_", 1)
        tf = folder / trades_name
        if tf.is_file():
            result[day_num] = (str(pf), str(tf))
    return {
        f"Day {d}": result[d]
        for d in sorted(result)
    }


def _rsi_wilder(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder RSI on a price series (here: mid from the book). Returns 0–100 where defined."""
    c = close.astype(float)
    delta = c.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    alpha = 1.0 / float(period)
    avg_gain = gain.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    al = avg_loss.replace(0.0, np.nan)
    rs = avg_gain / al
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi.mask((avg_gain > 0) & (avg_loss == 0), 100.0)
    rsi = rsi.mask((avg_gain == 0) & (avg_loss == 0), 50.0)
    return rsi


def _ema_mid(close: pd.Series, period: int) -> pd.Series:
    """Standard EMA on mid (span = period, adjust=False)."""
    c = close.astype(float)
    p = max(1, int(period))
    return c.ewm(span=p, adjust=False).mean()


@st.cache_data
def _load_prices(path_str: str) -> pd.DataFrame:
    return read_prices_csv(Path(path_str))


@st.cache_data
def _load_trades(path_str: str) -> pd.DataFrame:
    return read_trades_csv(Path(path_str))


@st.cache_data
def _compute_frame(
    prices_path: str,
    trades_path: str,
    product: str,
    strategy_id: str | None,
    initial_cash: float,
    _algo_fp: tuple[int, ...],
    subdir: str | None = None,
) -> tuple[pd.DataFrame, dict | None]:
    prices = _load_prices(prices_path)
    trades = _load_trades(trades_path)
    products = frozenset(prices["product"].dropna().astype(str).unique().tolist())
    if strategy_id:
        entry = get_strategy_entry(product, strategy_id, products, subdir=subdir)
        if entry is not None:
            _sid, _label, cls = entry
            df = replay_to_dataframe(prices, trades, product, cls, initial_cash=initial_cash)
            summ = replay_summary(df, initial_cash) if not df.empty else None
            return df, summ
    df = market_frame(prices, product)
    return df, None


def _trades_for_product(trades_path: str, product: str) -> pd.DataFrame:
    t = _load_trades(trades_path)
    t = t[t["symbol"] == product].copy()
    if t.empty:
        return t
    t["price"] = t["price"].astype(float)
    return t


def _apply_time_window(df: pd.DataFrame, t_min: int | None, t_max: int | None) -> pd.DataFrame:
    if df.empty:
        return df
    out = df
    if t_min is not None:
        out = out[out["timestamp"] >= t_min]
    if t_max is not None:
        out = out[out["timestamp"] <= t_max]
    return out


PANEL_PRICE = "price"
PANEL_VOLUME = "volume"
PANEL_RSI = "rsi"
PANEL_STRATEGY = "strategy"

_PANEL_LABELS: dict[str, str] = {
    PANEL_PRICE: "Price & tape",
    PANEL_VOLUME: "L1 volume",
    PANEL_RSI: "RSI",
    PANEL_STRATEGY: "Position & equity",
}


def _available_panels(has_strategy: bool) -> list[str]:
    out = [PANEL_PRICE, PANEL_VOLUME, PANEL_RSI]
    if has_strategy:
        out.append(PANEL_STRATEGY)
    return out


def _default_panel_order(available: list[str]) -> list[str]:
    return list(available)


def _sync_stored_panel_order(product: str, available: list[str]) -> None:
    """Keep session order in sync with what is possible for this product/strategy."""
    key = f"chart_panel_order_{product}"
    if key not in st.session_state:
        st.session_state[key] = _default_panel_order(available)
        return
    cur = [p for p in st.session_state[key] if p in available]
    for p in available:
        if p not in cur:
            cur.append(p)
    if not cur:
        cur = [PANEL_PRICE] if PANEL_PRICE in available else [available[0]]
    st.session_state[key] = cur


def _render_panel_controls(product: str, available: list[str]) -> list[str]:
    """
    Show/hide and reorder panels (↑/↓). Returns top-to-bottom panel ids for the figure.
    """
    key = f"chart_panel_order_{product}"
    _sync_stored_panel_order(product, available)
    order: list[str] = list(st.session_state[key])

    st.markdown("**Visible panels (top → bottom)**")
    for i, pid in enumerate(list(order)):
        label = _PANEL_LABELS.get(pid, pid)
        c0, c1, c2, c3, c4 = st.columns([0.35, 3.2, 0.55, 0.55, 0.55])
        c0.markdown(f"{i + 1}.")
        c1.markdown(label)
        if c2.button("↑", key=f"pnl_up_{product}_{i}", disabled=i == 0):
            order[i - 1], order[i] = order[i], order[i - 1]
            st.session_state[key] = order
            st.rerun()
        if c3.button("↓", key=f"pnl_dn_{product}_{i}", disabled=i >= len(order) - 1):
            order[i + 1], order[i] = order[i], order[i + 1]
            st.session_state[key] = order
            st.rerun()
        if c4.button("Hide", key=f"pnl_hide_{product}_{i}"):
            order.pop(i)
            if not order:
                order = [PANEL_PRICE] if PANEL_PRICE in available else [available[0]]
            st.session_state[key] = order
            st.rerun()

    hidden = [p for p in available if p not in order]
    if hidden:
        add_cols = st.columns([3, 1])
        choice = add_cols[0].selectbox(
            "Add panel",
            options=hidden,
            format_func=lambda p: _PANEL_LABELS.get(p, p),
            key=f"pnl_add_sel_{product}",
        )
        if add_cols[1].button("Add", key=f"pnl_add_btn_{product}"):
            order.append(choice)
            st.session_state[key] = order
            st.rerun()

    return list(st.session_state[key])


def _build_figure(
    df: pd.DataFrame,
    trades_pts: pd.DataFrame,
    has_strategy: bool,
    product: str,
    rsi_period: int = 14,
    ema_period: int = 20,
    panel_order: list[str] | None = None,
) -> go.Figure:
    available = _available_panels(has_strategy)
    if panel_order is None:
        panel_order = _default_panel_order(available)
    order = [p for p in panel_order if p in available]
    if not order:
        order = [PANEL_PRICE] if PANEL_PRICE in available else [available[0]]

    n_rows = len(order)
    row_map = {pid: i + 1 for i, pid in enumerate(order)}
    specs: list[list[dict]] = [[{"secondary_y": True}] if pid == PANEL_STRATEGY else [{}] for pid in order]
    row_heights = [1.0 / n_rows] * n_rows

    fig = make_subplots(
        rows=n_rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=row_heights,
        specs=specs,
    )

    x = df["timestamp"]
    rp = row_map.get(PANEL_PRICE)
    rv = row_map.get(PANEL_VOLUME)
    rr = row_map.get(PANEL_RSI)
    rs = row_map.get(PANEL_STRATEGY)

    if rp is not None:
        fig.add_trace(
            go.Scatter(x=x, y=df["best_bid"], name="Best bid", line=dict(width=1)), row=rp, col=1
        )
        fig.add_trace(
            go.Scatter(x=x, y=df["best_ask"], name="Best ask", line=dict(width=1)), row=rp, col=1
        )
        fig.add_trace(
            go.Scatter(x=x, y=df["mid"], name="Mid", line=dict(dash="dot", width=1)), row=rp, col=1
        )

        ema_mid = _ema_mid(df["mid"], ema_period)
        if ema_mid.notna().any():
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=ema_mid,
                    name=f"EMA mid ({ema_period})",
                    line=dict(width=2, color="#ff7f0e"),
                ),
                row=rp,
                col=1,
            )

        if has_strategy and "fair" in df.columns:
            fair = df["fair"]
            if fair.notna().any():
                fig.add_trace(
                    go.Scatter(x=x, y=fair, name="Fair (model)", line=dict(width=2)), row=rp, col=1
                )
        if has_strategy and "ema" in df.columns:
            ema = df["ema"]
            if ema.notna().any():
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=ema,
                        name="EMA (state)",
                        line=dict(width=2, dash="dot"),
                    ),
                    row=rp,
                    col=1,
                )
        if has_strategy and "bid_px" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=df["bid_px"],
                    name="Strategy bid",
                    line=dict(width=1, dash="dash"),
                    connectgaps=False,
                ),
                row=rp,
                col=1,
            )
        if has_strategy and "ask_px" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=df["ask_px"],
                    name="Strategy ask",
                    line=dict(width=1, dash="dash"),
                    connectgaps=False,
                ),
                row=rp,
                col=1,
            )

        if not trades_pts.empty:
            fig.add_trace(
                go.Scatter(
                    x=trades_pts["timestamp"],
                    y=trades_pts["price"],
                    mode="markers",
                    name="Trades (tape)",
                    marker=dict(size=5, opacity=0.45, color="#7f7f7f"),
                ),
                row=rp,
                col=1,
            )

        if has_strategy and "sim_buy_px" in df.columns:
            bmsk = (df["buy_fills_step"] > 0) & df["sim_buy_px"].notna()
            if bmsk.any():
                sub = df.loc[bmsk]
                sizes = (6 + sub["sim_buy_vol"].astype(float).clip(upper=36)).tolist()
                fig.add_trace(
                    go.Scatter(
                        x=sub["timestamp"],
                        y=sub["sim_buy_px"],
                        mode="markers",
                        name="Our buy (sim)",
                        marker=dict(
                            symbol="triangle-up",
                            size=sizes,
                            color="#2ca02c",
                            line=dict(width=1, color="white"),
                        ),
                        hovertemplate="buy %{y} × %{customdata}<extra></extra>",
                        customdata=sub["sim_buy_vol"],
                    ),
                    row=rp,
                    col=1,
                )
        if has_strategy and "sim_sell_px" in df.columns:
            smsk = (df["sell_fills_step"] > 0) & df["sim_sell_px"].notna()
            if smsk.any():
                sub = df.loc[smsk]
                sizes = (6 + sub["sim_sell_vol"].astype(float).clip(upper=36)).tolist()
                fig.add_trace(
                    go.Scatter(
                        x=sub["timestamp"],
                        y=sub["sim_sell_px"],
                        mode="markers",
                        name="Our sell (sim)",
                        marker=dict(
                            symbol="triangle-down",
                            size=sizes,
                            color="#d62728",
                            line=dict(width=1, color="white"),
                        ),
                        hovertemplate="sell %{y} × %{customdata}<extra></extra>",
                        customdata=sub["sim_sell_vol"],
                    ),
                    row=rp,
                    col=1,
                )

    if rv is not None:
        if df["bid_vol_1"].notna().any():
            fig.add_trace(
                go.Scatter(x=x, y=df["bid_vol_1"], name="Mkt bid vol L1", stackgroup=None),
                row=rv,
                col=1,
            )
        if df["ask_vol_1"].notna().any():
            fig.add_trace(
                go.Scatter(x=x, y=df["ask_vol_1"], name="Mkt ask vol L1", stackgroup=None),
                row=rv,
                col=1,
            )

        if has_strategy and "bid_size" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=df["bid_size"],
                    name="Quoted bid size",
                    line=dict(dash="dot"),
                    connectgaps=False,
                ),
                row=rv,
                col=1,
            )
        if has_strategy and "ask_size" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=df["ask_size"],
                    name="Quoted ask size",
                    line=dict(dash="dot"),
                    connectgaps=False,
                ),
                row=rv,
                col=1,
            )

    if rr is not None:
        rsi = _rsi_wilder(df["mid"], period=rsi_period)
        fig.add_trace(
            go.Scatter(
                x=x,
                y=rsi,
                name=f"RSI ({rsi_period})",
                line=dict(width=1.5, color="#9467bd"),
            ),
            row=rr,
            col=1,
        )
        fig.add_hline(
            y=70,
            line_dash="dash",
            line_color="rgba(120,120,120,0.55)",
            line_width=1,
            row=rr,
            col=1,
        )
        fig.add_hline(
            y=30,
            line_dash="dash",
            line_color="rgba(120,120,120,0.55)",
            line_width=1,
            row=rr,
            col=1,
        )

    if rs is not None and has_strategy:
        if "pos" in df.columns:
            fig.add_trace(
                go.Scatter(x=x, y=df["pos"], name="Position", line=dict(width=2)),
                row=rs,
                col=1,
                secondary_y=False,
            )
        if "equity" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=df["equity"],
                    name="Equity (MTM mid)",
                    line=dict(width=2, dash="dash"),
                ),
                row=rs,
                col=1,
                secondary_y=True,
            )

    fig.update_layout(
        height=160 + n_rows * 560,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=48, r=24, t=60, b=48),
        # Linked hover across stacked rows (Plotly defaults to "single" = only the panel under the cursor).
        hoversubplots="axis",
        hovermode="x unified",
        hoverlabel=dict(namelength=-1, align="left"),
        # Easier to pick a time slice on dense lines (matches nearest x across traces).
        hoverdistance=-1,
        spikedistance=-1,
    )
    fig.update_xaxes(
        showspikes=True,
        spikemode="across",
        spikesnap="hovered data",
        spikedash="solid",
        spikethickness=1,
        spikecolor="rgba(60,60,60,0.45)",
    )
    fig.update_xaxes(title_text="Timestamp", row=n_rows, col=1)
    if rp is not None:
        fig.update_yaxes(title_text="Price", row=rp, col=1)
    if rv is not None:
        fig.update_yaxes(title_text="Volume", row=rv, col=1)
    if rr is not None:
        fig.update_yaxes(title_text="RSI", range=[0, 100], row=rr, col=1)
    if rs is not None:
        fig.update_yaxes(title_text="Position", row=rs, col=1, secondary_y=False)
        fig.update_yaxes(title_text="Equity", row=rs, col=1, secondary_y=True)
    return fig


def _build_emeralds_mid_fair_volume(df: pd.DataFrame, px_per_step: float) -> go.Figure:
    """
    Evenly-spaced x (one step per row) so bubbles do not stack on dense timestamps.
    Marker size ~ sqrt(L1 bid vol + L1 ask vol). Mid vs fair use a small x jitter.
    """
    d = df.reset_index(drop=True)
    n = len(d)
    idx = np.arange(n, dtype=float)
    ts = d["timestamp"].astype(int).to_numpy()

    bv = d["bid_vol_1"].fillna(0.0).to_numpy(dtype=float)
    av = d["ask_vol_1"].fillna(0.0).to_numpy(dtype=float)
    vol = bv + av
    vmax = float(vol.max()) if vol.size and np.nanmax(vol) > 0 else 1.0
    raw = np.sqrt(np.clip(vol, 0.0, np.inf) / vmax)
    size = (6.0 + 34.0 * raw).tolist()

    fig = go.Figure()

    x_mid = idx - 0.22
    fig.add_trace(
        go.Scatter(
            x=x_mid,
            y=d["mid"],
            mode="lines+markers",
            name="Mid",
            line=dict(color="rgba(31,119,180,0.35)", width=1),
            marker=dict(
                size=size,
                color="#1f77b4",
                opacity=0.78,
                line=dict(width=0.6, color="white"),
            ),
            customdata=np.column_stack([ts, vol]),
            hovertemplate=(
                "<b>Mid</b><br>ts=%{customdata[0]}<br>price=%{y:.4f}<br>"
                "L1 vol (bid+ask)=%{customdata[1]:.0f}<extra></extra>"
            ),
        )
    )

    if "fair" in d.columns and d["fair"].notna().any():
        x_fair = idx + 0.22
        fig.add_trace(
            go.Scatter(
                x=x_fair,
                y=d["fair"],
                mode="lines+markers",
                name="Fair",
                line=dict(color="rgba(214,39,40,0.35)", width=1),
                marker=dict(
                    size=size,
                    color="#d62728",
                    opacity=0.78,
                    line=dict(width=0.6, color="white"),
                ),
                customdata=np.column_stack([ts, vol]),
                hovertemplate=(
                    "<b>Fair</b><br>ts=%{customdata[0]}<br>price=%{y:.4f}<br>"
                    "L1 vol (bid+ask)=%{customdata[1]:.0f}<extra></extra>"
                ),
            )
        )

    mid_lo, mid_hi = float(d["mid"].min()), float(d["mid"].max())
    y0, y1 = mid_lo, mid_hi
    if "fair" in d.columns and d["fair"].notna().any():
        y0 = min(y0, float(d["fair"].min()))
        y1 = max(y1, float(d["fair"].max()))
    span = max(y1 - y0, 1e-6)
    pad = span * 0.06 + 1.0
    y0 -= pad
    y1 += pad

    w = int(min(32_000, max(720, px_per_step * n)))

    dtick = max(1, n // 22) if n > 22 else 1
    fig.update_layout(
        title="EMERALDS — mid & fair (circle size = L1 bid+ask volume)",
        height=540,
        width=w,
        xaxis=dict(
            title="Tick index (even spacing — hover reads timestamp)",
            tickmode="linear",
            tick0=0,
            dtick=dtick,
            range=[-0.8, (n - 1) + 0.8],
        ),
        yaxis=dict(title="Price", range=[y0, y1]),
        margin=dict(l=56, r=28, t=72, b=56),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        paper_bgcolor="#fafafa",
        plot_bgcolor="#fafafa",
    )
    return fig


def _render_product_block(
    product: str,
    prices_path: str,
    trades_path: str,
    strategy_id: str | None,
    initial_cash: float,
    rsi_period: int,
    ema_period: int,
    use_window: bool,
    t_min: int | None,
    t_max: int | None,
    algo_fp: tuple[int, ...],
    subdir: str | None = None,
) -> None:
    df, summ = _compute_frame(
        prices_path,
        trades_path,
        product,
        strategy_id,
        float(initial_cash),
        algo_fp,
        subdir=subdir,
    )
    if df.empty:
        st.warning(f"No rows for **{product}** (empty book or missing in CSV).")
        return

    df = _apply_time_window(df, t_min if use_window else None, t_max if use_window else None)
    if df.empty:
        st.warning(f"No rows for **{product}** after time filter.")
        return

    has_strategy = summ is not None
    trades_pts = _trades_for_product(trades_path, product)
    if use_window and t_min is not None:
        trades_pts = trades_pts[trades_pts["timestamp"] >= t_min]
    if use_window and t_max is not None:
        trades_pts = trades_pts[trades_pts["timestamp"] <= t_max]

    tab_full, tab_em_mid = st.tabs(["Full replay", "EMERALDS · mid / fair · volume"])

    with tab_full:
        available_panels = _available_panels(has_strategy)
        with st.expander("Chart panels (show / order)", expanded=False):
            st.caption(
                "Hide rows with **Hide**, restore via **Add panel**. **RSI** / **EMA** use book **mid**."
            )
            panel_order = _render_panel_controls(product, available_panels)

        fig = _build_figure(
            df,
            trades_pts,
            has_strategy=has_strategy,
            product=product,
            rsi_period=rsi_period,
            ema_period=ema_period,
            panel_order=panel_order,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Hover is synced by **time** across stacked rows (recent Streamlit + Plotly)."
        )

        if summ:
            cols = st.columns(4)
            cols[0].metric("Ending position", f"{summ['ending_pos']}")
            cols[1].metric("Cash", f"{summ['cash']:.2f}")
            cols[2].metric("Equity (approx)", f"{summ['equity']:.2f}")
            cols[3].metric("PnL vs start", f"{summ['pnl']:+,.2f}")
            st.caption(
                f"Buy volume filled (sim): {summ['buy_fills']} · "
                f"Sell volume filled (sim): {summ['sell_fills']}"
            )
        elif strategy_id:
            st.info(f"No replay overlay for **{product}** (strategy does not trade this product).")
        else:
            st.info("Pick a strategy that includes this product for simulated PnL and overlays.")

    with tab_em_mid:
        if product != EMERALDS:
            st.info("This bubble chart is only for **EMERALDS**.")
        else:
            st.caption(
                "Even x-spacing; marker size ~ L1 bid+ask volume. Blue **mid** / red **fair** jittered."
            )
            px_step = st.slider(
                "Pixels per tick",
                4,
                36,
                14,
                2,
                key=f"px_step_{product}",
            )
            fig_em = _build_emeralds_mid_fair_volume(df, float(px_step))
            if "fair" not in df.columns or not df["fair"].notna().any():
                st.info("No **fair** in traderData for this run — mid only.")
            st.plotly_chart(
                fig_em,
                use_container_width=False,
                config={
                    "scrollZoom": True,
                    "displayModeBar": True,
                    "modeBarButtonsToAdd": ["pan2d"],
                },
            )


def main() -> None:
    st.set_page_config(page_title="Prosperity replay dashboard", layout="wide")
    st.title("Trading visualization (CSV + offline replay)")
    st.caption(
        "Pick a round and day in the sidebar. "
        "Algorithms load from `trading_algos/{ROUND}/` matching your selection."
    )

    with st.sidebar:
        # ── Round selector ────────────────────────────────────────────────
        st.header("Data")
        rounds = _scan_data_rounds()
        if not rounds:
            st.error("No round folders found under `Data/`.")
            st.stop()

        round_name: str = st.selectbox("Round", options=rounds, index=0)

        # ── Day selector ──────────────────────────────────────────────────
        day_map = _build_day_map(round_name)
        if not day_map:
            st.error(f"No matching prices+trades CSV pairs found in `Data/{round_name}/`.")
            st.stop()

        day_labels = list(day_map)
        day_choice: str = st.selectbox("Day", options=day_labels, index=len(day_labels) - 1)
        prices_path, trades_path = day_map[day_choice]

        initial_cash = st.number_input("Initial cash (replay)", value=0.0, step=1000.0)

        # ── Load prices to know which products exist ───────────────────────
        prices = _load_prices(prices_path)
        csv_products = sorted(prices["product"].dropna().unique().tolist())
        products_set = frozenset(prices["product"].dropna().astype(str).unique().tolist())

        # ── Algo discovery scoped to trading_algos/{round_name}/ ──────────
        algo_subdir = round_name  # e.g. "TUTORIAL" or "ROUND1"
        algo_fp = trading_algorithms_fingerprint(subdir=algo_subdir)
        discovered = discover_strategies(products_set, subdir=algo_subdir)
        n_disc = len(discovered)

        st.divider()

        # ── Indicator settings ────────────────────────────────────────────
        st.subheader("Indicators")
        rsi_period = int(
            st.number_input("RSI period (Wilder, on mid)", min_value=2, max_value=55, value=14, step=1)
        )
        ema_period = int(
            st.number_input("EMA period (on mid, chart overlay)", min_value=2, max_value=200, value=20, step=1)
        )

        st.divider()

        # ── Strategy selector ─────────────────────────────────────────────
        algos_dir = ROOT / "trading_algos" / algo_subdir
        algo_caption = (
            f"`trading_algos/{algo_subdir}/`"
            if algos_dir.is_dir()
            else f"`trading_algos/{algo_subdir}/` *(folder not found)*"
        )
        st.subheader(f"Strategy — {algo_caption}")

        strategies = list_all_strategies(products_set, subdir=algo_subdir)
        strat_labels = ["None (market data only)"] + [f"{sid}: {lbl}" for sid, lbl, _ in strategies]
        strat_choice = st.selectbox(
            "Algorithm",
            options=strat_labels,
            index=0,
            help="Only algorithms from the matching round folder are shown.",
        )

        strategy_id: str | None = None
        strat_products: frozenset[str] | None = None
        if strat_choice != "None (market data only)" and strategies:
            idx = strat_labels.index(strat_choice) - 1
            if 0 <= idx < len(strategies):
                strategy_id = strategies[idx][0]
                ent = find_strategy(products_set, strategy_id, subdir=algo_subdir)
                if ent is not None:
                    strat_products = frozenset(ent.products & products_set)

        hint = goal_hint(strategy_id, subdir=algo_subdir)
        if hint:
            with st.expander("Strategy note (GOAL_HINT)", expanded=False):
                st.write(hint)

        if strategy_id and strat_products:
            st.success(
                f"**{strategy_id}** covers: **{', '.join(sorted(strat_products))}**"
            )
        elif strategy_id:
            st.warning("Strategy not found or no product overlap with this CSV.")

        st.caption(f"Discovered **{n_disc}** algorithm(s) for **{round_name}**.")

        st.divider()

        # ── Time window ────────────────────────────────────────────────────
        st.subheader("Time window (optional)")
        ts_series = prices["timestamp"].dropna().astype(int)
        if not ts_series.empty:
            ts_min_all, ts_max_all = int(ts_series.min()), int(ts_series.max())
        else:
            ts_min_all, ts_max_all = 0, 0
        use_window = st.checkbox("Filter timestamps", value=False)
        t_min = t_max = None
        if use_window:
            t_min = int(st.number_input("Min timestamp", value=ts_min_all, step=100))
            t_max = int(st.number_input("Max timestamp", value=ts_max_all, step=100))

    # ── Main content ───────────────────────────────────────────────────────
    if strategy_id and strat_products:
        prod_list = sorted(strat_products)
    else:
        prod_list = csv_products
        if not prod_list:
            st.warning("No products found in the prices CSV.")
            st.stop()

    common_kwargs: dict = dict(
        prices_path=prices_path,
        trades_path=trades_path,
        strategy_id=strategy_id,
        initial_cash=initial_cash,
        rsi_period=rsi_period,
        ema_period=ema_period,
        use_window=use_window,
        t_min=t_min,
        t_max=t_max,
        algo_fp=algo_fp,
        subdir=algo_subdir,
    )

    if len(prod_list) == 1:
        _render_product_block(prod_list[0], **common_kwargs)
    else:
        tabs = st.tabs(prod_list)
        for tab, prod in zip(tabs, prod_list):
            with tab:
                _render_product_block(prod, **common_kwargs)


if __name__ == "__main__":
    main()
