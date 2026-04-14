# IMC Prosperity

We are working on algorithms for **EMERALDS** and **TOMATOES** while getting used to the **tutorial round** on [IMC Prosperity](https://prosperity.imc.com/).

## Tutorial combo ([`algos_tutorial/tutorial_combo.py`](algos_tutorial/tutorial_combo.py))

Single `Trader` that loops over `state.order_depths` and returns orders for both products in one `run`, with JSON `traderData` keyed under `"tomatoes"` for mid history and fit metadata.

**Emeralds (in this module)** — Fixed fair at 10,000 with one bid at 9,993 and one ask at 10,007 (±7 ticks), sized so inventory stays within ±80. No emerald-specific persistence beyond the exchange.

**Tomatoes (in this module)** — Each step appends mid to `traderData["tomatoes"]`, refits a short AR(4) one-step forecast via rolling least squares (tail of recent mids), rounds to an integer fair, and trades in a ±1 tick band around that prediction: aggressive lifts and hits where the book crosses the band, plus passive improving quotes at the touch; position cap 20, optional shorts (`ALLOW_SHORT_SELL`).

## TOMATOES

The tutorial-round tomato logic used in this repo lives in [`tutorial_combo.py`](algos_tutorial/tutorial_combo.py) (see above). For other tomato-only experiments, see the parent workspace under [`trading_algorithms/`](../trading_algorithms/) (e.g. EMA / RSI variants).

## EMERALDS

[`tutorial_combo.py`](algos_tutorial/tutorial_combo.py) uses a **static** fair and symmetric half-spread (simple market-making). Separately, we also ship richer emerald bots in the parent folder (EMA / anchor / skew). Both styles below use the same position limit and only trade EMERALDS.

| Module | Idea | Best PnL (IMC tutorial) | Best PnL (our backtests) |
|--------|------|-------------------------|---------------------------|
| [`emerald_algo.py`](../trading_algorithms/emerald_algo.py) | EMA-smoothed mid for “fair,” then skew + anchor blend, clamp, quote. | ~**+906** | ~**7437** (Monte Carlo on CSV replay, 10k trials, seed 42) |
| [`emerald_algo_no_ema.py`](../trading_algorithms/emerald_algo_no_ema.py) | Same, but reservation from **raw mid** each step (no EMA state). | ~**+1001** | **7427** (same Monte Carlo setup) |

IMC numbers come from logged tutorial runs noted in those files. Our numbers are from **`monte_carlo_sim`**: many random draws over the tunable parameters, each trial replayed on historical **prices/trades CSVs** with a simple fill model — useful for ranking settings, not a guarantee on the live exchange.
