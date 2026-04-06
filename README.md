# IMC Prosperity

We are working on algorithms for **EMERALDS** and **TOMATOES** while getting used to the **tutorial round** on [IMC Prosperity](https://prosperity.imc.com/).

## TOMATOES

Placeholder — no strategy here yet.

## EMERALDS

Two market-making style bots (symmetric bid/ask around a reservation price, inventory skew, blend toward a fixed anchor near 10 000, `QUOTE_OFFSET` half-spread). Both use the same position limit and only trade EMERALDS.

| Module | Idea | Best PnL (IMC tutorial) | Best PnL (our backtests) |
|--------|------|-------------------------|---------------------------|
| [`emerald_algo.py`](../trading_algorithms/emerald_algo.py) | EMA-smoothed mid for “fair,” then skew + anchor blend, clamp, quote. | ~**+906** | ~**7437** (Monte Carlo on CSV replay, 10k trials, seed 42) |
| [`emerald_algo_no_ema.py`](../trading_algorithms/emerald_algo_no_ema.py) | Same, but reservation from **raw mid** each step (no EMA state). | ~**+1001** | **7427** (same Monte Carlo setup) |

IMC numbers come from logged tutorial runs noted in those files. Our numbers are from **`monte_carlo_sim`**: many random draws over the tunable parameters, each trial replayed on historical **prices/trades CSVs** with a simple fill model — useful for ranking settings, not a guarantee on the live exchange.
