# IMC Prosperity

We are working on algorithms for **EMERALDS** and **TOMATOES** while getting used to the **tutorial round** on [IMC Prosperity](https://prosperity.imc.com/).

## Tutorial combo ([`algos_tutorial/tutorial_combo.py`](algos_tutorial/tutorial_combo.py))

Single `Trader` that loops over `state.order_depths` and returns orders for both products in one `run`, with JSON `traderData` keyed under `"tomatoes"` for mid history and fit metadata.

**Emeralds** — Fixed fair at 10,000 with one bid at 9,993 and one ask at 10,007 (±7 ticks), sized so inventory stays within ±80. No emerald-specific persistence beyond the exchange.

**Tomatoes** — Each step appends mid to `traderData["tomatoes"]`, refits a short AR(4) one-step forecast via rolling least squares (tail of recent mids), rounds to an integer fair, and trades in a ±1 tick band around that prediction: aggressive lifts and hits where the book crosses the band, plus passive improving quotes at the touch; position cap 20, optional shorts (`ALLOW_SHORT_SELL`).

IMC numbers come from logged tutorial runs noted in those files. Our numbers are from **`monte_carlo_sim`**: many random draws over the tunable parameters, each trial replayed on historical **prices/trades CSVs** with a simple fill model — useful for ranking settings, not a guarantee on the live exchange.
