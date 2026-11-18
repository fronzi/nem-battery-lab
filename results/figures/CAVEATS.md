# Figure caveats

All values are backtests on public AEMO data for a normalised 1 MW / 2 MWh price-taking battery.
They exclude market impact, FCAS, fees, network charges, capacity fade and outages.

- `fig1_value_capture`: share of the perfect-foresight benchmark, which no real operator can achieve.
- `fig2_cumulative_value`: value accumulates in a few extreme-price events, so paths are regime dependent.
- `fig3_interval_coverage`: marginal coverage over the test period. It does not guarantee coverage during spikes.
- `fig4_example_dispatch`: one illustrative day (one week before the final test day), not a typical-performance claim.
