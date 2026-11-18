# Decision log

| Decision | Reason | Effect on interpretation |
|---|---|---|
| Keep daily start and end state of charge at 50% | Avoids hidden value carried between days | Understates what a multi-day optimiser could earn |
| Binary charge/discharge variable | LP alone can charge and discharge together at negative prices | More realistic, slower than LP |
| Optimise expected price as well as median | Revenue is linear in price, so the mean is the relevant point forecast under risk neutrality | Quantile mean performed best among LightGBM point inputs |
| Conformal rank fixed after a unit test failed | The first implementation used an interpolation-style index and returned the wrong order statistic | Fixed before results were generated |
| Added the 7-day-profile comparison after results | It was the strongest simple baseline | Labelled post hoc in the README and output filename |
| Changed PSI reference to the same month one year earlier | A rolling year mixed seasons | Removes calendar seasonality from the reference |
| PSI uses five bins, Jeffreys smoothing and one value per day for daily features | Repeated daily values and sparse bins inflated PSI | Alerts remain common, so PSI is not used as a retraining trigger |
