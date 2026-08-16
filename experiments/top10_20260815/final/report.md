# Top-10 MMA development experiment campaign

## Result

The development incumbent remains `family-01-weighted-v8-control` on the Original 2022–2025 folds: 726/1,108 correct (65.5235%), log loss 0.619595481, Brier 0.215424365, ECE 0.026405312, calibration intercept -0.046801873, and calibration slope 1.213108830.

The historical period from 2026-01-01 through 2026-08-08 is compromised, permanently retired, and unscored. It has no prediction identity or gate metric. The software access ledger remains at zero.

## Ten experiment families

| Family | Experiment | Classification | Boundary | Accuracy | Log loss | Brier |
|---:|---|---|---|---:|---:|---:|
| 1 | `family-01-weighted-v8-control` | incumbent | Original 2022–2025, n=1,108 | 0.655235 | 0.619595 | 0.215424 |
| 2 | `family-02-horizon-recency` | negative | Original 2022–2025, n=1,108 | 0.646209 | 0.622901 | 0.216923 |
| 3 | `family-03-temporal-calibration` | negative | Original 2022–2025, n=1,108 | 0.644404 | 0.624646 | 0.217188 |
| 4 | `family-04-chronological-oof-ensemble` | negative | Original 2022–2025, n=1,108 | 0.648014 | 0.625677 | 0.217369 |
| 5 | `family-05-stable-semantic-portfolio` | negative | Original 2022–2025, n=1,108 | 0.636282 | 0.636814 | 0.223184 |
| 6 | `family-06-multiscale-count-aware-state` | inconclusive | failed before construction | — | — | — |
| 7 | `family-07-matchup-swap-geometry` | inconclusive | dependency unavailable | — | — | — |
| 8 | `family-08-catboost-native-specialist` | inconclusive | dependency unavailable | — | — | — |
| 9 | `family-09-capacity-foundation-context` | inconclusive | bounded Original-2025 probe, n=282 | 0.641844 | 0.622817 | 0.216190 |
| 10 | `family-10-outcome-decomposition` | negative | Original 2022–2025, n=1,108 | 0.654332* | 0.621052* | 0.216147* |

\* Best non-control Family-10 variant by log loss: `decision-finish-specialist-mixture`. It did not satisfy the frozen promotion rule. The direct incumbent control reproduced 0.655235 accuracy, 0.619595 log loss, and 0.215424 Brier.

Family 8 is inconclusive: CatBoost was not evaluated because the required Family-7 matchup dependency was unavailable. That is not negative evidence about CatBoost.

Family 10 attempt 1 remains visible as an unmerged engineering failure at revision `3f4bb5fd193273ac4ed41647d57a2561cbb5ab87`. Twelve component fits completed, but deterministic variant insertion order failed before any combined prediction or metric. Its artifact tree `5B1E7B59DA46BEB630E7ACB9010BC8D4AA52B89CAF4B3D44613E9E12E0CBA185` did not enter the accepted campaign. The explicit successor is the accepted joined Family-10 artifact tree `464D878572FA9758CF732A90C3894E31261CD02B8DAA03F01ECE90308D96EAA8`.

## Recommendation

Keep Family 1 as the development incumbent and research deployment candidate. Do not claim external validation or tune from the retired historical period. Record outcome-unknown post-2026-08-08 prospective predictions, wait for outcomes, then evaluate calibration and accuracy without changing those stored probabilities.
