# Robustness: trials, grid and sweeps

## Trials and the Deflated Sharpe Ratio

| Quantity | Value |
|---|---:|
| Trials (configurations evaluated) | 192 |
| Std of the trials' net Sharpe (5 bps) | 0.157 |
| Sharpe hurdle (expected max of the trials) | 0.433 |
| Probabilistic Sharpe of the default (net, vs 0) | 0.958 |
| Deflated Sharpe of the default (net) | 0.323 |

## The default among the trials (net Sharpe at 5 bps)

| Quantity | Value |
|---|---:|
| Minimum | -0.29 |
| Lower quartile | -0.00 |
| Median | 0.12 |
| Upper quartile | 0.23 |
| Maximum | 0.48 |
| Share of trials with Sharpe > 0 | 74.5% |
| Default | 0.34 |
| Default rank (1 = best) | 12 of 192 |
| Best trial | L 252 / vol 120 rolling / daily / skip 0 / lag 1 |
| Worst trial | L 63 / vol 20 rolling / daily / skip 0 / lag 2 |

## Cost sweep (default configuration; 0 bps is the gross result)

| Cost (bps) | Ann. return | Sharpe | Max drawdown | Grid median Sharpe | Grid share Sharpe > 0 |
|---:|---:|---:|---:|---:|---:|
| 0 | 3.45% | 0.37 | -39.46% | 0.20 | 96.35% |
| 5 | 3.17% | 0.34 | -40.42% | 0.12 | 74.48% |
| 10 | 2.88% | 0.32 | -41.37% | 0.02 | 54.17% |
| 20 | 2.30% | 0.26 | -43.22% | -0.11 | 36.46% |

## Lag sweep (default configuration)

| Lag (days) | Sharpe gross | Sharpe net | Ann. return net | Grid median Sharpe net |
|---:|---:|---:|---:|---:|
| 1 | 0.37 | 0.34 | 3.17% | 0.15 |
| 2 | 0.31 | 0.29 | 2.56% | 0.07 |
