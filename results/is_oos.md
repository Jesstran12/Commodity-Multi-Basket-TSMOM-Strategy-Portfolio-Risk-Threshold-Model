# In-sample / out-of-sample selection

The lookback `L` is selected on the in-sample net Sharpe ratio (each trial's life up to 2012-12-31, at 5 bps), every other parameter at its a-priori default, and then measured out of sample (2013-01-01 to the end of the data). Only `L` is selected. The headline result stays the a-priori default; this table shows what selecting would have done.

| Lookback | Sharpe IS | Sharpe OOS | IS - OOS | Days IS | Days OOS | Selected | Default |
|---:|---:|---:|---:|---:|---:|---|---|
| 63 | 0.19 | -0.22 | 0.41 | 3156 | 3442 | - | - |
| 126 | 0.26 | 0.07 | 0.19 | 3093 | 3442 | - | - |
| 189 | -0.01 | 0.19 | -0.20 | 3030 | 3442 | - | - |
| 252 | 0.29 | 0.38 | -0.09 | 2968 | 3442 | yes | yes |

## Summary

| Quantity | Value |
|---|---:|
| Selected lookback | 252 |
| Selected: Sharpe in sample | 0.29 |
| Selected: Sharpe out of sample | 0.38 |
| Selected: degradation (IS - OOS) | -0.09 |
| A-priori lookback | 252 |
| A-priori: Sharpe in sample | 0.29 |
| A-priori: Sharpe out of sample | 0.38 |
| A-priori: degradation (IS - OOS) | -0.09 |
| Out of sample, selected minus a-priori | +0.00 |
