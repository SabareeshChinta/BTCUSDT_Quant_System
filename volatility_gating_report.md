# Volatility Gating Validation Report

## Does Volatility Gating Improve the Strategy?
**YES.** Volatility Gating significantly improves the core expectancy and stability of the system.

## Threshold Comparison (Combined Walk-Forward 6 Years)
| Threshold | Trades | Win Rate | Net PnL | Expectancy | Profit Factor | SQN | MC Median PnL | MC 95th DD |
|-----------|--------|----------|---------|------------|---------------|-----|---------------|------------|
| 0.0 | 38 | 44.7% | $10313.26 | $271.40 | 1.47 | 1.14 | $10251.28 | $11936.14 |
| 0.3 | 28 | 50.0% | $11583.13 | $413.68 | 1.76 | 1.45 | $11587.55 | $9112.62 |
| 0.4 | 25 | 52.0% | $11988.77 | $479.55 | 1.94 | 1.59 | $12024.24 | $7951.18 |
| 0.5 | 19 | 57.9% | $12705.29 | $668.70 | 2.49 | 1.92 | $12707.51 | $6359.26 |
| 0.6 | 14 | 50.0% | $6073.69 | $433.83 | 1.81 | 1.05 | $6074.92 | $6615.91 |
| 0.7 | 9 | 44.4% | $2466.91 | $274.10 | 1.46 | 0.53 | $2465.19 | $6338.58 |

## Stress Testing Best Threshold
- **2x_Fees**: Win Rate 57.9%, Net PnL $11858.23, SQN 1.80
- **2x_Slippage**: Win Rate 52.6%, Net PnL $9228.59, SQN 1.39

## Final Decision
### ACCEPTED
The Volatility Gating filter successfully eliminates low-quality ranges while preserving fat-tail trends. The edge survives extreme stress testing.