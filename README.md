# Dynamic Pricing Optimizer

[![CI](https://github.com/cerenaaa/dynamic-pricing-optimizer/actions/workflows/ci.yml/badge.svg)](https://github.com/cerenaaa/dynamic-pricing-optimizer/actions)

Demand elasticity estimation and dynamic pricing optimization for subscription products. Originally built for Mars Petcare Wellness Plan subscriptions — contributed to **$5M in incremental revenue** via price point optimization.

## Problem

Given observed subscription prices and demand, estimate price elasticity by customer segment and product tier, then optimize price points to maximize revenue subject to churn and volume constraints.

## Approach

| Stage | Method |
|---|---|
| Elasticity estimation | Log-log OLS regression per segment, IV for endogeneity |
| Causal identification | Difference-in-differences on historical price tests |
| Optimization | Scipy constrained optimization (SLSQP) over price grid |
| Simulation | Monte Carlo demand simulation under price scenarios |

## Structure

```
dynamic-pricing-optimizer/
├── data/
│   └── synthetic_subscriptions.py    # Subscription + pricing dataset
├── elasticity/
│   ├── ols_elasticity.py             # Log-log OLS elasticity model
│   └── did_estimator.py              # Diff-in-diff causal identification
├── optimizer/
│   └── price_optimizer.py            # Revenue-maximizing price optimizer
├── simulation/
│   └── demand_simulator.py           # Monte Carlo demand/revenue simulation
├── train.py
└── requirements.txt
```

## Results

| Segment | Old Price | Optimal Price | Revenue Lift |
|---|---|---|---|
| Premium | $89 | $99 | +8.4% |
| Standard | $49 | $54 | +6.1% |
| Basic | $29 | $27 | +3.2% (volume gain) |

## Quickstart
```bash
pip install -r requirements.txt
python train.py
```