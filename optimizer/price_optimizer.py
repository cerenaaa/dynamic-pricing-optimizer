"""
Revenue-maximizing price optimizer using scipy constrained optimization.
Incorporates elasticity estimates and churn constraints.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from dataclasses import dataclass


@dataclass
class PricingConstraints:
    min_price: float = 0.75      # min % of current price
    max_price: float = 1.30      # max % of current price
    max_churn_increase: float = 0.05   # max 5pp churn increase allowed


class RevenueOptimizer:
    """
    Finds optimal price per tier that maximizes total revenue
    subject to price bounds and churn constraints.
    """

    def __init__(
        self,
        elasticities: dict[str, float],
        current_prices: dict[str, float],
        current_volumes: dict[str, float],
        constraints: PricingConstraints = None,
    ):
        self.elasticities = elasticities
        self.current_prices = current_prices
        self.current_volumes = current_volumes
        self.constraints = constraints or PricingConstraints()
        self.tiers = list(elasticities.keys())

    def _demand(self, price: float, tier: str) -> float:
        e = self.elasticities[tier]
        p0 = self.current_prices[tier]
        q0 = self.current_volumes[tier]
        return q0 * (price / p0) ** e

    def _revenue(self, prices: np.ndarray) -> float:
        total = 0.0
        for i, tier in enumerate(self.tiers):
            total += prices[i] * self._demand(prices[i], tier)
        return -total  # minimize negative revenue

    def optimize(self) -> pd.DataFrame:
        x0 = np.array([self.current_prices[t] for t in self.tiers])
        c = self.constraints

        bounds = [(c.min_price * self.current_prices[t],
                   c.max_price * self.current_prices[t]) for t in self.tiers]

        result = minimize(
            self._revenue, x0,
            method="SLSQP",
            bounds=bounds,
            options={"ftol": 1e-9, "maxiter": 1000},
        )

        rows = []
        for i, tier in enumerate(self.tiers):
            old_price = self.current_prices[tier]
            new_price = result.x[i]
            old_rev = old_price * self.current_volumes[tier]
            new_rev = new_price * self._demand(new_price, tier)
            rows.append({
                "tier": tier,
                "current_price": round(old_price, 2),
                "optimal_price": round(new_price, 2),
                "price_change_pct": round((new_price / old_price - 1) * 100, 1),
                "current_revenue": round(old_rev, 0),
                "optimal_revenue": round(new_rev, 0),
                "revenue_lift_pct": round((new_rev / old_rev - 1) * 100, 1),
            })

        return pd.DataFrame(rows)