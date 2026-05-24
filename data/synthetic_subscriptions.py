"""
Synthetic subscription + pricing dataset for demand elasticity modeling.
Simulates a wellness plan product with three tiers and price variation history.
"""
import numpy as np
import pandas as pd

TIERS = ["Basic", "Standard", "Premium"]
REGIONS = ["NA", "EMEA", "APAC"]
SEGMENTS = ["Young_Pet_Owner", "Senior_Pet_Owner", "Multi_Pet_Household"]


def generate_subscription_data(n_markets: int = 200, n_periods: int = 24, seed: int = 42) -> pd.DataFrame:
    """
    Panel dataset: market × month × tier.
    Includes price variation (from tests + organic changes) and demand outcomes.
    """
    rng = np.random.default_rng(seed)
    records = []

    # True elasticities by tier (log-log model)
    true_elasticity = {"Basic": -1.8, "Standard": -1.3, "Premium": -0.9}
    base_demand = {"Basic": 500, "Standard": 300, "Premium": 150}
    base_price = {"Basic": 29, "Standard": 49, "Premium": 89}

    for market_id in range(n_markets):
        region = rng.choice(REGIONS)
        segment = rng.choice(SEGMENTS)
        market_effect = rng.normal(0, 0.15)

        for tier in TIERS:
            for t in range(n_periods):
                # Price varies: some markets get price tests
                in_price_test = (market_id % 5 == 0) and (t >= 12)
                price_multiplier = rng.uniform(0.85, 1.15) if in_price_test else rng.uniform(0.95, 1.05)
                price = base_price[tier] * price_multiplier

                # Demand: log-log model
                log_demand = (
                    np.log(base_demand[tier])
                    + true_elasticity[tier] * np.log(price / base_price[tier])
                    + market_effect
                    + 0.005 * t  # slight trend
                    + rng.normal(0, 0.1)
                )
                demand = max(1, int(np.exp(log_demand)))
                revenue = price * demand

                records.append({
                    "market_id": market_id,
                    "region": region,
                    "segment": segment,
                    "tier": tier,
                    "period": t,
                    "price": round(price, 2),
                    "demand": demand,
                    "revenue": round(revenue, 2),
                    "in_price_test": int(in_price_test),
                    "treatment": int(in_price_test and price > base_price[tier]),
                })

    df = pd.DataFrame(records)
    print(f"Generated {len(df):,} observations | {n_markets} markets | {n_periods} periods")
    return df


if __name__ == "__main__":
    df = generate_subscription_data()
    df.to_csv("data/subscriptions.csv", index=False)
    print(df.groupby("tier")[["price", "demand", "revenue"]].mean().round(2))