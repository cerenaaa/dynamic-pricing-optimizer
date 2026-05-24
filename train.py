"""
Estimate elasticities and optimize pricing.
Usage: python train.py
"""
import pathlib
from data.synthetic_subscriptions import generate_subscription_data
from elasticity.ols_elasticity import ElasticityModel
from optimizer.price_optimizer import RevenueOptimizer, PricingConstraints


def main():
    pathlib.Path("results").mkdir(exist_ok=True)

    print("Generating subscription data...")
    df = generate_subscription_data()

    print("\nEstimating price elasticities by tier:")
    model = ElasticityModel()
    elasticities = model.fit(df, group_col="tier")

    current = df.groupby("tier").agg(
        avg_price=("price", "mean"),
        avg_demand=("demand", "mean"),
    )
    current_prices = current["avg_price"].to_dict()
    current_volumes = current["avg_demand"].to_dict()

    print("\nOptimizing prices...")
    optimizer = RevenueOptimizer(
        elasticities=elasticities,
        current_prices=current_prices,
        current_volumes=current_volumes,
        constraints=PricingConstraints(min_price=0.80, max_price=1.25),
    )
    results = optimizer.optimize()

    print("\n" + "=" * 60)
    print("PRICING OPTIMIZATION RESULTS")
    print("=" * 60)
    print(results.to_string(index=False))
    total_lift = (results["optimal_revenue"].sum() / results["current_revenue"].sum() - 1) * 100
    print(f"\nTotal portfolio revenue lift: +{total_lift:.1f}%")

    results.to_csv("results/pricing_recommendations.csv", index=False)
    print("\n✓ Saved to results/pricing_recommendations.csv")


if __name__ == "__main__":
    main()