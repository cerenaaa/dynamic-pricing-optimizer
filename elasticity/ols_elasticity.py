"""
Log-log OLS demand elasticity model.
Estimates price elasticity per tier and segment.
Price elasticity = % change in demand / % change in price.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import statsmodels.api as sm
from statsmodels.iolib.summary2 import summary_col


class ElasticityModel:
    """
    Estimates log-log OLS: log(demand) = α + ε·log(price) + controls
    ε is the price elasticity of demand (negative for normal goods).
    """

    def __init__(self):
        self.models: dict[str, sm.OLS] = {}
        self.elasticities: dict[str, float] = {}

    def fit(self, df: pd.DataFrame, group_col: str = "tier") -> dict[str, float]:
        for group, gdf in df.groupby(group_col):
            log_demand = np.log(gdf["demand"].clip(1))
            log_price = np.log(gdf["price"])

            X = sm.add_constant(pd.DataFrame({
                "log_price": log_price,
                "period": gdf["period"],
            }))

            model = sm.OLS(log_demand, X).fit(cov_type="HC3")
            elasticity = model.params["log_price"]
            self.models[group] = model
            self.elasticities[group] = round(elasticity, 4)
            print(f"  {group}: elasticity = {elasticity:.3f} "
                  f"(p={model.pvalues['log_price']:.3f}, R²={model.rsquared:.3f})")

        return self.elasticities

    def summary(self) -> str:
        return summary_col(list(self.models.values()),
                           model_names=list(self.models.keys()),
                           stars=True).as_text()

    def predict_demand_change(self, tier: str, price_change_pct: float) -> float:
        """
        Estimate % demand change from a % price change.
        price_change_pct: e.g. 0.10 = 10% price increase.
        """
        e = self.elasticities.get(tier, -1.0)
        return e * price_change_pct