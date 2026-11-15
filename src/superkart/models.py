"""Candidate models: two simple baselines plus the tree ensembles from the original notebook."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from superkart import config


class StoreMeanBaseline(BaseEstimator, RegressorMixin):
    """Predicts each store's average sales. Anything useful must beat this."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "StoreMeanBaseline":
        keys = X[config.STORE_COLUMNS].astype(str).agg("|".join, axis=1)
        self.means_ = pd.Series(np.asarray(y), index=keys.values).groupby(level=0).mean()
        self.global_mean_ = float(np.mean(y))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        keys = X[config.STORE_COLUMNS].astype(str).agg("|".join, axis=1)
        return keys.map(self.means_).fillna(self.global_mean_).to_numpy()


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", StandardScaler(), config.NUMERIC_FEATURES),
            # The API rejects unknown categories before they reach this, so "ignore" is never silent.
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), config.CATEGORICAL_FEATURES),
        ]
    )


def with_preprocessing(regressor) -> Pipeline:
    return Pipeline([("preprocessor", build_preprocessor()), ("regressor", regressor)])


def candidates() -> dict[str, tuple[object, dict]]:
    """name -> (estimator, param grid). Grids for RF and XGBoost match the original notebook."""
    seed = config.RANDOM_SEED
    return {
        "Store-average baseline": (StoreMeanBaseline(), {}),
        "Linear regression": (with_preprocessing(LinearRegression()), {}),
        "Decision tree": (
            with_preprocessing(DecisionTreeRegressor(random_state=seed)),
            {"regressor__max_depth": [6, 8, 10], "regressor__min_samples_leaf": [1, 5]},
        ),
        "Random forest": (
            with_preprocessing(RandomForestRegressor(random_state=seed, n_jobs=-1)),
            {
                "regressor__n_estimators": [100, 150],
                "regressor__max_depth": [6, 8, 10],
                "regressor__min_samples_split": [2, 5],
                "regressor__min_samples_leaf": [1, 2],
            },
        ),
        "Gradient boosting": (
            with_preprocessing(GradientBoostingRegressor(random_state=seed)),
            {"regressor__n_estimators": [100, 150], "regressor__max_depth": [3, 4]},
        ),
        "XGBoost": (
            with_preprocessing(XGBRegressor(random_state=seed, objective="reg:squarederror", n_jobs=-1)),
            {
                "regressor__n_estimators": [100, 150],
                "regressor__max_depth": [4, 6, 8],
                "regressor__learning_rate": [0.05, 0.1, 0.2],
                "regressor__subsample": [0.8, 1.0],
            },
        ),
    }
