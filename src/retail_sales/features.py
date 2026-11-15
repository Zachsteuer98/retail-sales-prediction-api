"""Cleaning and feature engineering, shared by training and the API so they cannot drift apart."""

from pathlib import Path

import pandas as pd

from retail_sales import config


def load_raw(path: Path = config.DATA_CSV) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}. See README > Data.")
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Fix inconsistent labels: 108 rows use 'reg' for 'Regular' sugar content."""
    return df.assign(Product_Sugar_Content=df["Product_Sugar_Content"].replace(config.SUGAR_LABEL_FIXES))


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Derive the model's inputs from raw columns (same rules as the original notebook)."""
    return df.assign(
        Store_Age_Years=config.REFERENCE_YEAR - df["Store_Establishment_Year"],
        Product_Id_char=df["Product_Id"].str[:2],
        Product_Type_Category=df["Product_Type"].map(
            lambda t: "Perishables" if t in config.PERISHABLE_PRODUCT_TYPES else "Non Perishables"
        ),
    )


def prepare(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Raw dataframe -> (model features, target)."""
    ready = engineer(clean(df))
    return ready[config.MODEL_FEATURES], ready[config.TARGET_COLUMN]
