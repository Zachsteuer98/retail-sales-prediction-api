"""What the model has actually seen, learned from training data and enforced at inference.

The data covers only 4 stores, and each store's size, city tier, type and age never vary. Store
attributes therefore act as a store ID: a combination outside those 4 is not a prediction, it is a
guess. The schema makes such inputs fail loudly instead of returning a plausible-looking number.
"""

from dataclasses import asdict, dataclass

import pandas as pd

from superkart import config


@dataclass(frozen=True)
class InputSchema:
    stores: dict[str, dict]  # Store_Id -> {Store_Size, Store_Location_City_Type, Store_Type, Store_Age_Years}
    categories: dict[str, list[str]]
    numeric_ranges: dict[str, tuple[float, float]]
    sugar_by_product_prefix: dict[str, list[str]]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "InputSchema":
        return cls(
            stores=data["stores"],
            categories=data["categories"],
            numeric_ranges={k: tuple(v) for k, v in data["numeric_ranges"].items()},
            sugar_by_product_prefix=data["sugar_by_product_prefix"],
        )


def learn_schema(raw: pd.DataFrame, features: pd.DataFrame) -> InputSchema:
    store_rows = pd.concat([raw[["Store_Id"]], features[config.STORE_COLUMNS]], axis=1).drop_duplicates()
    if store_rows["Store_Id"].duplicated().any():
        raise ValueError("A store has more than one attribute combination; schema assumption broken.")
    stores = {
        row.Store_Id: {col: _plain(getattr(row, col)) for col in config.STORE_COLUMNS}
        for row in store_rows.sort_values("Store_Id").itertuples(index=False)
    }
    return InputSchema(
        stores=stores,
        categories={col: sorted(features[col].unique().tolist()) for col in config.CATEGORICAL_FEATURES},
        numeric_ranges={
            col: (float(features[col].min()), float(features[col].max())) for col in config.NUMERIC_FEATURES
        },
        sugar_by_product_prefix={
            prefix: sorted(group.unique().tolist())
            for prefix, group in features.groupby("Product_Id_char")["Product_Sugar_Content"]
        },
    )


def _plain(value):
    return value.item() if hasattr(value, "item") else value


def validate_record(record: dict, schema: InputSchema) -> list[str]:
    """Return human-readable problems with one input row (empty list = valid)."""
    problems: list[str] = []
    for col, allowed in schema.categories.items():
        if record[col] not in allowed:
            problems.append(f"{col}='{record[col]}' was never seen in training; expected one of {allowed}.")
    for col, (low, high) in schema.numeric_ranges.items():
        if not low <= record[col] <= high:
            problems.append(f"{col}={record[col]} is outside the training range [{low}, {high}].")

    store = {col: record[col] for col in config.STORE_COLUMNS}
    if store not in schema.stores.values():
        known = "; ".join(f"{sid}: {attrs}" for sid, attrs in schema.stores.items())
        problems.append(f"Store attributes {store} do not match any of the 4 known stores ({known}).")

    allowed_sugar = schema.sugar_by_product_prefix.get(record["Product_Id_char"], [])
    if record["Product_Id_char"] in schema.sugar_by_product_prefix and (
        record["Product_Sugar_Content"] not in allowed_sugar
    ):
        problems.append(
            f"Product_Id_char='{record['Product_Id_char']}' only appears with sugar content {allowed_sugar}."
        )
    return problems
