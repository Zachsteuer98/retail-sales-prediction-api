"""Central configuration: paths, feature schema, and training settings."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
SAMPLE_DATA_DIR = PROJECT_ROOT / "data" / "sample"
ARTIFACTS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

DATA_CSV = RAW_DATA_DIR / "SuperKart.csv"
SAMPLE_CSV = SAMPLE_DATA_DIR / "sample_batch.csv"
MODEL_FILE = ARTIFACTS_DIR / "model.joblib"
METADATA_FILE = ARTIFACTS_DIR / "metadata.json"
METRICS_FILE = REPORTS_DIR / "metrics.json"

TARGET_COLUMN = "Product_Store_Sales_Total"

# The dataset is a 2017 snapshot; store age is measured from that year, as in the notebook.
REFERENCE_YEAR = 2017

NUMERIC_FEATURES: list[str] = ["Product_Weight", "Product_Allocated_Area", "Product_MRP", "Store_Age_Years"]
CATEGORICAL_FEATURES: list[str] = [
    "Product_Sugar_Content",
    "Store_Size",
    "Store_Location_City_Type",
    "Store_Type",
    "Product_Id_char",
    "Product_Type_Category",
]
MODEL_FEATURES: list[str] = [*NUMERIC_FEATURES, *CATEGORICAL_FEATURES]
STORE_COLUMNS: list[str] = ["Store_Size", "Store_Location_City_Type", "Store_Type", "Store_Age_Years"]

PERISHABLE_PRODUCT_TYPES = frozenset(
    {
        "Dairy",
        "Frozen Foods",
        "Fruits and Vegetables",
        "Meat",
        "Baking Goods",
        "Bread",
        "Breakfast",
        "Seafood",
        "Starchy Foods",
    }
)
SUGAR_LABEL_FIXES = {"reg": "Regular"}

RANDOM_SEED = 42
TEST_FRACTION = 0.3  # same 70/30 split as the original notebook
CV_FOLDS = 5
SAMPLE_ROWS = 25
