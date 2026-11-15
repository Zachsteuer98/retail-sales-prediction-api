import pandas as pd
import pytest

from retail_sales.schema import InputSchema

STORES = {
    "OUT001": {
        "Store_Size": "High",
        "Store_Location_City_Type": "Tier 2",
        "Store_Type": "Supermarket Type1",
        "Store_Age_Years": 30,
    },
    "OUT002": {
        "Store_Size": "Small",
        "Store_Location_City_Type": "Tier 3",
        "Store_Type": "Food Mart",
        "Store_Age_Years": 19,
    },
}


@pytest.fixture
def schema() -> InputSchema:
    return InputSchema(
        stores=STORES,
        categories={
            "Product_Sugar_Content": ["Low Sugar", "No Sugar", "Regular"],
            "Store_Size": ["High", "Small"],
            "Store_Location_City_Type": ["Tier 2", "Tier 3"],
            "Store_Type": ["Food Mart", "Supermarket Type1"],
            "Product_Id_char": ["DR", "FD", "NC"],
            "Product_Type_Category": ["Non Perishables", "Perishables"],
        },
        numeric_ranges={
            "Product_Weight": (4.0, 22.0),
            "Product_Allocated_Area": (0.004, 0.298),
            "Product_MRP": (31.0, 266.0),
            "Store_Age_Years": (8.0, 30.0),
        },
        sugar_by_product_prefix={
            "NC": ["No Sugar"],
            "FD": ["Low Sugar", "Regular"],
            "DR": ["Low Sugar", "Regular"],
        },
    )


@pytest.fixture
def valid_record() -> dict:
    return {
        "Product_Weight": 12.66,
        "Product_Sugar_Content": "Low Sugar",
        "Product_Allocated_Area": 0.027,
        "Product_MRP": 117.08,
        **STORES["OUT001"],
        "Product_Id_char": "FD",
        "Product_Type_Category": "Perishables",
    }


@pytest.fixture
def raw_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Product_Id": ["FD6114", "NC1180", "DR0001"],
            "Product_Weight": [12.66, 9.57, 10.0],
            "Product_Sugar_Content": ["Low Sugar", "No Sugar", "reg"],
            "Product_Allocated_Area": [0.027, 0.010, 0.05],
            "Product_Type": ["Frozen Foods", "Household", "Soft Drinks"],
            "Product_MRP": [117.08, 120.0, 90.0],
            "Store_Id": ["OUT001", "OUT002", "OUT001"],
            "Store_Establishment_Year": [1987, 1998, 1987],
            "Store_Size": ["High", "Small", "High"],
            "Store_Location_City_Type": ["Tier 2", "Tier 3", "Tier 2"],
            "Store_Type": ["Supermarket Type1", "Food Mart", "Supermarket Type1"],
            "Product_Store_Sales_Total": [2842.4, 1500.0, 3000.0],
        }
    )
