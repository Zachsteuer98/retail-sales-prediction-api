"""Streamlit demo for the Retail Sales Prediction model. All predictions go through the Flask API.

Run locally:
    streamlit run app/streamlit_app.py
"""

import io
import json
import logging
import os

import pandas as pd
import requests
import streamlit as st

from retail_sales import config
from retail_sales.urls import normalize_base_url

logger = logging.getLogger("retail_sales.app")

API_URL = normalize_base_url(os.environ.get("API_URL", "http://localhost:7860"))
REPO_URL = normalize_base_url(
    os.environ.get("REPO_URL", "https://github.com/Zachsteuer98/retail-sales-prediction-api")
)
ORIGINAL_BATCH_CSV = config.SAMPLE_DATA_DIR / "original_brief_batch.csv"
REQUEST_TIMEOUT_SECONDS = 90  # a sleeping free-tier API Space needs time to cold-start
PRODUCT_PREFIX_LABELS = {"FD": "FD (food)", "DR": "DR (drinks)", "NC": "NC (non-consumables)"}

st.set_page_config(page_title="Retail Sales Prediction", page_icon="🛒", layout="wide")


def call_api(method: str, path: str, **kwargs) -> tuple[int | None, dict | None]:
    try:
        with st.spinner("Calling the prediction API (first call may take a minute while it wakes up)..."):
            response = requests.request(method, f"{API_URL}{path}", timeout=REQUEST_TIMEOUT_SECONDS, **kwargs)
    except requests.RequestException as exc:
        logger.exception("API call to %s%s failed", API_URL, path)
        st.error(f"Prediction service at {API_URL} is unreachable ({type(exc).__name__}). Try again shortly.")
        return None, None
    try:
        return response.status_code, response.json()
    except ValueError:
        st.error(
            f"API returned {response.status_code} without JSON; it may still be starting. Retry shortly."
        )
        return response.status_code, None


@st.cache_data(ttl=600, show_spinner=False)
def load_schema() -> dict | None:
    try:
        response = requests.get(f"{API_URL}/v1/schema", timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        logger.exception("Could not load schema from API")
        return None


@st.cache_data
def load_metrics() -> dict:
    return json.loads(config.METRICS_FILE.read_text())


def show_validation_errors(body: dict) -> None:
    st.error(body.get("error", "Request rejected."))
    for row, problems in (body.get("details") or {}).items():
        for problem in problems:
            st.markdown(f"- **Row {row}:** {problem}")


def render_header() -> None:
    st.title("🛒 Retail Sales Prediction")
    st.markdown(
        "Predicts the total sales of a product in one of a retail chain's stores from product attributes "
        "(price, weight, display area) and which store it sells in. "
        f"[Source code & notebook]({REPO_URL}) · [API health]({API_URL}/health)"
    )
    metrics = load_metrics()
    test = metrics["models"][metrics["selected"]]["test"]
    cols = st.columns(4)
    cols[0].metric("Model", metrics["selected"])
    cols[1].metric("Test R²", f"{test['r2']:.3f}")
    cols[2].metric("Test RMSE", f"{test['rmse']:,.0f}")
    cols[3].metric("Test MAPE", f"{test['mape']:.1%}")
    st.caption("Held-out 30% test set (2,629 product-store rows). Selected by 5-fold cross-validation.")


def render_single(schema: dict) -> None:
    st.subheader("Predict sales for one product in one store")
    stores = schema["stores"]
    store_id = st.selectbox(
        "Store",
        list(stores),
        format_func=lambda s: f"{s}: {stores[s]['Store_Type']}, {stores[s]['Store_Size']} size, "
        f"{stores[s]['Store_Location_City_Type']}, {stores[s]['Store_Age_Years']} yrs",
    )
    st.caption(
        "The model was trained on these 4 stores only, so you pick a real store, not free-form attributes."
    )

    ranges, cats = schema["numeric_ranges"], schema["categories"]
    c1, c2 = st.columns(2)
    with c1:
        mrp = st.slider("Product MRP (max retail price)", *map(float, ranges["Product_MRP"]), 147.0)
        weight = st.slider("Product weight", *map(float, ranges["Product_Weight"]), 12.6)
        area = st.slider(
            "Share of store display area", *map(float, ranges["Product_Allocated_Area"]), 0.056, 0.001
        )
    with c2:
        prefix = st.selectbox(
            "Product ID prefix", cats["Product_Id_char"], format_func=PRODUCT_PREFIX_LABELS.get
        )
        sugar_options = schema["sugar_by_product_prefix"][prefix]
        sugar = st.selectbox(
            "Sugar content", sugar_options, help="Only values seen with this prefix are offered."
        )
        category = st.selectbox("Product type category", cats["Product_Type_Category"])

    if st.button("Predict sales", type="primary"):
        payload = {
            "Product_Weight": weight,
            "Product_Sugar_Content": sugar,
            "Product_Allocated_Area": area,
            "Product_MRP": mrp,
            **stores[store_id],
            "Product_Id_char": prefix,
            "Product_Type_Category": category,
        }
        status, body = call_api("POST", "/v1/predict", json=payload)
        if status == 200 and body:
            st.metric("Predicted product-store sales", f"{body['prediction']:,.2f}")
        elif body:
            show_validation_errors(body)
        with st.expander("Raw API request / response"):
            st.json({"request": payload, "response": body}, expanded=False)


def score_file(name: str, data: bytes) -> None:
    status, body = call_api("POST", "/v1/predictbatch", files={"file": (name, data, "text/csv")})
    if status == 200 and body:
        df = pd.read_csv(io.BytesIO(data))
        df["Predicted_Sales"] = body["predictions"]
        if "Actual_Sales" in df:
            df["Abs_Pct_Error"] = (df["Predicted_Sales"] - df["Actual_Sales"]).abs() / df["Actual_Sales"]
            st.success(f"{len(df)} rows scored · mean absolute % error {df['Abs_Pct_Error'].mean():.1%}")
        else:
            st.success(f"{len(df)} rows scored")
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button(
            "Download results", df.to_csv(index=False).encode(), "retail_sales_predictions.csv"
        )
    elif body:
        show_validation_errors(body)


def render_batch() -> None:
    st.subheader("Score a CSV of products")
    choice = st.radio(
        "Data",
        ["Demo sample (25 held-out rows)", "Original brief's batch file", "Upload my own CSV"],
        horizontal=True,
    )
    if choice.startswith("Demo"):
        if st.button("Score demo sample", type="primary"):
            score_file(config.SAMPLE_CSV.name, config.SAMPLE_CSV.read_bytes())
    elif choice.startswith("Original"):
        st.info(
            "The case study shipped this file for batch scoring. Every row describes a store the model has "
            "never seen (including a 'Supermarket Type3'). The original API silently scored it; v2 refuses "
            "and explains why."
        )
        if st.button("Send original batch file", type="primary"):
            score_file(ORIGINAL_BATCH_CSV.name, ORIGINAL_BATCH_CSV.read_bytes())
    else:
        st.caption(f"Columns required: {', '.join(config.MODEL_FEATURES)}. Extra columns are ignored.")
        upload = st.file_uploader("CSV file", type="csv")
        if upload is not None:
            score_file(upload.name, upload.getvalue())


def render_model_card() -> None:
    metrics = load_metrics()
    st.subheader("How the model was chosen")
    rows = [
        {
            "Model": name,
            "CV RMSE (selection)": round(r["cv_rmse"], 1),
            "Test RMSE": round(r["test"]["rmse"], 1),
            "Test R²": round(r["test"]["r2"], 3),
            "Test MAPE": f"{r['test']['mape']:.1%}",
        }
        for name, r in sorted(metrics["models"].items(), key=lambda kv: kv[1]["cv_rmse"])
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.markdown(
        """
- **Selection:** lowest 5-fold cross-validated RMSE on the training split; the test set is scored once.
- **What drives sales:** store (location, size, type) and product price and weight. Sugar content,
  product category and display area add almost nothing.
- **Limits:** data covers 4 stores and has no dates, so this predicts product-store sales totals,
  not a time-series forecast, and cannot speak for new stores.
"""
    )
    cols = st.columns(3)
    for col, figure in zip(
        cols, ["cv_comparison", "permutation_importance", "predicted_vs_actual"], strict=True
    ):
        col.image(str(config.FIGURES_DIR / f"{figure}.png"))


def main() -> None:
    render_header()
    tabs = st.tabs(["🔍 Single prediction", "📋 Batch scoring", "📄 Model card"])
    with tabs[0]:
        schema = load_schema()
        if schema:
            render_single(schema)
        else:
            st.error("Could not load store list from the API. It may be waking up; refresh in a minute.")
    with tabs[1]:
        render_batch()
    with tabs[2]:
        render_model_card()


main()
