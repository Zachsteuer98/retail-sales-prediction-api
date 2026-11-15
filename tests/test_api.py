import io

import numpy as np
import pandas as pd
import pytest

from api.app import MAX_BATCH_ROWS, create_app


class FakePredictor:
    """Deterministic stand-in: predicted sales = 20 x MRP."""

    model_version = "test"
    model_name = "Fake"

    def __init__(self, schema):
        self.schema = schema

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return features["Product_MRP"].to_numpy() * 20


@pytest.fixture
def client(schema):
    return create_app(FakePredictor(schema)).test_client()


def csv_upload(rows: list[dict]) -> dict:
    data = pd.DataFrame(rows).to_csv(index=False).encode()
    return {"file": (io.BytesIO(data), "batch.csv")}


def test_health(client):
    assert client.get("/health").get_json() == {"status": "ok", "model": "Fake", "model_version": "test"}


def test_schema_lists_known_stores(client):
    assert set(client.get("/v1/schema").get_json()["stores"]) == {"OUT001", "OUT002"}


def test_predict_single(client, valid_record):
    response = client.post("/v1/predict", json=valid_record)
    assert response.status_code == 200
    assert response.get_json() == {"prediction": 2341.6, "model_version": "test"}


@pytest.mark.parametrize(
    "change",
    [
        {"Store_Type": "Supermarket Type3"},  # unknown category
        {"Store_Age_Years": 19},  # store combination that doesn't exist
        {"Product_Weight": 3.5},  # outside training range
        {"Product_MRP": "cheap"},  # wrong type
        {"Discount": 0.1},  # unknown field
    ],
)
def test_predict_rejects_invalid_input_with_details(client, valid_record, change):
    response = client.post("/v1/predict", json={**valid_record, **change})
    assert response.status_code == 422
    assert response.get_json()["details"]["0"]


def test_predict_rejects_missing_field(client, valid_record):
    body = {k: v for k, v in valid_record.items() if k != "Product_MRP"}
    assert client.post("/v1/predict", json=body).status_code == 422


def test_predict_rejects_non_object_body(client):
    assert client.post("/v1/predict", data="not json", content_type="application/json").status_code == 422


def test_batch_predicts_every_row_and_ignores_extra_columns(client, valid_record):
    rows = [
        {**valid_record, "Store_Id": "OUT001", "Actual_Sales": 1.0},
        {**valid_record, "Product_MRP": 100.0},
    ]
    response = client.post("/v1/predictbatch", data=csv_upload(rows))
    assert response.status_code == 200
    assert response.get_json()["predictions"] == [2341.6, 2000.0]


def test_batch_reports_errors_per_row(client, valid_record):
    rows = [valid_record, {**valid_record, "Store_Type": "Supermarket Type3"}]
    response = client.post("/v1/predictbatch", data=csv_upload(rows))
    assert response.status_code == 422
    assert list(response.get_json()["details"]) == ["1"]


def test_batch_rejects_missing_columns(client, valid_record):
    rows = [{k: v for k, v in valid_record.items() if k != "Store_Type"}]
    response = client.post("/v1/predictbatch", data=csv_upload(rows))
    assert response.status_code == 422
    assert "Store_Type" in response.get_json()["error"]


def test_batch_requires_file_and_row_limits(client, valid_record):
    assert client.post("/v1/predictbatch").status_code == 422
    empty = {"file": (io.BytesIO(b""), "empty.csv")}
    assert client.post("/v1/predictbatch", data=empty).status_code == 422
    too_many = csv_upload([valid_record] * (MAX_BATCH_ROWS + 1))
    assert client.post("/v1/predictbatch", data=too_many).status_code == 422


def test_unknown_route_returns_json(client):
    response = client.get("/nope")
    assert response.status_code == 404 and "error" in response.get_json()
