"""Flask API for Retail Sales Prediction (product-store sales).

Run locally:   flask --app api.app run --port 7860
Production:    gunicorn "api.app:create_app()" --bind 0.0.0.0:7860
"""

import io
import logging
from typing import Annotated

import pandas as pd
from flask import Flask, jsonify, request
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from werkzeug.exceptions import HTTPException

from retail_sales import config
from retail_sales.predictor import JoblibSalesPredictor, SalesPredictor
from retail_sales.schema import validate_record

logger = logging.getLogger("retail_sales.api")

MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_BATCH_ROWS = 1000
FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]


class SalesRecord(BaseModel):
    """One product in one store. Field names match the original API contract."""

    model_config = ConfigDict(extra="forbid")

    Product_Weight: FiniteFloat
    Product_Sugar_Content: str
    Product_Allocated_Area: FiniteFloat
    Product_MRP: FiniteFloat
    Store_Size: str
    Store_Location_City_Type: str
    Store_Type: str
    Product_Id_char: str
    Store_Age_Years: int
    Product_Type_Category: str


class InvalidInputError(Exception):
    def __init__(self, message: str, details: dict | list | None = None) -> None:
        super().__init__(message)
        self.message, self.details = message, details


def parse_records(rows: list[dict], predictor: SalesPredictor) -> pd.DataFrame:
    """Type-check and domain-check every row; collect all problems before rejecting."""
    records, errors = [], {}
    for i, row in enumerate(rows):
        try:
            record = SalesRecord.model_validate(row).model_dump()
        except ValidationError as exc:
            errors[str(i)] = [f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()]
            continue
        problems = validate_record(record, predictor.schema)
        if problems:
            errors[str(i)] = problems
        records.append(record)
    if errors:
        raise InvalidInputError("Input failed validation.", errors)
    return pd.DataFrame(records, columns=config.MODEL_FEATURES)


def read_csv_upload() -> list[dict]:
    upload = request.files.get("file")
    if upload is None or not upload.filename:
        raise InvalidInputError("Attach a CSV file under the form field 'file'.")
    try:
        df = pd.read_csv(io.BytesIO(upload.read()))
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        raise InvalidInputError(f"Could not parse CSV: {exc}") from exc
    if not 0 < len(df) <= MAX_BATCH_ROWS:
        raise InvalidInputError(f"CSV must contain 1-{MAX_BATCH_ROWS} rows.")
    missing = [c for c in config.MODEL_FEATURES if c not in df.columns]
    if missing:
        raise InvalidInputError(f"Missing columns: {missing}")
    # Extra columns (e.g. Store_Id, Actual_Sales) are allowed in files and ignored.
    return df[config.MODEL_FEATURES].to_dict(orient="records")


def create_app(predictor: SalesPredictor | None = None) -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
    model = predictor or JoblibSalesPredictor()
    logger.info("Loaded %s (version %s)", model.model_name, model.model_version)

    @app.errorhandler(InvalidInputError)
    def invalid_input(exc: InvalidInputError):
        return jsonify({"error": exc.message, "details": exc.details}), 422

    @app.errorhandler(HTTPException)
    def http_error(exc: HTTPException):
        return jsonify({"error": exc.description}), exc.code

    @app.errorhandler(Exception)
    def unexpected(exc: Exception):
        logger.exception("Unhandled error")  # details go to logs, never to the client
        return jsonify({"error": "Internal server error."}), 500

    @app.get("/")
    @app.get("/health")
    def health():
        return jsonify({"status": "ok", "model": model.model_name, "model_version": model.model_version})

    @app.get("/v1/schema")
    def schema():
        """The 4 known stores and the accepted values, so clients never hard-code them."""
        return jsonify(model.schema.to_dict())

    @app.post("/v1/predict")
    def predict():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            raise InvalidInputError("Send one JSON object with the 10 feature fields.")
        features = parse_records([payload], model)
        return jsonify(
            {"prediction": round(float(model.predict(features)[0]), 2), "model_version": model.model_version}
        )

    @app.post("/v1/predictbatch")
    def predict_batch():
        features = parse_records(read_csv_upload(), model)
        predictions = model.predict(features)
        return jsonify(
            {
                "model_version": model.model_version,
                "predictions": [round(float(p), 2) for p in predictions],
            }
        )

    return app
