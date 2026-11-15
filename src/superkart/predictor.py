"""Inference: load the trained pipeline and its input schema once."""

import json
from pathlib import Path
from typing import Protocol

import joblib
import numpy as np
import pandas as pd

from superkart import config
from superkart.schema import InputSchema


class SalesPredictor(Protocol):
    """Interface the API depends on, so tests can use a lightweight fake."""

    model_version: str
    model_name: str
    schema: InputSchema

    def predict(self, features: pd.DataFrame) -> np.ndarray: ...


class JoblibSalesPredictor:
    def __init__(self, artifacts_dir: Path = config.ARTIFACTS_DIR) -> None:
        metadata_path = artifacts_dir / config.METADATA_FILE.name
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"No trained artifacts in {artifacts_dir}. Run `python -m superkart.train` first."
            )
        metadata = json.loads(metadata_path.read_text())
        self.model_version: str = metadata["model_version"]
        self.model_name: str = metadata["selected_model"]
        self.schema = InputSchema.from_dict(metadata["schema"])
        self._pipeline = joblib.load(artifacts_dir / config.MODEL_FILE.name)

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return np.asarray(self._pipeline.predict(features[config.MODEL_FEATURES]))
