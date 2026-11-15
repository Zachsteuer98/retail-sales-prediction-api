"""v2 training pipeline.

Fixes from the review of the original notebook:
- preprocessing is fit inside each training fold (it was fit on all rows before the split);
- the model is chosen by cross-validated RMSE on the training split (it was chosen on the test set);
- the test set is scored once, for reporting only;
- two simple baselines show how much the tree models actually add.

Usage:
    python -m superkart.train
"""

import json
import logging
from datetime import UTC, datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV, KFold, train_test_split

from superkart import __version__, config
from superkart.evaluation import regression_report
from superkart.features import load_raw, prepare
from superkart.models import candidates
from superkart.plots import save_cv_comparison, save_importance, save_predicted_vs_actual
from superkart.schema import learn_schema

logger = logging.getLogger("superkart.train")
PERMUTATION_REPEATS = 10


def tune(estimator, grid: dict, X: pd.DataFrame, y: pd.Series, folds: KFold) -> GridSearchCV:
    search = GridSearchCV(
        estimator, param_grid=grid or [{}], cv=folds, scoring="neg_root_mean_squared_error", n_jobs=-1
    )
    return search.fit(X, y)


def write_demo_sample(raw: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """Labelled test rows (with store IDs) for the demo; the full dataset stays out of git."""
    sample = X_test.assign(Store_Id=raw.loc[X_test.index, "Store_Id"], Actual_Sales=y_test)
    sample = sample.sample(config.SAMPLE_ROWS, random_state=config.RANDOM_SEED)
    config.SAMPLE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    sample.to_csv(config.SAMPLE_CSV, index=False)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raw = load_raw()
    X, y = prepare(raw)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_FRACTION, random_state=config.RANDOM_SEED
    )
    folds = KFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_SEED)
    n_features = len(config.MODEL_FEATURES)

    results, searches = {}, {}
    for name, (estimator, grid) in candidates().items():
        search = tune(estimator, grid, X_train, y_train, folds)
        best = search.best_index_
        searches[name] = search
        results[name] = {
            "cv_rmse": float(-search.cv_results_["mean_test_score"][best]),
            "cv_rmse_std": float(search.cv_results_["std_test_score"][best]),
            "best_params": {k.replace("regressor__", ""): v for k, v in search.best_params_.items()},
            # Reported for transparency; NOT used for selection.
            "test": regression_report(y_test, search.predict(X_test), n_features),
        }
        logger.info("%-22s CV RMSE %.1f", name, results[name]["cv_rmse"])

    selected = min(results, key=lambda n: results[n]["cv_rmse"])
    logger.info("Selected by CV: %s", selected)
    train_fitted = searches[selected].best_estimator_

    importance = permutation_importance(
        train_fitted,
        X_test,
        y_test,
        scoring="neg_root_mean_squared_error",
        n_repeats=PERMUTATION_REPEATS,
        random_state=config.RANDOM_SEED,
        n_jobs=-1,
    )
    importances = dict(
        sorted(
            zip(X_test.columns, importance.importances_mean.astype(float), strict=True),
            key=lambda kv: -kv[1],
        )
    )

    # Production model: same settings, refit on all labelled rows.
    production = clone(train_fitted).fit(X, y)

    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(production, config.MODEL_FILE, compress=3)
    metadata = {
        "model_version": __version__,
        "trained_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "selected_model": selected,
        "best_params": results[selected]["best_params"],
        "test_metrics": results[selected]["test"],
        "schema": learn_schema(raw, X).to_dict(),
    }
    config.METADATA_FILE.write_text(json.dumps(metadata, indent=2))
    config.METRICS_FILE.write_text(
        json.dumps({"selected": selected, "models": results, "permutation_importance": importances}, indent=2)
    )

    save_cv_comparison(results, selected, config.FIGURES_DIR / "cv_comparison.png")
    save_predicted_vs_actual(
        y_test.to_numpy(),
        np.asarray(train_fitted.predict(X_test)),
        config.FIGURES_DIR / "predicted_vs_actual.png",
    )
    save_importance(importances, config.FIGURES_DIR / "permutation_importance.png")
    write_demo_sample(raw, X_test, y_test)
    logger.info("Test metrics (%s): %s", selected, json.dumps(results[selected]["test"]))


if __name__ == "__main__":
    main()
