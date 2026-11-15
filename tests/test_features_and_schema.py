import numpy as np
import pytest

from superkart import config
from superkart.evaluation import adjusted_r2, regression_report
from superkart.features import prepare
from superkart.models import StoreMeanBaseline
from superkart.schema import learn_schema, validate_record


def test_prepare_engineers_features_and_fixes_labels(raw_rows):
    X, y = prepare(raw_rows)
    assert list(X.columns) == config.MODEL_FEATURES
    assert X["Store_Age_Years"].tolist() == [30, 19, 30]
    assert X["Product_Id_char"].tolist() == ["FD", "NC", "DR"]
    assert X["Product_Type_Category"].tolist() == ["Perishables", "Non Perishables", "Non Perishables"]
    assert "reg" not in X["Product_Sugar_Content"].values  # regression: 'reg' was its own category
    assert y.tolist() == [2842.4, 1500.0, 3000.0]


def test_learn_schema_maps_each_store_to_its_attributes(raw_rows):
    X, _ = prepare(raw_rows)
    schema = learn_schema(raw_rows, X)
    assert set(schema.stores) == {"OUT001", "OUT002"}
    assert schema.stores["OUT002"]["Store_Type"] == "Food Mart"
    assert schema.sugar_by_product_prefix["NC"] == ["No Sugar"]


def test_learn_schema_rejects_store_with_two_configurations(raw_rows):
    raw_rows.loc[2, "Store_Size"] = "Small"
    X, _ = prepare(raw_rows)
    with pytest.raises(ValueError, match="more than one"):
        learn_schema(raw_rows, X)


def test_valid_record_has_no_problems(valid_record, schema):
    assert validate_record(valid_record, schema) == []


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        ({"Store_Type": "Supermarket Type3"}, "never seen"),  # regression: silently one-hot ignored
        ({"Product_Weight": 3.5}, "outside the training range"),  # regression: value from the batch file
        ({"Store_Age_Years": 19}, "do not match any of the 4 known stores"),
        ({"Product_Id_char": "NC"}, "only appears with sugar content"),
    ],
)
def test_invalid_records_explain_the_problem(valid_record, schema, change, expected):
    problems = validate_record({**valid_record, **change}, schema)
    assert any(expected in p for p in problems)


def test_store_mean_baseline_predicts_store_average(raw_rows):
    X, y = prepare(raw_rows)
    preds = StoreMeanBaseline().fit(X, y).predict(X)
    np.testing.assert_allclose(preds, [2921.2, 1500.0, 2921.2])


def test_regression_report_perfect_predictions():
    report = regression_report([100.0, 200.0, 300.0, 400.0], [100.0, 200.0, 300.0, 400.0], n_features=1)
    assert report["rmse"] == 0 and report["r2"] == 1 and report["mape"] == 0


def test_adjusted_r2_requires_enough_rows():
    with pytest.raises(ValueError):
        adjusted_r2(0.9, n_rows=3, n_features=2)
