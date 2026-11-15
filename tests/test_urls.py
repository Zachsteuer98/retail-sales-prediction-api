import pytest

from superkart.urls import normalize_base_url

EXPECTED = "https://zachsteuer98-retail-sales-api.hf.space"


@pytest.mark.parametrize(
    "raw",
    [
        EXPECTED,
        f"{EXPECTED}/",
        f"  {EXPECTED}  ",
        f'"{EXPECTED}"',
        "zachsteuer98-retail-sales-api.hf.space",  # regression: scheme omitted in Space variable
        f"{EXPECTED} │",  # regression: table border copied from terminal output
        f"│ {EXPECTED} │",
    ],
)
def test_normalize_base_url_fixes_copy_paste_mistakes(raw):
    assert normalize_base_url(raw) == EXPECTED


def test_normalize_base_url_keeps_explicit_http():
    assert normalize_base_url("http://localhost:8000/") == "http://localhost:8000"
