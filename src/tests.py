# include your tests here 
# for example for your Progress report you should be able to load data from at least one API source.
"""
Progress-report smoke tests:
 - verifies YELP_API_KEY exists
 - performs a tiny live request (limit=200) and checks schema
Run:  python -m src.tests
"""
import os
import json
from pathlib import Path
import pandas as pd

from config import YELP_API_KEY, RAW_DIR, PROCESSED_DIR, RESULTS_DIR
from load import fetch_and_cache
from process import flatten_businesses
from analyze import analyze_yelp_data

def _tag(term: str, location: str) -> str:
    return f"{term}_{location}".replace(" ", "_").replace(",", "").replace("/", "_")

def test_has_api_key():
    assert YELP_API_KEY, "YELP_API_KEY missing. Create .env with your key."

def test_tiny_fetch_and_process():
    raw = fetch_and_cache(term="restaurants", location="Los Angeles, CA", limit=50, pages=1)
    assert Path(raw).exists(), "Raw JSON not saved."

    # verify JSON structure minimally
    data = json.loads(Path(raw).read_text())
    assert isinstance(data, list) and len(data) >= 1, "No businesses returned."

    csv_path = flatten_businesses(Path(raw))
    df = pd.read_csv(csv_path)
    # expected columns for later modeling/EDA
    expected = {"id","name","rating","review_count","price","categories","latitude","longitude","city","zip_code","address","is_closed"}
    assert expected.issubset(set(df.columns)), "Processed CSV missing expected columns."
    assert len(df) >= 1, "Processed CSV is empty."

def test_analyze_outputs():
    # small end-to-end run (reuse tiny fetch to keep the call fast)
    raw = fetch_and_cache(term="restaurants", location="Los Angeles, CA", limit=50, pages=1)
    csv_path = flatten_businesses(Path(raw))

    tag = _tag("restaurants", "Los Angeles, CA")
    df = analyze_yelp_data(str(csv_path), dataset_name=tag)
    assert len(df) >= 1, "Analysis received empty dataframe."

    # check a couple representative outputs exist
    heatmap_png = Path(RESULTS_DIR) / f"{tag}_correlation_heatmap.png"
    feat_csv = Path(RESULTS_DIR) / f"{tag}_feature_importance.csv"

    assert heatmap_png.exists(), "Correlation heatmap not created."
    assert feat_csv.exists() and feat_csv.stat().st_size > 0, "Feature importance CSV missing or empty."

if __name__ == "__main__":
    # simple runner without pytest
    try:
        test_has_api_key()
        print("✓ API key present")
        test_tiny_fetch_and_process()
        print("✓ Live fetch + process passed")
        test_analyze_outputs()
        print("✓ Analyze step produced outputs")
        print("ALL TESTS PASSED")
    except AssertionError as e:
        print(f"TEST FAILED: {e}")
        raise

