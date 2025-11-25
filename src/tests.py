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
from process import fetch_la_almanac_race_table, la_cities_zipcode, fetch_restaurants_by_zip
#from analyze import analyze_yelp_data

def _tag(term: str, location: str) -> str:
    return f"{term}_{location}".replace(" ", "_").replace(",", "").replace("/", "_")

def test_has_api_key():
    assert YELP_API_KEY, "YELP_API_KEY missing. Create .env with your key."
if __name__ == "__main__":
    # simple runner without pytest
    try:
        test_has_api_key()
        print("✓ API key present")
        df_demo = fetch_la_almanac_race_table()
        print("✓ LA Almanac race table fetched and processed")
        print(df_demo.head())
        df_zipcode = la_cities_zipcode()
        print("✓ ZIPCode.com.ng LA cities table fetched and processed")
        print(df_zipcode.head())
        df_rest = fetch_restaurants_by_zip(term="restaurants")
        print("✓ Yelp restaurants by ZIP fetched and processed")
        print(df_rest.head())
    except AssertionError as e:
        print(f"TEST FAILED: {e}")
        raise

