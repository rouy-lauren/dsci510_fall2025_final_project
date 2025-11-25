from pathlib import Path
from typing import Iterable, Dict
import json
import pandas as pd
import unicodedata
import requests
import re
from bs4 import BeautifulSoup
from config import RAW_DIR, PROCESSED_DIR, LA_RACE_URL, ZIPCODE_LA_URL
from load import yelp_search

def _read_json_list(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def fetch_la_almanac_race_table(save: bool = True) -> pd.DataFrame:
    """
    Fetch LA Almanac racial composition table, clean multi-row headers,
    remove percentage rows, and return a tidy DataFrame.
    """
    import pandas as pd

    # Read the HTML tables
    tables = pd.read_html(LA_RACE_URL)
    df = pd.concat(tables, ignore_index=True)

    # Find the "City / Community" column 
    name_cols = [c for c in df.columns if "City / Community" in str(c)]
    if not name_cols:
        raise RuntimeError(
            f"Could not find 'City / Community' column. Columns: {df.columns}"
        )
    name_col = name_cols[0]

    # Keep only rows that have an actual city/community name
    df = df[df[name_col].notna()]
    df = df[df[name_col] != "City / Community"]  # drop repeated header rows

    # Remove percentage ROWS: any cell in the row containing '%'
    has_percent = df.astype(str).apply(lambda col: col.str.contains("%"))
    df = df[~has_percent.any(axis=1)].copy()

    other_cols = [c for c in df.columns if c != name_col]
    value_cols = other_cols[:9]  # first 9 numeric columns in the block
    df = df[[name_col] + value_cols].copy()

    # remove footnote markers like ‡
    df[name_col] = (
        df[name_col]
        .astype(str)
        .str.replace("‡", "", regex=False)
        .str.strip()
    )
    df[name_col] = df[name_col].replace({"La Ca√±ada Flintridge": "La Cañada Flintridge"})

    # remove commas, cast to numeric
    numeric_cols = value_cols
    for col in numeric_cols:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.strip()
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # assign clean, meaningful column names
    df.columns = [
        "city_community",
        "total_population",
        "pop_american_indian_alaska_native",
        "pop_asian",
        "pop_black_african_american",
        "pop_native_hawaiian_pacific_islander",
        "pop_white_non_hispanic",
        "pop_some_other_race",
        "pop_two_or_more_races",
        "pop_hispanic_or_latino",
    ]

    # drop empty city rows 
    df = df[df["city_community"].str.strip() != ""].reset_index(drop=True)

    if save:
        out_path = PROCESSED_DIR / "la_almanac_race_counts.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"Saved LA Almanac demographics table to: {out_path}")
    return df

def la_cities_zipcode(save: bool = True) -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
      - city_community
      - zip_code
    """
    # download page
    resp = requests.get(ZIPCODE_LA_URL, timeout=20)
    resp.raise_for_status()

    # get plain text
    soup = BeautifulSoup(resp.text, "html.parser")
    text = soup.get_text(" ")

    # normalize whitespace
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"([A-Za-z])(\d{5})", r"\1 \2", text)
    text = text.replace("California City ZIP Code County State ", "")

    # regex: CityName ZIP Los Angeles California
    pattern = re.compile(r"([A-Z][A-Za-z\s\.\-']*?)\s+(\d{5})\s+Los Angeles\s+California")
    matches = pattern.findall(text)

    rows = []
    for city, z in matches:
        city = city.strip()
        zip_code = z.strip().zfill(5)
        rows.append({"city_community": city, "zip_code": zip_code})

    df = pd.DataFrame(rows, columns=["city_community", "zip_code"]).drop_duplicates()

    if save:
        out_path = PROCESSED_DIR / "la_cities_zipcodes.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"Saved LA city/ZIP list to: {out_path}")
        print("✓ ZIPCode.com.ng LA cities table fetched and processed")

    return df

def fetch_restaurants_by_zip(term: str = "restaurants", pages: int = 4, limit: int = 50, save: bool = True) -> pd.DataFrame:
    """
    Use Yelp API to fetch restaurants by ZIP code.
    """
    # 1) get all zip codes from your scraper
    zip_df = la_cities_zipcode(save=True)
    zip_list = sorted(zip_df["zip_code"].unique())

    rows = []

    for z in zip_list:
        location = f"{z}, CA"        # Yelp accepts ZIP as location
        print(f"Fetching Yelp for ZIP {location} ...")
        businesses = yelp_search(term, location=location, limit=limit, pages=pages)

        for b in businesses:
            rows.append(
                {
                    "zip_code": z,
                    "name": b.get("name"),
                    "rating": b.get("rating"),
                    "review_count": b.get("review_count"),
                    "price": b.get("price"),
                    "categories": ", ".join([c.get("title") for c in b.get("categories", [])]),
                    "latitude": (b.get("coordinates") or {}).get("latitude"),
                    "longitude": (b.get("coordinates") or {}).get("longitude"),
                    "city": (b.get("location") or {}).get("city"),
                    "address": " ".join((b.get("location") or {}).get("display_address", [])),
                    "is_closed": b.get("is_closed"),
                }
            )

    df = pd.DataFrame(rows)

    if save:
        out_path = PROCESSED_DIR / "yelp_restaurants_by_zip.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"Saved Yelp restaurants-by-ZIP to: {out_path}")

    return df
