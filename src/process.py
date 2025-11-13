from pathlib import Path
from typing import Iterable, Dict
import json
import pandas as pd
from config import RAW_DIR, PROCESSED_DIR

def _read_json_list(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def flatten_businesses(raw_json_path: Path) -> Path:
    """
    Turn Yelp 'businesses' JSON into a tidy CSV for modeling/EDA.
    """
    businesses = _read_json_list(raw_json_path)
    rows = []
    for b in businesses:
        rows.append({
            "id": b.get("id"),
            "name": b.get("name"),
            "rating": b.get("rating"),
            "review_count": b.get("review_count"),
            "price": b.get("price"),
            "categories": ", ".join([c.get("title") for c in b.get("categories", [])]),
            "latitude": (b.get("coordinates") or {}).get("latitude"),
            "longitude": (b.get("coordinates") or {}).get("longitude"),
            "city": (b.get("location") or {}).get("city"),
            "zip_code": (b.get("location") or {}).get("zip_code"),
            "address": " ".join((b.get("location") or {}).get("display_address", [])),
            "is_closed": b.get("is_closed"),
        })
    df = pd.DataFrame(rows)
    out_path = PROCESSED_DIR / "yelp_restaurants.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return out_path
