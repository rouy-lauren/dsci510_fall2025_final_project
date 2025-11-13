import time
import requests
from typing import Dict, List, Optional
from pathlib import Path
import json
from config import YELP_API_KEY, YELP_SEARCH, YELP_BUSINESS, YELP_REVIEWS, RAW_DIR

def _headers() -> Dict[str, str]:
    if not YELP_API_KEY:
        raise RuntimeError("Missing YELP_API_KEY in environment (.env).")
    return {"Authorization": f"Bearer {YELP_API_KEY}"}

def yelp_search(term: str, location: str, limit: int = 50, pages: int = 1, pause: float = 0.25) -> List[Dict]:
    """
    Paginated Yelp search. Returns list of business dicts.
    """
    out: List[Dict] = []
    limit = min(int(limit), 50)          # never exceed 50 per request
    MAX_OFFSET = 1000                    # Yelp won't allow >= 1000

    for page in range(pages):
        offset = page * limit
        # stop before we exceed Yelp's offset window
        if offset >= MAX_OFFSET:
            print("⚠️ Reached Yelp offset limit (1000). Stopping early.")
            break

        params = {"term": term, "location": location, "limit": limit,"offset": offset}

        try:
            r = requests.get(YELP_SEARCH, headers=_headers(), params=params, timeout=20)
            r.raise_for_status()
        except requests.HTTPError as e:
            # If Yelp returns 400 when approaching offset edge, stop gracefully
            print(f"HTTP {r.status_code} at offset {offset}: {e}. Stopping.")
            break

        payload = r.json() or {}
        businesses = payload.get("businesses", [])

        if not businesses:
            break
        out.extend(businesses)

        # polite pause to avoid rate limiting
        time.sleep(pause)
        if len(businesses) < limit:
            break
    return out

def yelp_business_details(business_id: str) -> Dict:
    r = requests.get(YELP_BUSINESS.format(id=business_id), headers=_headers(), timeout=20)
    r.raise_for_status()
    return r.json()

def yelp_business_reviews(business_id: str) -> Dict:
    r = requests.get(YELP_REVIEWS.format(id=business_id), headers=_headers(), timeout=20)
    r.raise_for_status()
    return r.json()

def save_json(obj, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    return path

def fetch_and_cache(term: str, location: str, limit: int, pages: int) -> Path:
    """
    Fetch search results and cache raw JSON under data/raw/.
    """
    data = yelp_search(term, location, limit=limit, pages=pages)
    raw_path = RAW_DIR / f"yelp_{term.replace(' ','_')}_{location.replace(',','').replace(' ','_')}.json"
    return save_json(data, raw_path)


