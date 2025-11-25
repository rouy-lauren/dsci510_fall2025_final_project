from pathlib import Path
from dotenv import load_dotenv
import os

# load secret env
env_path = Path(__file__).resolve().parent.parent / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)

# project dirs (relative to repo root)
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = ROOT / "results"

# create folders if missing
for p in [DATA_DIR, RAW_DIR, PROCESSED_DIR, RESULTS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# Yelp API
YELP_API_KEY = os.getenv("YELP_API_KEY", "")
YELP_HOST = "https://api.yelp.com"
YELP_SEARCH = f"{YELP_HOST}/v3/businesses/search"
YELP_BUSINESS = f"{YELP_HOST}/v3/businesses/{{id}}"
YELP_REVIEWS = f"{YELP_HOST}/v3/businesses/{{id}}/reviews"

# sensible defaults for your proposal (LA restaurants)
DEFAULT_TERM = "restaurants"
DEFAULT_LOCATION = "Los Angeles, CA"
DEFAULT_LIMIT = 50   # per page (max 50)
DEFAULT_PAGES = 20   # 20*50 = 1000 results

# LA Almanac demographics table
LA_RACE_URL = "https://www.laalmanac.com/population/po38.php"

# LA cities and ZIP codes table
ZIPCODE_LA_URL = "https://www.zipcode.com.ng/2022/06/los-angeles-zip-codes.html"
