import argparse
from config import (DEFAULT_TERM, DEFAULT_LOCATION, DEFAULT_LIMIT, DEFAULT_PAGES,RESULTS_DIR,)
from load import fetch_and_cache
from process import fetch_la_almanac_race_table, la_cities_zipcode, fetch_restaurants_by_zip

def _tag(term: str, location: str) -> str:
    return f"{term}_{location}".replace(" ", "_").replace(",", "").replace("/", "_")

def parse_args():
    p = argparse.ArgumentParser(description="Yelp → CSV → plots")
    p.add_argument("--term", default=DEFAULT_TERM)
    p.add_argument("--location", default=DEFAULT_LOCATION)
    p.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    p.add_argument("--pages", type=int, default=DEFAULT_PAGES)
    return p.parse_args()

def main():
    args = parse_args()
    tag = _tag(args.term, args.location)

    print(f"Searching Yelp for term='{args.term}' in location='{args.location}' ...")
    raw_path = fetch_and_cache(args.term, args.location, args.limit, args.pages)
    print(f"Saved raw JSON → {raw_path}")

    df_demo = fetch_la_almanac_race_table()
    print("✓ LA Almanac race table fetched and processed")
    print(df_demo.head())
    print()

    df_zipcode = la_cities_zipcode()
    print("✓ ZIPCode.com.ng LA cities table fetched and processed")
    print(df_zipcode.head())
    print()

    df_rest = fetch_restaurants_by_zip(term=args.term)
    print("✓ Yelp restaurants by ZIP fetched and processed")
    print(df_rest.head())
    print()


if __name__ == "__main__":
    main()

