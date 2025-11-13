import argparse
from config import (
    DEFAULT_TERM, DEFAULT_LOCATION, DEFAULT_LIMIT, DEFAULT_PAGES,RESULTS_DIR,
)
from load import fetch_and_cache
from process import flatten_businesses
from analyze import analyze_yelp_data

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

    csv_path = flatten_businesses(raw_path)
    print(f"Saved processed CSV → {csv_path}")

    print("Running analysis on processed Yelp data ...")
    analyze_yelp_data(str(csv_path), dataset_name=tag)
    print(f"Done. See results in: {RESULTS_DIR}")



if __name__ == "__main__":
    main()
