import os
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error
from math import sqrt
from config import RESULTS_DIR, PROCESSED_DIR

"""
Analyze Yelp restaurant data: correlations, regression models,
and feature importance to identify drivers of ratings.
"""

# Load & clean Yelp data
GENERIC_CATEGORIES = {
    "Food",
    "Restaurants",
    "Food Trucks",
    "Bars",
    "Nightlife",
    "Breakfast & Brunch",
    "Cafes",
    "Coffee & Tea",
    "Fast Food",
}

def extract_main_category(cat_str: str):
    if not isinstance(cat_str, str) or not cat_str.strip():
        return np.nan

    cats = [c.strip() for c in cat_str.split(",") if c.strip()]

    # iterate backwards for a specific category
    for c in reversed(cats):
        if c not in GENERIC_CATEGORIES:
            return c

    # fallback if all categories are generic
    return cats[0] if cats else np.nan

def load_and_clean_yelp(path: Path = PROCESSED_DIR / "yelp_restaurants_by_zip.csv") -> pd.DataFrame:
    """
    Load yelp_restaurants.csv and perform basic cleaning/feature engineering.

    - Drop rows with missing price (since price_level is important).
    - Convert price ($, $$, $$$) to numeric price_level.
    - Extract main_category from categories.
    - Compute log_review_count.
    - Drop rows missing essential info.
    """
    df = pd.read_csv(path)

    # keep only open businesses
    if "is_closed" in df.columns:
        df = df[df["is_closed"] == False]
    # Convert to numeric
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["review_count"] = pd.to_numeric(df["review_count"], errors="coerce")
    # Drop rows with no price
    df = df.dropna(subset=["price"])

    # price_level: $ -> 1, $$ -> 2, ...
    df["price_level"] = df["price"].map(
        lambda x: len(str(x)) if pd.notna(x) else np.nan
    )
    # main_category = first category string
    df["main_category"] = df["categories"].map(extract_main_category)

    # popularity proxy
    df["log_review_count"] = np.log1p(df["review_count"])

    # drop rows missing essential info
    df = df.dropna(subset=["rating", "zip_code", "latitude", "longitude", "price_level"])

    # drop duplicate businesses and unnecessary columns
    if "address" in df.columns:
        df = df.drop_duplicates(subset=["address"])
    df = df.drop(columns=["review_count", "price", "categories","is_closed"], errors="ignore")


    print(f"Yelp data loaded: {len(df)} rows after cleaning")
    return df

# Load ZIP mapping and LA Almanac demographics, aggregate by ZIP
def load_la_almanac_race_table(path: Path = PROCESSED_DIR / "la_almanac_race_counts.csv") -> pd.DataFrame:
    """
    Load LA Almanac racial composition table.
    """
    df_race = pd.read_csv(path)
    df_race = df_race.rename(columns={"city_community": "city"})
    return df_race  

def build_full_dataset() -> pd.DataFrame:
    """
    Merge cleaned Yelp data with LA Almanac demographics
    using only city names.
    """

    # Load cleaned Yelp data
    yelp_df = load_and_clean_yelp()

    # Load LA Almanac demographics
    dem_df = load_la_almanac_race_table()

    # Merge
    full_df = yelp_df.merge(
        dem_df,
        on="city",
        how="left"
    )
    full_df = full_df.dropna(subset=["total_population"]).reset_index(drop=True)

    print(f"Full merged dataset: {len(full_df)} restaurant-level rows")
    return full_df

def build_zip_level_dataset(full_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate restaurant-level full_df to ZIP-code level and compute
    neighborhood features like diversity and restaurant density.

    Expected columns in full_df:
        - zip_code
        - rating
        - price_level
        - log_review_count
        - latitude, longitude
        - city
        - main_category
        - total_population
        - pop_american_indian_alaska_native
        - pop_asian
        - pop_black_african_american
        - pop_native_hawaiian_pacific_islander
        - pop_white_non_hispanic
        - pop_some_other_race
        - pop_two_or_more_races
        - pop_hispanic_or_latino
    """

    df = full_df.copy()

    # --- 1. Basic ZIP-level aggregates (ratings / counts / location) ---
    race_cols = [
        "pop_american_indian_alaska_native",
        "pop_asian",
        "pop_black_african_american",
        "pop_native_hawaiian_pacific_islander",
        "pop_white_non_hispanic",
        "pop_some_other_race",
        "pop_two_or_more_races",
        "pop_hispanic_or_latino",
    ]

    agg_basic = (
        df.groupby("zip_code")
        .agg(
            avg_rating=("rating", "mean"),
            avg_price_level=("price_level", "mean"),
            avg_log_reviews=("log_review_count", "mean"),
            n_restaurants=("name", "count"),
            # representative location = mean lat/lon
            latitude=("latitude", "mean"),
            longitude=("longitude", "mean"),
            # most common city name in this ZIP
            city=("city", lambda x: x.mode().iat[0] if not x.mode().empty else x.iloc[0]),
        )
        .reset_index()
    )

    # --- 2. Attach demographic counts (they are city-level, same within city) ---
    # Use "first" since within a city they are identical; ZIPs may contain one main city.
    pop_agg = (
        df.groupby("zip_code")[["total_population"] + race_cols]
        .first()
        .reset_index()
    )

    zip_df = agg_basic.merge(pop_agg, on="zip_code", how="left")

    # Drop ZIPs with missing population
    zip_df = zip_df.dropna(subset=["total_population"])

    # --- 3. Race shares and racial diversity index ---
    for c in race_cols:
        share_col = c + "_share"
        zip_df[share_col] = zip_df[c] / zip_df["total_population"].replace(0, np.nan)

    share_cols = [c + "_share" for c in race_cols]

    # Simpson diversity index: 1 - sum(p_i^2)
    zip_df["racial_diversity"] = 1.0 - (zip_df[share_cols] ** 2).sum(axis=1)

    # --- 4. Cuisine diversity per ZIP (Simpson index over cuisines) ---
    cuisine_counts = (
        df.groupby(["zip_code", "main_category"])
        .size()
        .reset_index(name="count")
    )

    cuisine_counts["prop"] = (
        cuisine_counts["count"]
        / cuisine_counts.groupby("zip_code")["count"].transform("sum")
    )

    cuisine_div = (
        cuisine_counts.groupby("zip_code")["prop"]
        .apply(lambda p: 1.0 - np.sum(p**2))
        .reset_index(name="cuisine_diversity")
    )

    zip_df = zip_df.merge(cuisine_div, on="zip_code", how="left")

    # --- 5. Restaurant density features ---
    # restaurants per 10,000 residents
    zip_df["restaurants_per_10k"] = (
        zip_df["n_restaurants"] / zip_df["total_population"].replace(0, np.nan) * 10000
    )

    # Average reviews per restaurant as a separate intensity feature
    zip_df["avg_reviews_per_restaurant"] = np.exp(zip_df["avg_log_reviews"]) - 1

    # Clean up any remaining inf/nan from division
    zip_df = zip_df.replace([np.inf, -np.inf], np.nan)

    print(f"ZIP-level dataset built: {len(zip_df)} ZIP codes")
    return zip_df

def run_zip_level_models(zip_df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42,) -> dict:
    """
    Run Linear Regression and Random Forest on ZIP-level data.
    Target:
        - avg_rating

    Features:
        - avg_price_level
        - avg_log_reviews
        - restaurants_per_10k
        - racial_diversity
        - cuisine_diversity
        - selected racial share columns

    Returns a dict with metrics, coefficients, and feature importances.
    """

    feature_cols = [
        "avg_price_level",
        "avg_log_reviews",
        "restaurants_per_10k",
        "racial_diversity",
        "cuisine_diversity",
        "pop_white_non_hispanic_share",
        "pop_asian_share",
        "pop_black_african_american_share",
        "pop_hispanic_or_latino_share",
    ]

    # check columns
    missing = [c for c in feature_cols + ["avg_rating"] if c not in zip_df.columns]
    if missing:
        raise ValueError(f"Missing columns in zip_df: {missing}")

    X = zip_df[feature_cols].fillna(0)
    y = zip_df["avg_rating"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # ----- Linear Regression -----
    linreg = LinearRegression()
    linreg.fit(X_train, y_train)

    y_pred_lr = linreg.predict(X_test)
    r2_lr = r2_score(y_test, y_pred_lr)
    rmse_lr = np.sqrt(np.mean((y_test - y_pred_lr) ** 2))


    coef_series = pd.Series(linreg.coef_, index=feature_cols)

    print("=== ZIP-level Linear Regression ===")
    print(f"R^2:  {r2_lr:.4f}")
    print(f"RMSE: {rmse_lr:.4f}")

    # ----- Random Forest Regressor -----
    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        random_state=random_state,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    y_pred_rf = rf.predict(X_test)
    r2_rf = r2_score(y_test, y_pred_rf)
    rmse_rf = np.sqrt(np.mean((y_test - y_pred_rf) ** 2))

    rf_importances = pd.Series(rf.feature_importances_, index=feature_cols)

    print("\n=== ZIP-level Random Forest Regressor ===")
    print(f"R^2:  {r2_rf:.4f}")
    print(f"RMSE: {rmse_rf:.4f}")

    results = {
        "feature_cols": feature_cols,
        "linear": {
            "model": linreg,
            "r2": r2_lr,
            "rmse": rmse_lr,
            "coefficients": coef_series.sort_values(ascending=False),
        },
        "random_forest": {
            "model": rf,
            "r2": r2_rf,
            "rmse": rmse_rf,
            "importances": rf_importances.sort_values(ascending=False),
        },
    }

    return results

def prepare_model_features(full_df: pd.DataFrame, top_n_cuisines: int = 15):
    """
    From the merged full_df, build an ML-ready feature matrix X and target y.
    Features:
      - price_level
      - log_review_count (popularity proxy)
      - total_population (rough proxy for density)
      - racial proportions (each race / total_population)
      - one-hot encoded main_category (top N frequent cuisines)

    Returns
    -------
    X : np.ndarray
    y : np.ndarray
    feature_names : list[str]
    df_model : pd.DataFrame (cleaned subset you can also use for correlations)
    """

    df = full_df.copy()

    # 1) Drop rows with missing essentials
    required_cols = [
        "rating", "price_level", "log_review_count",
        "total_population", "main_category", "city",
        "latitude", "longitude"
    ]
    race_cols = [
        "pop_american_indian_alaska_native",
        "pop_asian",
        "pop_black_african_american",
        "pop_native_hawaiian_pacific_islander",
        "pop_white_non_hispanic",
        "pop_some_other_race",
        "pop_two_or_more_races",
        "pop_hispanic_or_latino",
    ]
    required_cols += race_cols

    df = df.dropna(subset=required_cols)

    # Racial proportions (neighborhood diversity)
    for col in race_cols:
        share_col = col + "_share"
        df[share_col] = df[col] / df["total_population"].replace(0, np.nan)

    share_cols = [c + "_share" for c in race_cols]

    # Top-N cuisines (main_category) one-hot encoding
    top_cuisines = (
        df["main_category"]
        .value_counts()
        .head(top_n_cuisines)
        .index
    )
    df["main_category_clean"] = df["main_category"].where(
        df["main_category"].isin(top_cuisines),
        other="Other"
    )
    cuisine_dummies = pd.get_dummies(
        df["main_category_clean"],
        prefix="cuisine"
    )

    # Assemble final feature matrix
    numeric_features = [
        "price_level",
        "log_review_count",
        "total_population",
    ] + share_cols

    df_model = pd.concat([df[numeric_features], cuisine_dummies, df["rating"]], axis=1)

    # Drop any remaining NaNs
    df_model = df_model.dropna(subset=["rating"])

    X = df_model.drop(columns=["rating"]).values
    y = df_model["rating"].values
    feature_names = df_model.drop(columns=["rating"]).columns.tolist()

    return X, y, feature_names, df_model

def correlation_and_heatmap(df_model: pd.DataFrame):
    """
    Compute correlations and plot a heatmap.

    df_model is the output df_model from prepare_model_features.
    """

    corr = df_model.corr(numeric_only=True)

    # Print correlations with rating (sorted)
    print("Correlation with rating:")
    print(corr["rating"].sort_values(ascending=False))

    # Heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap (Rating vs Features)")
    plt.tight_layout()
    plt.show()

    return corr

def run_regression_models(full_df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Fit Linear Regression and Random Forest models to predict rating.
    """

    X, y, feature_names, df_model = prepare_model_features(full_df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # ---------------- Linear Regression ----------------
    linreg = LinearRegression()
    linreg.fit(X_train, y_train)
    y_pred_lr = linreg.predict(X_test)

    print("\n=== Linear Regression ===")
    print("R^2:", r2_score(y_test, y_pred_lr))
    print("RMSE:", np.sqrt(np.mean((y_test - y_pred_lr) ** 2)))

    # Coefficients as feature importance
    coef_series = pd.Series(linreg.coef_, index=feature_names).sort_values(ascending=False)
    print("\nTop Linear Coefficients (positive influence on rating):")
    print(coef_series.head(15))
    print("\nMost negative coefficients (lower ratings):")
    print(coef_series.tail(15))

    # ---------------- Random Forest Regressor ----------------
    rf = RandomForestRegressor(
        n_estimators=300,
        random_state=random_state,
        n_jobs=-1,
        max_depth=None
    )
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    print("\n=== Random Forest Regressor ===")
    print("R^2:", r2_score(y_test, y_pred_rf))
    print("RMSE:", np.sqrt(np.mean((y_test - y_pred_rf) ** 2)))

    importances = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=False)
    print("\nTop Random Forest feature importances:")
    print(importances.head(20))

    # Plot feature importances (top 20)
    plt.figure(figsize=(10, 6))
    importances.head(20).sort_values().plot(kind="barh")
    plt.title("Random Forest Feature Importance (Top 20)")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.show()

    return {
        "df_model": df_model,
        "corr": df_model.corr(numeric_only=True),
        "linreg": linreg,
        "rf": rf,
        "linreg_coefs": coef_series,
        "rf_importances": importances,
    }

def plot_rating_by_cuisine(full_df: pd.DataFrame, top_n: int = 15):
    avg_by_cuisine = (
        full_df.groupby("main_category")["rating"]
        .agg(["mean", "count"])
        .sort_values("mean", ascending=False)
        .head(top_n)
    )

    plt.figure(figsize=(10, 6))
    avg_by_cuisine["mean"].sort_values().plot(kind="barh")
    plt.xlabel("Average Rating")
    plt.title(f"Average Rating by Cuisine Type (Top {top_n})")
    plt.tight_layout()
    plt.show()

    return avg_by_cuisine

def plot_rating_by_city(full_df: pd.DataFrame, min_restaurants: int = 20, top_n: int = 25):
    city_stats = (
        full_df.groupby("city")["rating"]
        .agg(["mean", "count"])
    )
    city_stats = city_stats[city_stats["count"] >= min_restaurants]
    city_stats = city_stats.sort_values("mean", ascending=False).head(top_n)

    plt.figure(figsize=(12, 6))
    city_stats["mean"].sort_values().plot(kind="barh")
    plt.xlabel("Average Rating")
    plt.title(f"Average Rating by City (≥{min_restaurants} restaurants, top {top_n})")
    plt.tight_layout()
    plt.show()

    return city_stats

def plot_geo_ratings(full_df: pd.DataFrame, sample: int = 5000):
    """
    Simple geographic scatterplot of restaurants colored by rating.
    """
    df = full_df.dropna(subset=["latitude", "longitude", "rating"])

    if sample and len(df) > sample:
        df = df.sample(sample, random_state=42)

    plt.figure(figsize=(8, 8))
    sc = plt.scatter(
        df["longitude"],
        df["latitude"],
        c=df["rating"],
        cmap="viridis",
        s=10,
        alpha=0.6
    )
    plt.colorbar(sc, label="Rating")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Restaurant Ratings Across Los Angeles")
    plt.tight_layout()
    plt.show()

def plot_zip_scatter(zip_df, x_col, y_col="avg_rating", title=None):
    """
    Simple ZIP-level scatter plot for rating vs a predictor.
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    plt.scatter(zip_df[x_col], zip_df[y_col], alpha=0.6)
    plt.xlabel(x_col.replace("_", " ").title())
    plt.ylabel("Average Rating")
    plt.title(title or f"{y_col} vs {x_col}")
    plt.tight_layout()
    plt.show()

def plot_zip_correlation_heatmap(zip_df):
    """
    Correlation heatmap for ZIP-level dataset.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    features = [
        "avg_rating",
        "avg_log_reviews",
        "avg_price_level",
        "restaurants_per_10k",
        "racial_diversity",
        "cuisine_diversity",
        "pop_white_non_hispanic_share",
        "pop_black_african_american_share",
        "pop_asian_share",
        "pop_hispanic_or_latino_share",
    ]

    corr = zip_df[features].corr()

    plt.figure(figsize=(8, 6))
    plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(label="Correlation")

    plt.xticks(range(len(features)), features, rotation=90)
    plt.yticks(range(len(features)), features)

    plt.title("Correlation Heatmap (ZIP-Level Features)")
    plt.tight_layout()
    plt.show()

def plot_top_zip_ratings(zip_df, top_n=20):
    """
    Bar chart showing top ZIP codes by average rating.
    """
    import matplotlib.pyplot as plt

    top_zip = zip_df.sort_values("avg_rating", ascending=False).head(top_n)

    plt.figure(figsize=(10, 6))
    plt.barh(top_zip["zip_code"].astype(str), top_zip["avg_rating"])
    plt.gca().invert_yaxis()
    plt.xlabel("Average Rating")
    plt.ylabel("ZIP Code")
    plt.title(f"Top {top_n} ZIP Codes by Average Rating")
    plt.tight_layout()
    plt.show()



