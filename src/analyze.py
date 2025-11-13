import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error
from config import RESULTS_DIR

"""
Analyze Yelp restaurant data: correlations, regression models,
and feature importance to identify drivers of ratings.
"""

def analyze_yelp_data(yelp_csv_path: str, dataset_name: str = "Yelp_LA", notebook_plot: bool = False) -> pd.DataFrame:
    """
    Analyze Yelp restaurant data and compute correlations + feature importance.
    :param yelp_csv_path: Path to processed Yelp CSV
    :param dataset_name: Dataset label for file naming
    :param notebook_plot: Whether to display plots inline (e.g., in Jupyter)
    :return: Cleaned DataFrame with analysis summary
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load and clean
    df = pd.read_csv(yelp_csv_path)
    df = df.dropna(subset=["rating"])
    df["price_level"] = df["price"].map({"$": 1, "$$": 2, "$$$": 3, "$$$$": 4})
    df["review_count"] = pd.to_numeric(df["review_count"], errors="coerce")
    df = df.dropna(subset=["price_level", "review_count"])

    # Compute Correlation Matrix for All Numeric Variables
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    corr = df[numeric_cols].corr(method="pearson")
    corr_path = os.path.join(RESULTS_DIR, f"{dataset_name}_correlations.csv")
    corr.to_csv(corr_path)
    print(f"Saved correlation matrix to {corr_path}")

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title(f"Correlation Heatmap - {dataset_name}")
    plt.tight_layout()
    if notebook_plot:
        plt.show()
    else:
        plt.savefig(os.path.join(RESULTS_DIR, f"{dataset_name}_correlation_heatmap.png"))
    plt.close()

    # Encode Categorical Variables
    cat_cols = ["categories", "city", "zip_code"]
    for col in cat_cols:
        if col not in df.columns:
            print(f"Warning: {col} not found in data, skipping.")
    cat_cols = [c for c in cat_cols if c in df.columns]

    features = ["review_count", "price_level"] + cat_cols
    X = df[features].copy()
    y = df["rating"]

    # Build Pipelines for Regression and Random Forest
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", max_categories=10)
    preprocessor = ColumnTransformer(
        transformers=[("cat", categorical_transformer, cat_cols)], remainder="passthrough"
    )

    # Multiple Linear Regression 
    linreg_model = Pipeline(
        steps=[("preprocessor", preprocessor), ("model", LinearRegression())]
    )
    linreg_model.fit(X, y)
    y_pred_lin = linreg_model.predict(X)
    linreg_r2 = r2_score(y, y_pred_lin)
    linreg_mae = mean_absolute_error(y, y_pred_lin)
    print(f"\nLinear Regression: R²={linreg_r2:.3f}, MAE={linreg_mae:.3f}")

    # Random Forest Regressor 
    rf_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", RandomForestRegressor(n_estimators=100, random_state=42)),
        ]
    )
    rf_model.fit(X, y)
    y_pred_rf = rf_model.predict(X)
    rf_r2 = r2_score(y, y_pred_rf)
    rf_mae = mean_absolute_error(y, y_pred_rf)
    print(f"Random Forest: R²={rf_r2:.3f}, MAE={rf_mae:.3f}")

    # Feature Importance (Random Forest)
    # Get transformed feature names
    feature_names = (
        rf_model.named_steps["preprocessor"]
        .transformers_[0][1]
        .get_feature_names_out(cat_cols)
    )
    all_features = list(feature_names) + ["review_count", "price_level"]
    importances = rf_model.named_steps["model"].feature_importances_

    importance_df = (
        pd.DataFrame({"Feature": all_features, "Importance": importances})
        .sort_values("Importance", ascending=False)
    )

    importance_path = os.path.join(RESULTS_DIR, f"{dataset_name}_feature_importance.csv")
    importance_df.to_csv(importance_path, index=False)
    print(f"Saved feature importances to {importance_path}")

    # Plot Feature Importances
    plt.figure(figsize=(8, 6))
    sns.barplot(data=importance_df.head(15), x="Importance", y="Feature")
    plt.title(f"Top 15 Feature Importances - {dataset_name}")
    plt.tight_layout()
    if notebook_plot:
        plt.show()
    else:
        plt.savefig(os.path.join(RESULTS_DIR, f"{dataset_name}_feature_importance.png"))
    plt.close()

    print(f"\nAnalysis complete. Results saved in: {RESULTS_DIR}")
    return df
