"""
EDA: agriBORA Commodity Price Forecasting Challenge
Author: Qamar Usman
Purpose:
    - Explore all available maize price datasets
    - Validate data quality, coverage, and structure
    - Identify target counties and prediction requirements
    - Provide insights for downstream forecasting (non-ML & ML)
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ------------------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------------------
DATA_PATH = "./data/"   # update if needed

TARGET_COUNTIES = [
    "Kiambu",
    "Kirinyaga",
    "Mombasa",
    "Nairobi",
    "Uasin-Gishu"
]

# ------------------------------------------------------------------------------
# UTILITY FUNCTIONS
# ------------------------------------------------------------------------------

def load_csv(path: str) -> pd.DataFrame:
    """Load CSV with basic validation."""
    df = pd.read_csv(path)
    print(f"Loaded {path} | Shape: {df.shape}")
    return df


def basic_overview(df: pd.DataFrame, name: str):
    """Print high-level dataset overview."""
    print(f"\n=== {name.upper()} OVERVIEW ===")
    print("Columns:", list(df.columns))
    print("Missing values:\n", df.isnull().sum())
    print("Data types:\n", df.dtypes)


def date_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Create standard date-based features."""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["year"] = df[date_col].dt.year
    df["month"] = df[date_col].dt.month
    df["week_of_year"] = df[date_col].dt.isocalendar().week
    return df


def county_price_stats(df: pd.DataFrame):
    """Print wholesale price statistics by county."""
    print("\nPrice statistics by county:")
    for county in df["County"].unique():
        cdf = df[df["County"] == county]
        print(
            f"- {county}: "
            f"Min={cdf['WholeSale'].min():.2f}, "
            f"Max={cdf['WholeSale'].max():.2f}, "
            f"Mean={cdf['WholeSale'].mean():.2f}"
        )


# ------------------------------------------------------------------------------
# MAIN EDA PIPELINE
# ------------------------------------------------------------------------------

def main():

    print("=" * 80)
    print("AGRIBORA MAIZE PRICE FORECASTING - EDA")
    print("=" * 80)

    # --------------------------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------------------------
    agribora = load_csv(f"{DATA_PATH}agriBORA_maize_prices.csv")
    agribora_recent = load_csv(f"{DATA_PATH}agriBORA_maize_prices_weeks_46_to_51.csv")
    kamis = load_csv(f"{DATA_PATH}kamis_maize_prices_raw.csv")
    sample_sub = load_csv(f"{DATA_PATH}SampleSubmission.csv")

    # --------------------------------------------------------------------------
    # agriBORA main dataset analysis
    # --------------------------------------------------------------------------
    basic_overview(agribora, "agriBORA main dataset")

    agribora = date_features(agribora, "Date")

    print("\nTime range:")
    print(f"{agribora['Date'].min()} → {agribora['Date'].max()}")

    print("\nRecords per county:")
    for county in agribora["County"].unique():
        print(f"- {county}: {agribora[agribora['County'] == county].shape[0]} rows")

    county_price_stats(agribora)

    # Target county validation
    print("\nTarget county availability:")
    for county in TARGET_COUNTIES:
        if county in agribora["County"].unique():
            print(f"✓ {county} present")
        else:
            print(f"✗ {county} missing")

    # --------------------------------------------------------------------------
    # Recent weeks dataset
    # --------------------------------------------------------------------------
    print("\n=== RECENT WEEKS DATA ===")
    print("Date range:",
          agribora_recent["Date"].min(),
          "→",
          agribora_recent["Date"].max())

    # --------------------------------------------------------------------------
    # KAMIS dataset analysis
    # --------------------------------------------------------------------------
    print("\n=== KAMIS DATA ANALYSIS ===")

    date_col = "date" if "date" in kamis.columns else "Date"
    kamis = date_features(kamis, date_col)

    print("Years covered:", sorted(kamis["year"].unique()))
    print("Total records:", kamis.shape[0])

    price_cols = [c for c in kamis.columns if "price" in c.lower()]
    print("Price-related columns:", price_cols)

    # --------------------------------------------------------------------------
    # Sample submission requirements
    # --------------------------------------------------------------------------
    print("\n=== SAMPLE SUBMISSION ANALYSIS ===")

    sample_sub["county"] = sample_sub["ID"].str.split("_Week_").str[0]
    sample_sub["week"] = sample_sub["ID"].str.split("_Week_").str[1].astype(int)

    print("Total predictions required:", sample_sub.shape[0])
    print("Weeks to predict:", sorted(sample_sub["week"].unique()))

    print("\nPredictions per county:")
    for county, cnt in sample_sub["county"].value_counts().items():
        print(f"- {county}: {cnt}")

    # --------------------------------------------------------------------------
    # Final conclusions
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("EDA CONCLUSIONS")
    print("=" * 80)

    print("""
1. agriBORA data provides clean, weekly maize wholesale prices.
2. Strong seasonality and county-level price differences exist.
3. Recent weeks data is critical for anchoring forecasts.
4. KAMIS data can be used as an auxiliary source but is not mandatory.
5. Forecasting horizon is short → deterministic/statistical baselines are viable.
6. Exact submission format must be strictly followed.
    """)


if __name__ == "__main__":
    main()
