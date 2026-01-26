"""
Streamlit App: Deterministic Maize Price Forecasting
Author: Qamar Usman
Approach: Rule-based / Statistical (No Machine Learning)

Run:
    streamlit run forecasting.py
"""

import streamlit as st
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# ------------------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------------------
# CORRECTED PATHS - Files are in the root directory, not ./data/
TARGET_COUNTIES = [
    "Kiambu",
    "Kirinyaga",
    "Mombasa",
    "Nairobi",
    "Uasin-Gishu"
]

SEASONAL_ADJUSTMENTS = {
    50: 1.10,
    51: 1.15,
    52: 1.20,
    1: 1.15,
    2: 1.10
}

COUNTY_FACTORS = {
    "Kiambu": 1.00,
    "Kirinyaga": 0.98,
    "Mombasa": 0.95,
    "Nairobi": 1.00,
    "Uasin-Gishu": 1.05
}

# ------------------------------------------------------------------------------
# DATA LOADING
# ------------------------------------------------------------------------------
@st.cache_data
def load_data():
    # CORRECTED: Removed DATA_PATH and using direct file names
    try:
        # Try to load from the root directory
        agribora = pd.read_csv("agriBORA_maize_prices.csv")
        agribora_recent = pd.read_csv("agriBORA_maize_prices_weeks_46_to_51.csv")
    except FileNotFoundError:
        # If files are in a different location, you might need to adjust
        st.error("Error: CSV files not found. Please ensure the following files are in the same directory:")
        st.error("1. agriBORA_maize_prices.csv")
        st.error("2. agriBORA_maize_prices_weeks_46_to_51.csv")
        st.stop()

    agribora["Date"] = pd.to_datetime(agribora["Date"])
    agribora_recent["Date"] = pd.to_datetime(agribora_recent["Date"])

    agribora = agribora[agribora["County"].isin(TARGET_COUNTIES)]
    agribora_recent = agribora_recent[
        agribora_recent["County"].isin(TARGET_COUNTIES)
    ]

    return agribora, agribora_recent


# ------------------------------------------------------------------------------
# BASE PRICE EXTRACTION
# ------------------------------------------------------------------------------
def get_recent_prices(agribora_recent):
    return (
        agribora_recent
        .sort_values("Date")
        .groupby("County")
        .tail(1)
        .set_index("County")["WholeSale"]
        .to_dict()
    )


# ------------------------------------------------------------------------------
# FORECAST FUNCTION (CORE LOGIC)
# ------------------------------------------------------------------------------
def forecast_prices(
    agribora,
    recent_prices,
    week_number
):
    predictions = []

    for county in TARGET_COUNTIES:
        base_price = recent_prices[county]

        hist = agribora[agribora["County"] == county]
        min_price = hist["WholeSale"].min() * 0.9
        max_price = hist["WholeSale"].max() * 1.1

        seasonal_factor = SEASONAL_ADJUSTMENTS.get(week_number, 1.0)
        county_factor = COUNTY_FACTORS[county]

        price = base_price * seasonal_factor * county_factor
        price = max(min_price, min(max_price, price))
        price = round(price, 2)

        predictions.append({
            "County": county,
            "Predicted_Wholesale_Price_KES": price
        })

    return pd.DataFrame(predictions)


# ------------------------------------------------------------------------------
# STREAMLIT UI
# ------------------------------------------------------------------------------
def main():

    st.set_page_config(
        page_title="Maize Price Forecasting",
        layout="centered"
    )

    st.title("🌽 Maize Price Forecasting Dashboard")
    st.markdown(
        """
        **Approach:** Deterministic, rule-based forecasting  
        **No Machine Learning – powered by statistics & domain knowledge**
        """
    )

    # Load data
    with st.spinner("Loading data..."):
        agribora, agribora_recent = load_data()
    
    # Show data info
    with st.expander("📁 Data Information"):
        st.write(f"Historical data shape: {agribora.shape}")
        st.write(f"Recent data shape: {agribora_recent.shape}")
        st.write("Available counties in recent data:", agribora_recent["County"].unique())
    
    recent_prices = get_recent_prices(agribora_recent)
    
    # Show current prices
    with st.expander("📈 Current Prices (Most Recent)"):
        current_df = pd.DataFrame(list(recent_prices.items()), 
                                 columns=["County", "Current_Price"])
        st.dataframe(current_df)

    # Sidebar
    st.sidebar.header("Forecast Settings")
    week_number = st.sidebar.selectbox(
        "Select Week Number",
        options=sorted(SEASONAL_ADJUSTMENTS.keys())
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Target Counties**")
    for c in TARGET_COUNTIES:
        st.sidebar.write(f"- {c}")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Seasonal Adjustments**")
    for week, factor in SEASONAL_ADJUSTMENTS.items():
        st.sidebar.write(f"Week {week}: {factor}x")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**County Factors**")
    for county, factor in COUNTY_FACTORS.items():
        st.sidebar.write(f"{county}: {factor}x")

    # Forecast button
    if st.button("🔮 Generate Forecast", type="primary"):
        with st.spinner(f"Generating forecast for Week {week_number}..."):
            forecast_df = forecast_prices(
                agribora,
                recent_prices,
                week_number
            )

        st.subheader(f"📊 Predicted Wholesale Prices (Week {week_number})")
        st.dataframe(forecast_df, use_container_width=True)
        
        # Add visualization
        col1, col2 = st.columns(2)
        with col1:
            st.bar_chart(forecast_df.set_index("County")["Predicted_Wholesale_Price_KES"])
        
        with col2:
            # Show percentage change
            forecast_df["Current_Price"] = forecast_df["County"].map(recent_prices)
            forecast_df["Change_%"] = ((forecast_df["Predicted_Wholesale_Price_KES"] - 
                                      forecast_df["Current_Price"]) / 
                                     forecast_df["Current_Price"] * 100).round(2)
            st.dataframe(forecast_df[["County", "Change_%"]], use_container_width=True)

        st.download_button(
            label="⬇️ Download Predictions (CSV)",
            data=forecast_df.to_csv(index=False),
            file_name=f"maize_price_forecast_week_{week_number}.csv",
            mime="text/csv",
            type="secondary"
        )

    # Footer
    st.markdown("---")
    st.caption(
        "Built by Qamar Usman | Deterministic Forecasting | agriBORA Challenge"
    )


if __name__ == "__main__":
    main()
