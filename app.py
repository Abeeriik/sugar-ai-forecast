import streamlit as st
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

import matplotlib.pyplot as plt

st.title("Kenya Sugar AI Dashboard v3 (Advanced Forecasting System)")

uploaded_file = st.file_uploader("Upload Full Dataset (2020–2025)", type=["csv"])

if uploaded_file is not None:

    # =========================
    # LOAD DATA
    # =========================
    df = pd.read_csv(uploaded_file)

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

    st.subheader("Raw Data Preview")
    st.dataframe(df.head())

    # =========================
    # FEATURE ENGINEERING
    # =========================
    df["Month"] = df["Date"].dt.month
    df["Year"] = df["Date"].dt.year

    for col in ["Sug_Production", "Molasses_Prod", "Sugar_Imports", "Retail_Sug1kg_Price"]:
        df[f"{col}_lag1"] = df[col].shift(1)
        df[f"{col}_lag2"] = df[col].shift(2)

    df["prod_roll3"] = df["Sug_Production"].rolling(3).mean()
    df["molasses_roll3"] = df["Molasses_Prod"].rolling(3).mean()

    df["stock_pressure"] = df["Sug_Closing_Stock"] / (df["Sug_Sales"] + 1)
    df["import_dependency"] = df["Sugar_Imports"] / (df["Sug_Production"] + 1)

    df = df.dropna()

    # =========================
    # TRAIN / TEST SPLIT (REAL 2025 TEST)
    # =========================
    train_df = df[df["Date"].dt.year <= 2024]
    test_df = df[df["Date"].dt.year == 2025]

    st.subheader("Data Split")
    st.write("Train:", len(train_df))
    st.write("Test (2025):", len(test_df))

    # =========================
    # FEATURES & TARGETS
    # =========================
    features = [
        "Sug_Sales",
        "Sug_Closing_Stock",
        "Cane_Crushed",
        "Sugar_Made",
        "Molasses_Sales",
        "Molasses_Closing_Stocks",
        "Molasses_Price",
        "ExFactory_Sug50kg_Price",
        "WS_Sug50kg_Price",
        "Month",
        "Year",
        "stock_pressure",
        "import_dependency"
    ]

    targets = [
        "Sug_Production",
        "Molasses_Prod",
        "Sugar_Imports",
        "Retail_Sug1kg_Price"
    ]

    # =========================
    # STORE RESULTS
    # =========================
    results = {}
    forecast_results = {}

    st.subheader("📊 Model Training & Evaluation")

    # =========================
    # MODEL LOOP
    # =========================
    for target in targets:

        X_train = train_df[features]
        y_train = train_df[target]

        X_test = test_df[features]
        y_test = test_df[target]

        # MODELS
        rf = RandomForestRegressor(n_estimators=300, random_state=42)
        ridge = Ridge()

        rf.fit(X_train, y_train)
        ridge.fit(X_train, y_train)

        rf_pred = rf.predict(X_test)
        ridge_pred = ridge.predict(X_test)

        # ENSEMBLE (AVERAGE)
        ensemble_pred = (rf_pred + ridge_pred) / 2

        # CONFIDENCE BAND (simple uncertainty estimate)
        std_dev = np.std([rf_pred, ridge_pred], axis=0)

        lower = ensemble_pred - 1.5 * std_dev
        upper = ensemble_pred + 1.5 * std_dev

        # ACCURACY (ensemble)
        mape = np.mean(np.abs((y_test - ensemble_pred) / y_test))
        accuracy = (1 - mape) * 100

        results[target] = {
            "MAPE": mape,
            "Accuracy": accuracy
        }

        forecast_results[target] = {
            "actual": y_test.values,
            "predicted": ensemble_pred,
            "lower": lower,
            "upper": upper
        }

    # =========================
    # RESULTS TABLE
    # =========================
    st.subheader("📈 Model Performance Summary")
    st.dataframe(pd.DataFrame(results).T)

    # =========================
    # VISUAL DASHBOARD
    # =========================
    st.subheader("📊 Forecast vs Actual Dashboard (2025)")

    selected_target = st.selectbox("Select Variable", targets)

    data = forecast_results[selected_target]

    fig, ax = plt.subplots()

    ax.plot(data["actual"], label="Actual", marker="o")
    ax.plot(data["predicted"], label="Forecast (Ensemble)", marker="o")
    ax.fill_between(
        range(len(data["actual"])),
        data["lower"],
        data["upper"],
        alpha=0.2,
        label="Confidence Range"
    )

    ax.set_title(f"{selected_target} Forecast vs Actual (2025)")
    ax.legend()

    st.pyplot(fig)