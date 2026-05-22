import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

# =========================================================
# APP
# =========================================================

st.title("Kenya Sugar AI Dashboard v4 - Forecasting System")

uploaded_file = st.file_uploader("Upload Dataset (2020–2025)", type=["csv"])

if uploaded_file is not None:

    # =====================================================
    # LOAD DATA
    # =====================================================

    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()

    df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
    df.rename(columns={df.columns[0]: "Date"}, inplace=True)

    df = df.sort_values("Date").reset_index(drop=True)

    df["Month"] = df["Date"].dt.month
    df["Year"] = df["Date"].dt.year

    # =====================================================
    # FEATURES ENGINEERING
    # =====================================================

    df["stock_pressure"] = df["Sug_Closing_Stock"] / (df["Sug_Sales"] + 1)
    df["import_dependency"] = df["Sugar_Imports"] / (df["Sug_Production"] + 1)

    df["sin_month"] = np.sin(2 * np.pi * df["Month"] / 12)
    df["cos_month"] = np.cos(2 * np.pi * df["Month"] / 12)

    df = df.dropna().reset_index(drop=True)

    # =====================================================
    # SPLIT
    # =====================================================

    train_df = df[df["Year"] <= 2024]
    test_df = df[df["Year"] == 2025]

    # =====================================================
    # FEATURES & TARGETS
    # =====================================================

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
        "import_dependency",
        "sin_month",
        "cos_month"
    ]

    targets = [
        "Sug_Production",
        "Molasses_Prod",
        "Sugar_Imports",
        "Retail_Sug1kg_Price"
    ]

    # =====================================================
    # STORAGE
    # =====================================================

    results = {}
    forecast_results = {}

    st.subheader("Model Training & Performance (2025)")

    # =====================================================
    # TRAIN MODELS
    # =====================================================

    for target in targets:

        X_train = train_df[features]
        y_train = train_df[target]

        X_test = test_df[features]
        y_test = test_df[target]

        rf = RandomForestRegressor(n_estimators=300, random_state=42)

        xgb = XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )

        ridge = Ridge()

        rf.fit(X_train, y_train)
        xgb.fit(X_train, y_train)
        ridge.fit(X_train, y_train)

        rf_pred = rf.predict(X_test)
        xgb_pred = xgb.predict(X_test)
        ridge_pred = ridge.predict(X_test)

        ensemble_pred = (0.4 * rf_pred) + (0.4 * xgb_pred) + (0.2 * ridge_pred)

        std_dev = np.std([rf_pred, xgb_pred, ridge_pred], axis=0)

        lower = ensemble_pred - 1.5 * std_dev
        upper = ensemble_pred + 1.5 * std_dev

        mape = np.mean(np.abs((y_test - ensemble_pred) / y_test))
        accuracy = (1 - mape) * 100

        results[target] = {
            "MAPE": mape,
            "Accuracy (%)": accuracy
        }

        forecast_results[target] = {
            "actual": y_test.values,
            "predicted": ensemble_pred,
            "lower": lower,
            "upper": upper
        }

    # =====================================================
    # RESULTS TABLE
    # =====================================================

    st.subheader("Model Performance Table")
    st.dataframe(pd.DataFrame(results).T)

    # =====================================================
    # 2025 PLOT
    # =====================================================

    st.subheader("2025 Actual vs Forecast")

    selected = st.selectbox("Select Variable", targets)

    data = forecast_results[selected]

    fig, ax = plt.subplots()

    ax.plot(data["actual"], label="Actual", marker="o")
    ax.plot(data["predicted"], label="Forecast", marker="o")

    ax.fill_between(
        range(len(data["actual"])),
        data["lower"],
        data["upper"],
        alpha=0.2,
        label="Confidence Interval"
    )

    ax.legend()
    ax.set_title(selected)

    st.pyplot(fig)

    # =====================================================
    # 2026 FORECAST
    # =====================================================

    st.subheader("2026 Forecast Projections")

    future_dates = pd.date_range(start="2026-01-01", periods=12, freq="MS")

    future_df = pd.DataFrame({"Date": future_dates})

    future_df["Month"] = future_df["Date"].dt.month
    future_df["Year"] = future_df["Date"].dt.year

    for col in features:
        if col not in ["Month", "Year", "sin_month", "cos_month"]:
            future_df[col] = df[col].iloc[-1]

    future_df["sin_month"] = np.sin(2 * np.pi * future_df["Month"] / 12)
    future_df["cos_month"] = np.cos(2 * np.pi * future_df["Month"] / 12)

    forecast_2026 = {}

    for target in targets:

        rf.fit(df[features], df[target])
        xgb.fit(df[features], df[target])

        rf_pred = rf.predict(future_df[features])
        xgb_pred = xgb.predict(future_df[features])

        forecast_2026[target] = (0.5 * rf_pred + 0.5 * xgb_pred)

    forecast_table = pd.DataFrame(forecast_2026)
    forecast_table.insert(0, "Date", future_dates)

    st.subheader("2026 Forecast Table")
    st.dataframe(forecast_table)

    # =====================================================
    # 2026 PLOT
    # =====================================================

    fig2, ax2 = plt.subplots()

    for col in targets:
        ax2.plot(forecast_table["Date"], forecast_table[col], label=col)

    ax2.legend()
    ax2.set_title("2026 Forecast")

    plt.xticks(rotation=45)

    st.pyplot(fig2)
