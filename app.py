import streamlit as st
import pandas as pd
import numpy as np
import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

import matplotlib.pyplot as plt

# =========================
# APP TITLE
# =========================
st.title("Kenya Sugar AI Dashboard v4 (2025 Evaluation + 2026 Forecasting)")

# =========================
# DATA LOADING
# =========================
uploaded_file = st.file_uploader("Upload Dataset (CSV)", type=["csv"])

DATA_PATH = "Integrated_Sugar_Forecasting_Data.csv"

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
    else:
        st.error("Dataset not found. Upload CSV or add file to repo.")
        st.stop()

# =========================
# PREPROCESSING
# =========================
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
# TRAIN / TEST SPLIT
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
# MODEL TRAINING + EVALUATION
# =========================
results = {}
forecast_2025 = {}

st.subheader("📊 Model Training & 2025 Evaluation")

for target in targets:

    X_train = train_df[features]
    y_train = train_df[target]

    X_test = test_df[features]
    y_test = test_df[target]

    rf = RandomForestRegressor(n_estimators=300, random_state=42)
    ridge = Ridge()

    rf.fit(X_train, y_train)
    ridge.fit(X_train, y_train)

    rf_pred = rf.predict(X_test)
    ridge_pred = ridge.predict(X_test)

    ensemble_pred = (rf_pred + ridge_pred) / 2

    mape = np.mean(np.abs((y_test - ensemble_pred) / y_test))
    accuracy = (1 - mape) * 100

    results[target] = {
        "MAPE": mape,
        "Accuracy (%)": accuracy
    }

    forecast_2025[target] = {
        "actual": y_test.values,
        "predicted": ensemble_pred
    }

# =========================
# MODEL REPORT
# =========================
st.subheader("📄 Model Report")

st.markdown("""
### Models Used
- Random Forest Regressor (300 trees)
- Ridge Regression
- Ensemble Averaging

### Feature Engineering
- Lag features (1–2 months)
- Rolling averages (3-month)
- Stock pressure ratio
- Import dependency ratio

### Evaluation
- MAPE used for error
- Accuracy = 1 - MAPE
""")

st.subheader("📊 Accuracy Summary")
st.dataframe(pd.DataFrame(results).T)

# =========================
# VISUAL (2025)
# =========================
st.subheader("📉 2025 Forecast vs Actual")

selected = st.selectbox("Select Variable (2025)", targets)

fig, ax = plt.subplots()

ax.plot(forecast_2025[selected]["actual"], label="Actual", marker="o")
ax.plot(forecast_2025[selected]["predicted"], label="Forecast", marker="o")

ax.set_title(selected)
ax.legend()

st.pyplot(fig)

# =========================
# =========================
# 🔮 2026 FORECAST ENGINE
# =========================
# =========================

st.subheader("🔮 2026 Forecast Projection")

future_dates = pd.date_range(start="2026-01-01", periods=12, freq="MS")

future_df = pd.DataFrame()
future_df["Date"] = future_dates
future_df["Month"] = future_df["Date"].dt.month
future_df["Year"] = future_df["Date"].dt.year

last = df.iloc[-1]

for col in features:
    if col in last:
        future_df[col] = last[col]

# mild trend growth
for col in ["Sug_Sales", "Sug_Closing_Stock", "Cane_Crushed"]:
    future_df[col] = last[col] * (1 + 0.01) ** np.arange(12)

future_results = {}

for target in targets:

    X = df[features]
    y = df[target]

    rf = RandomForestRegressor(n_estimators=300, random_state=42)
    ridge = Ridge()

    rf.fit(X, y)
    ridge.fit(X, y)

    rf_pred = rf.predict(future_df[features])
    ridge_pred = ridge.predict(future_df[features])

    final_pred = (rf_pred + ridge_pred) / 2

    future_results[target] = final_pred

# =========================
# 2026 TABLE
# =========================
forecast_2026 = pd.DataFrame({
    "Month": future_dates.strftime("%Y-%m"),
    "Sug_Production": future_results["Sug_Production"],
    "Molasses_Production": future_results["Molasses_Prod"],
    "Sugar_Imports": future_results["Sugar_Imports"],
    "Retail_Price": future_results["Retail_Sug1kg_Price"]
})

st.subheader("📅 2026 Monthly Forecast Table")
st.dataframe(forecast_2026)

# =========================
# 2026 CHART
# =========================
st.subheader("📊 2026 Forecast Trends")

fig, ax = plt.subplots()

ax.plot(forecast_2026["Sug_Production"], label="Sugar Production")
ax.plot(forecast_2026["Molasses_Production"], label="Molasses Production")
ax.plot(forecast_2026["Sugar_Imports"], label="Imports")
ax.plot(forecast_2026["Retail_Price"], label="Retail Price")

ax.legend()
ax.set_title("2026 Forecast Trends")

st.pyplot(fig)
