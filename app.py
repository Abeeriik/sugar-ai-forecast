
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor, Ridge
from xgboost import XGBRegressor

# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv("Integrated_Sugar_Forecasting_Data.csv")
df = df.sort_values(["Year", "Month"]).reset_index(drop=True)

# =========================================================
# FEATURE ENGINEERING
# =========================================================

df["stock_pressure"] = df["Sug_Closing_Stock"] / (df["Cane_Crushed"] + 1)
df["import_dependency"] = df["Sugar_Imports"] / (df["Sug_Sales"] + 1)

# seasonality
df["sin_month"] = np.sin(2 * np.pi * df["Month"] / 12)
df["cos_month"] = np.cos(2 * np.pi * df["Month"] / 12)

# =========================================================
# FEATURES
# =========================================================

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
    "stock_pressure",
    "import_dependency",
    "sin_month",
    "cos_month",
    "Month",
    "Year"
]

# =========================================================
# TARGETS
# =========================================================

targets = {
    "Sug_Production": "Sugar_Made",
    "Molasses_Prod": "Molasses_Sales",
    "Sugar_Imports": "Sugar_Imports",
    "Retail_Sug1kg_Price": "WS_Sug50kg_Price"
}

# =========================================================
# MODEL STORAGE
# =========================================================

models = {}
results = {}

# =========================================================
# TRAIN MODELS
# =========================================================

for name, target in targets.items():

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # ----------------------------
    # MODEL SELECTION LOGIC
    # ----------------------------

    if name in ["Sugar_Imports", "Retail_Sug1kg_Price"]:
        model = XGBRegressor(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
    else:
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=6,
            random_state=42
        )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    mape = mean_absolute_percentage_error(y_test, preds)
    accuracy = (1 - mape) * 100

    models[name] = model
    results[name] = [mape, accuracy]

# =========================================================
# PERFORMANCE TABLE
# =========================================================

performance_df = pd.DataFrame(results).T
performance_df.columns = ["MAPE", "Accuracy (%)"]

print("\nMODEL PERFORMANCE")
print(performance_df)

# =========================================================
# 2026 FORECASTING
# =========================================================

future = pd.DataFrame({
    "Year": [2026] * 12,
    "Month": range(1, 13)
})

# carry forward latest values
for col in features:
    if col not in ["Month", "Year", "sin_month", "cos_month"]:
        future[col] = df[col].iloc[-1]

# seasonality
future["sin_month"] = np.sin(2 * np.pi * future["Month"] / 12)
future["cos_month"] = np.cos(2 * np.pi * future["Month"] / 12)

# =========================================================
# PREDICTIONS
# =========================================================

for name, model in models.items():
    future[f"Forecast_{name}"] = model.predict(future[features])

# =========================================================
# OUTPUT TABLE
# =========================================================

forecast_table = future[
    ["Year", "Month"] +
    [f"Forecast_{k}" for k in targets.keys()]
]

print("\n2026 FORECAST")
print(forecast_table)
