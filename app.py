import streamlit as st
import pandas as pd
import numpy as np
import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

import matplotlib.pyplot as plt

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image
)

from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

import tempfile

# =========================
# APP TITLE
# =========================
st.title("Kenya Sugar AI Dashboard v5")

# =========================
# LOAD DATA
# =========================
uploaded_file = st.file_uploader("Upload Dataset", type=["csv"])

DATA_PATH = "Integrated_Sugar_Forecasting_Data.csv"

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

elif os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)

else:
    st.error("Dataset missing")
    st.stop()

# =========================
# PREPROCESS
# =========================
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date")

df["Month"] = df["Date"].dt.month
df["Year"] = df["Date"].dt.year

# =========================
# FEATURE ENGINEERING
# =========================
for col in [
    "Sug_Production",
    "Molasses_Prod",
    "Sugar_Imports",
    "Retail_Sug1kg_Price"
]:
    df[f"{col}_lag1"] = df[col].shift(1)
    df[f"{col}_lag2"] = df[col].shift(2)

df["stock_pressure"] = (
    df["Sug_Closing_Stock"] /
    (df["Sug_Sales"] + 1)
)

df["import_dependency"] = (
    df["Sugar_Imports"] /
    (df["Sug_Production"] + 1)
)

df = df.dropna()

# =========================
# FEATURES
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
# TRAIN / TEST
# =========================
train_df = df[df["Year"] <= 2024]
test_df = df[df["Year"] == 2025]

results = {}
predictions_2025 = {}

# =========================
# MODEL TRAINING
# =========================
models = {}

for target in targets:

    X_train = train_df[features]
    y_train = train_df[target]

    X_test = test_df[features]
    y_test = test_df[target]

    rf = RandomForestRegressor(
        n_estimators=300,
        random_state=42
    )

    ridge = Ridge()

    rf.fit(X_train, y_train)
    ridge.fit(X_train, y_train)

    rf_pred = rf.predict(X_test)
    ridge_pred = ridge.predict(X_test)

    ensemble = (rf_pred + ridge_pred) / 2

    mape = np.mean(
        np.abs((y_test - ensemble) / y_test)
    )

    accuracy = (1 - mape) * 100

    results[target] = {
        "MAPE": round(mape, 4),
        "Accuracy (%)": round(accuracy, 2)
    }

    predictions_2025[target] = ensemble

    models[target] = {
        "rf": rf,
        "ridge": ridge
    }

# =========================
# REPORT SECTION
# =========================
st.subheader("Model Accuracy")
st.dataframe(pd.DataFrame(results).T)

# =========================
# 2025 VISUAL
# =========================
selected = st.selectbox(
    "Select Variable",
    targets
)

fig, ax = plt.subplots()

ax.plot(
    test_df[selected].values,
    label="Actual",
    marker="o"
)

ax.plot(
    predictions_2025[selected],
    label="Forecast",
    marker="o"
)

ax.legend()
ax.set_title(selected)

st.pyplot(fig)

# =========================
# SAVE CHART
# =========================
chart_path = tempfile.NamedTemporaryFile(
    delete=False,
    suffix=".png"
).name

fig.savefig(chart_path)

# =========================
# 🔮 TRUE 2026 FORECAST
# =========================
st.subheader("2026 Forecast")

future_dates = pd.date_range(
    start="2026-01-01",
    periods=12,
    freq="MS"
)

future_rows = []

last_row = df.iloc[-1].copy()

for i, date in enumerate(future_dates):

    row = last_row.copy()

    row["Month"] = date.month
    row["Year"] = 2026

    # SEASONALITY
    seasonal_factor = (
        1 + 0.08 * np.sin(2 * np.pi * i / 12)
    )

    # DYNAMIC CHANGES
    row["Sug_Sales"] *= seasonal_factor
    row["Cane_Crushed"] *= seasonal_factor
    row["Sugar_Made"] *= seasonal_factor

    # UPDATE RATIOS
    row["stock_pressure"] = (
        row["Sug_Closing_Stock"] /
        (row["Sug_Sales"] + 1)
    )

    row["import_dependency"] = (
        row["Sugar_Imports"] /
        (row["Sug_Production"] + 1)
    )

    # PREDICT TARGETS
    for target in targets:

        rf_model = models[target]["rf"]
        ridge_model = models[target]["ridge"]

        rf_pred = rf_model.predict(
            pd.DataFrame([row[features]])
        )[0]

        ridge_pred = ridge_model.predict(
            pd.DataFrame([row[features]])
        )[0]

        pred = (rf_pred + ridge_pred) / 2

        row[target] = pred

    future_rows.append(row.copy())

    # RECURSIVE UPDATE
    last_row = row.copy()

forecast_df = pd.DataFrame(future_rows)

# =========================
# FORECAST TABLE
# =========================
display_df = pd.DataFrame({
    "Month": future_dates.strftime("%Y-%m"),
    "Sugar_Production":
        forecast_df["Sug_Production"].round(2),

    "Molasses_Production":
        forecast_df["Molasses_Prod"].round(2),

    "Sugar_Imports":
        forecast_df["Sugar_Imports"].round(2),

    "Retail_Price":
        forecast_df["Retail_Sug1kg_Price"].round(2)
})

st.dataframe(display_df)

# =========================
# FORECAST CHART
# =========================
fig2, ax2 = plt.subplots()

ax2.plot(
    display_df["Sugar_Production"],
    label="Sugar Production"
)

ax2.plot(
    display_df["Molasses_Production"],
    label="Molasses Production"
)

ax2.plot(
    display_df["Sugar_Imports"],
    label="Sugar Imports"
)

ax2.plot(
    display_df["Retail_Price"],
    label="Retail Price"
)

ax2.legend()
ax2.set_title("2026 Forecast Trends")

st.pyplot(fig2)

# =========================
# PDF FUNCTION
# =========================
def generate_pdf():

    pdf_path = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ).name

    doc = SimpleDocTemplate(pdf_path)

    styles = getSampleStyleSheet()

    story = []

    story.append(
        Paragraph(
            "Kenya Sugar AI Forecast Report",
            styles["Title"]
        )
    )

    story.append(Spacer(1, 12))

    story.append(
        Paragraph(
            "Methodology",
            styles["Heading2"]
        )
    )

    story.append(
        Paragraph(
            """
            Random Forest + Ridge Regression ensemble
            with recursive monthly forecasting,
            seasonal adjustment,
            lag relationships,
            stock pressure ratios,
            and import dependency indicators.
            """,
            styles["BodyText"]
        )
    )

    story.append(Spacer(1, 12))

    # ACCURACY TABLE
    acc_table = [["Target", "MAPE", "Accuracy"]]

    for k, v in results.items():
        acc_table.append([
            k,
            v["MAPE"],
            v["Accuracy (%)"]
        ])

    t1 = Table(acc_table)

    t1.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.grey),
        ("TEXTCOLOR", (0,0), (-1,0), colors.whitesmoke),
        ("GRID", (0,0), (-1,-1), 0.5, colors.black),
    ]))

    story.append(t1)

    story.append(Spacer(1, 12))

    # FORECAST TABLE
    forecast_table = [display_df.columns.tolist()]

    for row in display_df.values.tolist():
        forecast_table.append(row)

    t2 = Table(forecast_table)

    t2.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.darkblue),
        ("TEXTCOLOR", (0,0), (-1,0), colors.whitesmoke),
        ("GRID", (0,0), (-1,-1), 0.5, colors.black),
    ]))

    story.append(t2)

    story.append(Spacer(1, 12))

    story.append(Image(chart_path, width=400, height=250))

    doc.build(story)

    return pdf_path

# =========================
# DOWNLOAD PDF
# =========================
if st.button("Generate PDF Report"):

    pdf = generate_pdf()

    with open(pdf, "rb") as f:

        st.download_button(
            "Download PDF",
            f,
            file_name="Kenya_Sugar_AI_Report.pdf",
            mime="application/pdf"
        )
