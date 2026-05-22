import streamlit as st
import pandas as pd
import numpy as np
import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

import matplotlib.pyplot as plt

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
import tempfile

# =========================
# APP TITLE
# =========================
st.title("Kenya Sugar AI Dashboard v4 (Forecasting + PDF Reporting)")

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

df["stock_pressure"] = df["Sug_Closing_Stock"] / (df["Sug_Sales"] + 1)
df["import_dependency"] = df["Sugar_Imports"] / (df["Sug_Production"] + 1)

df = df.dropna()

# =========================
# TRAIN / TEST SPLIT
# =========================
train_df = df[df["Date"].dt.year <= 2024]
test_df = df[df["Date"].dt.year == 2025]

st.subheader("Train/Test Split")
st.write("Train rows:", len(train_df))
st.write("Test rows (2025):", len(test_df))

# =========================
# FEATURES / TARGETS
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
# MODEL TRAINING
# =========================
results = {}
pred_2025 = {}

st.subheader("Model Training (2025 Evaluation)")

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

    ensemble = (rf_pred + ridge_pred) / 2

    mape = np.mean(np.abs((y_test - ensemble) / y_test))
    accuracy = (1 - mape) * 100

    results[target] = {
        "MAPE": mape,
        "Accuracy (%)": accuracy
    }

    pred_2025[target] = ensemble

# =========================
# RESULTS TABLE
# =========================
st.subheader("Accuracy Summary")
st.dataframe(pd.DataFrame(results).T)

# =========================
# 2025 VISUALIZATION
# =========================
st.subheader("2025 Forecast vs Actual")

selected = st.selectbox("Select Variable", targets)

fig, ax = plt.subplots()

ax.plot(test_df[selected].values, label="Actual", marker="o")
ax.plot(pred_2025[selected], label="Forecast", marker="o")

ax.legend()
ax.set_title(selected)

st.pyplot(fig)

# Save chart for PDF
chart_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
fig.savefig(chart_path)

# =========================
# 2026 FORECAST
# =========================
st.subheader("2026 Forecast Projection")

future_dates = pd.date_range("2026-01-01", periods=12, freq="MS")

future_df = pd.DataFrame()
future_df["Date"] = future_dates
future_df["Month"] = future_df["Date"].dt.month
future_df["Year"] = future_df["Date"].dt.year

last = df.iloc[-1]

for col in features:
    if col in last:
        future_df[col] = last[col]

for col in ["Sug_Sales", "Sug_Closing_Stock", "Cane_Crushed"]:
    future_df[col] = last[col] * (1 + 0.01) ** np.arange(12)

forecast_2026 = {}

for target in targets:

    X = df[features]
    y = df[target]

    rf = RandomForestRegressor(n_estimators=300, random_state=42)
    ridge = Ridge()

    rf.fit(X, y)
    ridge.fit(X, y)

    forecast_2026[target] = (rf.predict(future_df[features]) + ridge.predict(future_df[features])) / 2

forecast_df = pd.DataFrame({
    "Month": future_dates.strftime("%Y-%m"),
    "Sug_Production": forecast_2026["Sug_Production"],
    "Molasses_Production": forecast_2026["Molasses_Prod"],
    "Sugar_Imports": forecast_2026["Sugar_Imports"],
    "Retail_Price": forecast_2026["Retail_Sug1kg_Price"]
})

st.dataframe(forecast_df)

# =========================
# 2026 CHART
# =========================
fig2, ax2 = plt.subplots()

ax2.plot(forecast_df["Sug_Production"], label="Sugar Production")
ax2.plot(forecast_df["Molasses_Production"], label="Molasses Production")
ax2.plot(forecast_df["Sugar_Imports"], label="Imports")
ax2.plot(forecast_df["Retail_Price"], label="Retail Price")

ax2.legend()
ax2.set_title("2026 Forecast Trends")

st.pyplot(fig2)

# =========================
# PDF GENERATION FUNCTION
# =========================
def generate_pdf(results, forecast_df, chart_path):

    file_path = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf").name
    doc = SimpleDocTemplate(file_path)

    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("Kenya Sugar Forecasting AI Report", styles["Title"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Methodology", styles["Heading2"]))
    story.append(Paragraph(
        "Random Forest + Ridge Regression ensemble with lag features, "
        "stock pressure ratios, and import dependency indicators.",
        styles["BodyText"]
    ))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Model Accuracy", styles["Heading2"]))

    acc_table = [["Target", "MAPE", "Accuracy (%)"]]
    for k, v in results.items():
        acc_table.append([k, round(v["MAPE"], 4), round(v["Accuracy (%)"], 2)])

    table = Table(acc_table)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
    ]))

    story.append(table)
    story.append(Spacer(1, 12))

    story.append(Paragraph("2026 Forecast Summary", styles["Heading2"]))

    data_table = [["Month", "Sugar", "Molasses", "Imports", "Price"]]

    for i in range(len(forecast_df)):
        data_table.append([
            forecast_df.iloc[i]["Month"],
            round(forecast_df.iloc[i]["Sug_Production"], 2),
            round(forecast_df.iloc[i]["Molasses_Production"], 2),
            round(forecast_df.iloc[i]["Sugar_Imports"], 2),
            round(forecast_df.iloc[i]["Retail_Price"], 2),
        ])

    table2 = Table(data_table)
    table2.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.darkblue),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
    ]))

    story.append(table2)
    story.append(Spacer(1, 12))

    story.append(Paragraph("Forecast Chart", styles["Heading2"]))
    story.append(Image(chart_path, width=400, height=250))

    doc.build(story)

    return file_path

# =========================
# DOWNLOAD BUTTON
# =========================
st.subheader("📄 Download Full Report")

if st.button("Generate PDF Report"):

    pdf_file = generate_pdf(results, forecast_df, chart_path)

    with open(pdf_file, "rb") as f:
        st.download_button(
            "Download PDF",
            f,
            file_name="Kenya_Sugar_Forecast_Report.pdf",
            mime="application/pdf"
        )
