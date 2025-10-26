import streamlit as st
import yfinance as yf
from prophet import Prophet
import pandas as pd
import plotly.graph_objects as go
import os

st.set_page_config(page_title="Financial KPI Forecast", layout="wide")
from PIL import Image

# --- Logo path ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
logo_path = os.path.join(BASE_DIR, "assets", "kpi_logo.png")  # Make sure the logo is placed in /assets

# --- Display logo ---
if os.path.exists(logo_path):
    logo = Image.open(logo_path)
    st.image(logo, width=120)  # Adjust width for visual balance

# --- Updated title ---
st.title("Financial KPI Forecast & Risk Tracker")  # Clean, emoji-free title

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEMO_FILE = os.path.join(BASE_DIR, "data", "financial_stocks.csv")

# Sidebar
st.sidebar.header("📁 Data Source")
mode = st.sidebar.radio(
    "Choose Data Mode:",
    ["Demo Dataset (Built-in)", "Upload CSV", "Live Yahoo Finance"],
    index=0
)

uploaded_file = None
ticker = None
if mode == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload a financial CSV", type=["csv"])
elif mode == "Live Yahoo Finance":
    ticker = st.text_input("Enter Stock Symbol", "AAPL")

try:
    # --- Load Data ---
    if mode == "Demo Dataset (Built-in)":
        if not os.path.exists(DEMO_FILE):
            st.error(f"❌ Demo file not found at: {DEMO_FILE}")
            st.stop()
        st.success("✅ Using built-in demo dataset.")
        data = pd.read_csv(DEMO_FILE)
        if {"Date", "Ticker", "Close"}.issubset(data.columns):
            tickers = data["Ticker"].unique().tolist()
            selected_ticker = st.selectbox("Select a stock:", tickers)
            df = data[data["Ticker"] == selected_ticker][["Date", "Close"]].copy()
        else:
            st.error("Demo dataset must include Date, Ticker, and Close columns.")
            st.stop()

    elif mode == "Upload CSV" and uploaded_file:
        st.success("✅ Using uploaded CSV.")
        data = pd.read_csv(uploaded_file)
        if {"Date", "Ticker", "Close"}.issubset(data.columns):
            tickers = data["Ticker"].unique().tolist()
            selected_ticker = st.selectbox("Select a stock:", tickers)
            df = data[data["Ticker"] == selected_ticker][["Date", "Close"]].copy()
        elif {"Date", "Close"}.issubset(data.columns):
            df = data[["Date", "Close"]].copy()
        else:
            st.error("CSV must have 'Date' and 'Close' columns.")
            st.stop()

    elif mode == "Live Yahoo Finance":
        st.info("📡 Fetching live data...")
        data = yf.download(ticker, period="1y", group_by="ticker")

        if data.empty:
            st.warning("⚠️ No data found for this ticker.")
            st.stop()

        # --- Flatten MultiIndex columns if needed ---
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = ['_'.join(col).strip() for col in data.columns.values]

        # --- Find a usable 'Close' column ---
        price_col = None
        for col in data.columns:
            if "Close" in col and pd.api.types.is_numeric_dtype(data[col]):
                price_col = col
                break
        if price_col is None:
            st.error(f"❌ No numeric 'Close' column found. Columns: {list(data.columns)}")
            st.stop()

        # --- Prepare DataFrame for Prophet ---
        data = data.reset_index()
        df = pd.DataFrame({
            "ds": pd.to_datetime(data["Date"], errors="coerce"),
            "y": pd.to_numeric(data[price_col], errors="coerce")
        }).dropna(subset=["ds", "y"]).reset_index(drop=True)
        df["y"] = df["y"].astype(float)

        st.write("✅ Prophet input sample (live data):")
        st.dataframe(df.head())
        st.text(f"Data shape: {df.shape}, ds dtype: {df['ds'].dtype}, y dtype: {df['y'].dtype}")

    else:
        st.warning("⚠️ Please upload a valid CSV or select Demo mode.")
        st.stop()

    # --- Clean Data ---
    df = df.rename(columns={"Date": "ds", "Close": "y"})
    df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce").astype(float)
    df = df.dropna(subset=["ds", "y"]).reset_index(drop=True)

    # --- Forecast ---
    model = Prophet(daily_seasonality=True)
    model.fit(df)
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)

    # --- Plot ---
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["ds"], y=df["y"], mode="lines", name="Actual Price"))
    fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"], mode="lines", name="Predicted Price"))
    fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_upper"], mode="lines",
                             name="Upper Bound", line=dict(dash="dot", color="green")))
    fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_lower"], mode="lines",
                             name="Lower Bound", line=dict(dash="dot", color="red")))
    fig.update_layout(
        title="Stock Price Forecast (Next 30 Days)",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.write("📈 Forecast Data Sample:")
    st.dataframe(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(10))

except Exception as e:
    st.error(f"❌ Unexpected error: {e}")

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; font-size: 0.9em; color: gray;'>
        🚀 Built with Streamlit, Prophet, Plotly & yFinance | 📊 Designed for financial forecasting and portfolio clarity<br>
        💼 Crafted by Vikrant Thenge — Senior Data Analyst & Automation Strategist<br><br>
        <a href='https://www.linkedin.com/in/vthenge/' target='_blank'>
            <img src='https://cdn.jsdelivr.net/gh/simple-icons/simple-icons/icons/linkedin.svg' alt='LinkedIn' width='24' style='vertical-align:middle; margin-right:6px;'/>
            LinkedIn
        </a>
        &nbsp;&nbsp;&nbsp;
        <a href='https://github.com/Vikrantthenge' target='_blank'>
            <img src='https://cdn.jsdelivr.net/gh/simple-icons/simple-icons/icons/github.svg' alt='GitHub' width='24' style='vertical-align:middle; margin-right:6px;'/>
            GitHub
        </a>
        <br><br>
        🤝 Reach out for collaborations
    </div>
    """,
    unsafe_allow_html=True
)