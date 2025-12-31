import streamlit as st
from streamlit_autorefresh import st_autorefresh
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ---------------- PAGE SETTINGS ----------------
st.set_page_config(
    page_title="Gold Trading Dashboard",
    layout="wide"
)

# Auto refresh every 1 minute
st_autorefresh(interval=60 * 1000, key="datarefresh")

# ---------------- TITLE ----------------
st.title("🟡 Gold (XAUUSD) Trading Dashboard")

# ---------------- SIDEBAR ----------------
st.sidebar.title("Settings")

timeframe = st.sidebar.selectbox(
    "Select Timeframe",
    ["5m", "15m", "1h"],
    index=0
)

show_rsi = st.sidebar.checkbox("Show RSI", value=True)

# ---------------- DATA FETCH ----------------
gold = yf.download(
    tickers="GC=F",
    period="5d",
    interval=timeframe,
    group_by="ticker",
    progress=False
)

if isinstance(gold.columns, pd.MultiIndex):
    gold = gold["GC=F"]

if gold.empty:
    st.error("No data received. Try another timeframe.")
    st.stop()

# ---------------- RSI FUNCTION ----------------
def calculate_rsi(data, period=14):
    delta = data.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# ---------------- INDICATORS ----------------
if show_rsi:
    gold["RSI"] = calculate_rsi(gold["Close"])

# ---------------- LATEST VALUES ----------------
latest_price = float(gold["Close"].iloc[-1])
latest_rsi = float(gold["RSI"].iloc[-1]) if show_rsi else np.nan

# ---------------- SIGNAL LOGIC ----------------
if latest_rsi < 30:
    signal = "🟢 BUY"
elif latest_rsi > 70:
    signal = "🔴 SELL"
else:
    signal = "🟡 HOLD"

# ---------------- METRICS ----------------
col1, col2, col3 = st.columns(3)

col1.metric(
    "Gold Price",
    f"{latest_price:,.4f}"
)

col2.metric(
    "RSI (14)",
    f"{latest_rsi:.4f}"
)

col3.metric(
    "Signal",
    signal
)

# ---------------- EXACT VALUES (SMALL FONT) ----------------
st.markdown(
    f"""
    <small>
    Exact Gold Price: <b>{latest_price:,.6f}</b><br>
    Exact RSI (14): <b>{latest_rsi:.6f}</b>
    </small>
    """,
    unsafe_allow_html=True
)

# ---------------- PRICE CHART ----------------
price_fig = go.Figure()

price_fig.add_trace(
    go.Scatter(
        x=gold.index,
        y=gold["Close"],
        name="Gold Price",
        line=dict(width=2),
        hovertemplate="Price: %{y:,.4f}<extra></extra>"
    )
)

price_fig.update_layout(
    title="Gold Price Chart",
    xaxis_title="Time",
    yaxis_title="Price",
    height=400,
    font=dict(size=10),
    yaxis=dict(
        tickformat=","
    )
)

st.plotly_chart(price_fig, use_container_width=True)

# ---------------- RSI CHART ----------------
if show_rsi:
    rsi_fig = go.Figure()

    rsi_fig.add_trace(
        go.Scatter(
            x=gold.index,
            y=gold["RSI"],
            name="RSI",
            line=dict(color="orange"),
            hovertemplate="RSI: %{y:.4f}<extra></extra>"
        )
    )

    rsi_fig.add_hline(y=70, line_dash="dash", line_color="red")
    rsi_fig.add_hline(y=30, line_dash="dash", line_color="green")

    rsi_fig.update_layout(
        title="RSI Indicator",
        xaxis_title="Time",
        yaxis_title="RSI",
        height=300,
        font=dict(size=10),
        yaxis=dict(
            tickformat=".4f"
        )
    )

    st.plotly_chart(rsi_fig, use_container_width=True)

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("⚠️ Educational use only. Not financial advice.")
