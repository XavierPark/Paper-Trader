import streamlit as st
import yfinance as yf
import pandas as pd
import datetime
import numpy as np

st.set_page_config(page_title="📈 Paper Trader", layout="wide")

st.title("📊 Paper Trader Dashboard")
symbol = st.text_input("Enter Stock Symbol", "AAPL")

period = st.selectbox("Select Period", ["1d", "5d", "1mo", "3mo", "6mo", "1y"], index=2)
interval = st.selectbox("Select Interval", ["1m", "5m", "15m", "1h", "1d"], index=2)

if st.button("Fetch Data"):
    with st.spinner("Downloading data..."):
        data = yf.download(symbol, period=period, interval=interval)
        if not data.empty:
            st.success("Data fetched successfully!")
            st.line_chart(data["Close"])
            st.dataframe(data.tail(10))
        else:
            st.error("No data found for this symbol.")

# Placeholder for future AI prediction logic
st.subheader("🔮 AI Movement Prediction (Coming Soon)")
st.info("AI analysis of large movements will be added here.")

def detect_large_moves(df):
    df["Range"] = df["High"] - df["Low"]
    avg_range = df["Range"].mean()
    large_moves = df[df["Range"] > 3 * avg_range]
    return large_moves

def detect_volume_spikes(df):
    avg_volume = df["Volume"].mean()
    spikes = df[df["Volume"] > 2.5 * avg_volume]
    return spikes

if not data.empty:
    st.subheader("🔎 Detected Large Movements")
    large_moves = detect_large_moves(data)
    if not large_moves.empty:
        st.warning(f"Found {len(large_moves)} large movement candles:")
        st.dataframe(large_moves.tail(5))
    else:
        st.info("No unusually large price movements found.")

    st.subheader("📈 Volume Spike Alerts")
    spikes = detect_volume_spikes(data)
    if not spikes.empty:
        st.warning(f"Detected {len(spikes)} volume spikes:")
        st.dataframe(spikes.tail(5))
    else:
        st.info("No significant volume spikes found.")

