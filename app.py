import streamlit as st
import yfinance as yf
import pandas as pd
import datetime

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
