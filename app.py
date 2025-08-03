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

data = pd.DataFrame()

# 🕯️ Pattern detection logic
def detect_candle_patterns(df):
    df = df.copy()
    df['CandlePattern'] = None
    df['Doji'] = False
    df['BullishEngulfing'] = False
    df['BearishEngulfing'] = False
    df['Hammer'] = False
    df['ShootingStar'] = False

    for i in range(1, len(df)):
        prev = df.iloc[i - 1]
        curr = df.iloc[i]

        o1, h1, l1, c1 = prev['Open'], prev['High'], prev['Low'], prev['Close']
        o2, h2, l2, c2 = curr['Open'], curr['High'], curr['Low'], curr['Close']
        body1 = abs(c1 - o1)
        body2 = abs(c2 - o2)
        range2 = h2 - l2
        upper_wick = h2 - max(o2, c2)
        lower_wick = min(o2, c2) - l2

        pattern = None

        if body2 < range2 * 0.1:
            pattern = 'Doji'
            df.at[i, 'Doji'] = True
        elif (c1 < o1) and (c2 > o2) and (o2 < c1) and (c2 > o1):
            pattern = 'BullishEngulfing'
            df.at[i, 'BullishEngulfing'] = True
        elif (c1 > o1) and (c2 < o2) and (o2 > c1) and (c2 < o1):
            pattern = 'BearishEngulfing'
            df.at[i, 'BearishEngulfing'] = True
        elif body2 < range2 * 0.3 and lower_wick > body2 * 2:
            pattern = 'Hammer'
            df.at[i, 'Hammer'] = True
        elif body2 < range2 * 0.3 and upper_wick > body2 * 2:
            pattern = 'ShootingStar'
            df.at[i, 'ShootingStar'] = True

        if pattern:
            df.at[i, 'CandlePattern'] = pattern

    pattern_df = df[df['CandlePattern'].notnull()].copy()
    pattern_df.reset_index(inplace=True)
    return pattern_df

# 🔍 Large move/volume detection
def detect_large_moves(df):
    df["Range"] = df["High"] - df["Low"]
    avg_range = df["Range"].mean()
    large_moves = df[df["Range"] > 3 * avg_range]
    return large_moves

def detect_volume_spikes(df):
    avg_volume = df["Volume"].mean()
    spikes = df[df["Volume"] > 2.5 * avg_volume]
    return spikes

# 📥 Main fetch trigger
if st.button("Fetch Data"):
    with st.spinner("Downloading data..."):
        data = yf.download(symbol, period=period, interval=interval)

        if not data.empty:
            st.success("Data fetched successfully!")
            st.line_chart(data["Close"])
            st.dataframe(data.tail(10))

            # 🕯️ Detect candle patterns
            pattern_df = detect_candle_patterns(data)
            if not pattern_df.empty:
                st.subheader("🕯️ Candle Pattern Detection")
                st.dataframe(pattern_df[["Datetime", "Open", "High", "Low", "Close", "CandlePattern"]].tail(10))
            else:
                st.info("No notable candle patterns detected.")

            # 🔎 Detect large moves
            st.subheader("🔎 Detected Large Movements")
            large_moves = detect_large_moves(data)
            if not large_moves.empty:
                st.warning(f"Found {len(large_moves)} large movement candles:")
                st.dataframe(large_moves.tail(5))
            else:
                st.info("No unusually large price movements found.")

            # 📈 Volume spikes
            st.subheader("📈 Volume Spike Alerts")
            spikes = detect_volume_spikes(data)
            if not spikes.empty:
                st.warning(f"Detected {len(spikes)} volume spikes:")
                st.dataframe(spikes.tail(5))
            else:
                st.info("No significant volume spikes found.")
        else:
            st.error("No data found for this symbol.")

# 🧠 AI Prediction Placeholder
st.subheader("🔮 AI Movement Prediction (Coming Soon)")
st.info("AI analysis of large movements will be added here.")
