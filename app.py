# Stock App - Rebuild Based on Design Doc

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Initialize session state
if 'trades' not in st.session_state:
    st.session_state.trades = []
if 'mode' not in st.session_state:
    st.session_state.mode = 'Live'

# Load data
@st.cache_data

def load_data(symbol, mode='Live'):
    if mode == 'Live':
        return yf.download(symbol, period="1mo", interval="1d")
    elif mode == 'Replay':
        return pd.read_csv("historical.csv")  # Placeholder path
    else:
        return pd.read_csv("static_snapshot.csv")

# Feature engineering
def engineer_features(df):
    df['MA_5'] = df['Close'].rolling(5).mean()
    df['MA_10'] = df['Close'].rolling(10).mean()
    df['Volatility'] = df['Close'].rolling(5).std()
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    return df.dropna()

# Train model and generate predictions
def ai_predict(df):
    features = ['MA_5', 'MA_10', 'Volatility']
    X = df[features]
    y = df['Target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    df['Prediction'] = model.predict(X)
    df['Confidence'] = model.predict_proba(X)[:, 1]
    accuracy = accuracy_score(y_test, model.predict(X_test))
    return df, model, accuracy

# Log trade
def log_trade(row, action):
    st.session_state.trades.append({
        'Time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'Ticker': row.name,
        'Action': action,
        'Price': row['Close'],
        'Confidence': row['Confidence'],
        'Reason': f"AI Prediction = {int(row['Prediction'].item() if hasattr(row['Prediction'], 'item') else row['Prediction'])}, MA_5 = {float(row['MA_5']) if hasattr(row['MA_5'], 'item') else row['MA_5']:.2f}"
    })

# UI
st.title("📈 Stock App - AI Paper Trader")
st.sidebar.header("Configuration")

ticker = st.sidebar.text_input("Symbol", "AAPL")
mode = st.sidebar.radio("Data Mode", ['Live', 'Replay', 'Static'])
st.session_state.mode = mode

if st.sidebar.button("Load Data"):
    df = load_data(ticker, mode)
    df = engineer_features(df)
    df, model, acc = ai_predict(df)

    st.metric("Model Accuracy", f"{acc * 100:.2f}%")
    st.subheader("📊 Predictions")
    st.dataframe(df[['Close', 'MA_5', 'MA_10', 'Prediction', 'Confidence']].tail())

    last = df.iloc[-1]
    last = last.copy()  # Ensure it's a Series, not a view of a DataFrame
    action = "BUY" if int(last['Prediction']) == 1 else "SELL"
    log_trade(last, action)
    st.success(f"AI suggests to {action} at ${last['Close']:.2f} (Confidence: {last['Confidence']:.2f})")

    st.subheader("🧾 Trade Log")
trade_df = pd.DataFrame(st.session_state.trades)
if not trade_df.empty:
    trade_df['Action Color'] = trade_df['Action'].apply(lambda x: '🟢 BUY' if x == 'BUY' else '🔴 SELL')
    styled_df = trade_df.style.hide(axis='columns', subset=['Action Color']).apply(
        lambda x: ["background-color: #d4fcd4" if v == 'BUY' else "background-color: #fcd4d4" for v in trade_df['Action']],
        axis=0
    )
    st.dataframe(styled_df, use_container_width=True)
    csv = trade_df.drop(columns=['Action Color']).to_csv(index=False)
    st.download_button("📥 Download Trade Log as CSV", data=csv, file_name="trade_log.csv", mime="text/csv")
else:
    st.info("No trades logged yet.")
