import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import plotly.graph_objects as go
from datetime import datetime

# Session state to simulate portfolio
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = {
        'cash': 10000.0,
        'shares': 0,
        'last_price': 0.0,
        'history': [],
        'trade_log': []
    }

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

        o1, h1, l1, c1 = float(prev['Open']), float(prev['High']), float(prev['Low']), float(prev['Close'])
        o2, h2, l2, c2 = float(curr['Open']), float(curr['High']), float(curr['Low']), float(curr['Close'])

        body1 = abs(c1 - o1)
        body2 = abs(c2 - o2)
        range2 = h2 - l2
        upper_wick = h2 - max(float(o2), float(c2))
        lower_wick = min(float(o2), float(c2)) - l2

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

def detect_large_moves(df):
    df["Range"] = df["High"] - df["Low"]
    avg_range = df["Range"].mean()
    return df[df["Range"] > 3 * avg_range]

def detect_volume_spikes(df):
    avg_volume = df["Volume"].mean()
    return df[df["Volume"] > 2.5 * avg_volume]

def engineer_features(df):
    df = df.copy()
    df['Return'] = df['Close'].pct_change().shift(-1)
    df['Target'] = (df['Return'] > 0).astype(int)
    df['MA_5'] = df['Close'].rolling(5).mean()
    df['MA_10'] = df['Close'].rolling(10).mean()
    df['Volatility'] = df['Close'].rolling(5).std()
    df['Volume_Change'] = df['Volume'].pct_change()
    return df.dropna()

def train_ai_model(df):
    features = ['MA_5', 'MA_10', 'Volatility', 'Volume_Change']
    X = df[features]
    y = df['Target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    df['Prediction'] = model.predict(X)
    return df, model, accuracy_score(y_test, model.predict(X_test))

def plot_candlestick_chart(df, pred_df=None):
    fig = go.Figure(data=[
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Candlesticks'
        )
    ])

    if pred_df is not None:
        buys = pred_df[pred_df['Prediction'] == 1]
        sells = pred_df[pred_df['Prediction'] == 0]

        fig.add_trace(go.Scatter(
            x=buys.index,
            y=buys['Close'],
            mode='markers',
            marker=dict(symbol='arrow-up', color='green', size=10),
            name='AI Buy',
            hovertemplate='Buy @ %{y}<br>%{x}'
        ))

        fig.add_trace(go.Scatter(
            x=sells.index,
            y=sells['Close'],
            mode='markers',
            marker=dict(symbol='arrow-down', color='red', size=10),
            name='AI Sell',
            hovertemplate='Sell @ %{y}<br>%{x}'
        ))

    fig.update_layout(
        title='Candlestick Chart with AI Trades',
        xaxis_title='Date',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False
    )
    return fig

def run_paper_trade(signal, price):
    p = st.session_state.portfolio
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if signal == 1 and p['cash'] >= price:
        shares_to_buy = int(p['cash'] // price)
        total_cost = shares_to_buy * price
        p['shares'] += shares_to_buy
        p['cash'] -= total_cost
        p['last_price'] = price
        portfolio_value = p['cash'] + p['shares'] * price
        trade_msg = f"BUY {shares_to_buy} @ ${price:.2f}"
        p['history'].append(trade_msg)
        p['trade_log'].append({
            'Datetime': now,
            'Type': 'BUY',
            'Shares': shares_to_buy,
            'Price': price,
            'Total Value': total_cost,
            'Cash After': p['cash'],
            'Portfolio Value After': portfolio_value,
            'Signal': 'BUY'
        })

    elif signal == 0 and p['shares'] > 0:
        total_gain = p['shares'] * price
        p['cash'] += total_gain
        portfolio_value = p['cash']
        trade_msg = f"SELL {p['shares']} @ ${price:.2f}"
        p['history'].append(trade_msg)
        p['trade_log'].append({
            'Datetime': now,
            'Type': 'SELL',
            'Shares': p['shares'],
            'Price': price,
            'Total Value': total_gain,
            'Cash After': p['cash'],
            'Portfolio Value After': portfolio_value,
            'Signal': 'SELL'
        })
        p['shares'] = 0
        p['last_price'] = price

def display_portfolio():
    p = st.session_state.portfolio
    if 'trade_log' not in p:
        p['trade_log'] = []

    st.sidebar.subheader("📊 Portfolio Simulation")
    st.sidebar.write(f"Cash: ${p['cash']:.2f}")
    st.sidebar.write(f"Shares: {p['shares']}")
    st.sidebar.write(f"Last Price: ${p['last_price']:.2f}")
    portfolio_value = p['cash'] + p['shares'] * p['last_price']
    st.sidebar.write(f"Portfolio Value: ${portfolio_value:.2f}")

    if p['history']:
        st.sidebar.write("Recent Trades:")
        for entry in reversed(p['history'][-5:]):
            st.sidebar.text(entry)

    if p['trade_log']:
        st.subheader("📜 Trade Log")
        trade_df = pd.DataFrame(p['trade_log'])
        st.dataframe(trade_df)

        profit = 0
        win_trades = 0
        total_sells = 0

        for i, trade in enumerate(trade_df.itertuples()):
            if trade.Type == 'SELL':
                for j in range(i - 1, -1, -1):
                    if trade_df.iloc[j]['Type'] == 'BUY':
                        buy_price = trade_df.iloc[j]['Price']
                        sell_price = trade.Price
                        shares = trade_df.iloc[j]['Shares']
                        pnl = (sell_price - buy_price) * shares
                        profit += pnl
                        if pnl > 0:
                            win_trades += 1
                        total_sells += 1
                        break

        win_rate = (win_trades / total_sells * 100) if total_sells > 0 else 0
        unrealized = p['shares'] * p['last_price']

        st.metric("💰 Realized Profit", f"${profit:.2f}")
        st.metric("📈 Unrealized Value", f"${unrealized:.2f}")
        st.metric("📊 Win Rate", f"{win_rate:.1f}%" if total_sells > 0 else "N/A")
        st.metric("🧮 Total Trades", f"{len(trade_df)}")

        if st.button("Download Trade Log as CSV"):
            csv = trade_df.to_csv(index=False)
            st.download_button("📥 Download CSV", csv, "trade_log.csv", "text/csv")

st.set_page_config(page_title="Paper Trader", layout="wide")
st.title("📊 Paper Trader Dashboard")

symbol = st.text_input("Enter Stock Symbol", "AAPL")
period = st.selectbox("Select Period", ["1d", "5d", "1mo", "3mo", "6mo", "1y"], index=2)
interval = st.selectbox("Select Interval", ["1m", "5m", "15m", "1h", "1d"], index=2)

if st.button("Fetch Data"):
    with st.spinner("Downloading data..."):
        data = yf.download(symbol, period=period, interval=interval)

        if not data.empty:
            st.success("Data fetched successfully!")

            enriched_data = engineer_features(data)
            if not enriched_data.empty:
                pred_df, model, accuracy = train_ai_model(enriched_data)
                recent_signal = int(pred_df.iloc[-1]['Prediction'])
                recent_price = float(pred_df.iloc[-1]['Close'])
                signal_text = "BUY" if recent_signal == 1 else "SELL"
                st.info(f"AI Suggests: {signal_text} at ${recent_price:.2f}")

                run_paper_trade(recent_signal, recent_price)
                st.plotly_chart(plot_candlestick_chart(data, pred_df), use_container_width=True)
                st.dataframe(data.tail(10))

                st.subheader("🔧 Candle Pattern Detection")
                pattern_df = detect_candle_patterns(data)
                if not pattern_df.empty:
                    st.dataframe(pattern_df[["Datetime", "Open", "High", "Low", "Close", "CandlePattern"]].tail(10))
                else:
                    st.info("No notable candle patterns detected.")

                st.subheader("🔎 Detected Large Movements")
                st.dataframe(detect_large_moves(data).tail(5))

                st.subheader("📈 Volume Spike Alerts")
                st.dataframe(detect_volume_spikes(data).tail(5))

                st.subheader("🧠 AI Movement Prediction")
                st.success(f"Prediction Model Accuracy: {accuracy:.2f}")
                st.dataframe(pred_df[['Close', 'Prediction']].tail(10))

                display_portfolio()
            else:
                st.error("Not enough data to train the model.")
        else:
            st.error("No data found for this symbol.")
