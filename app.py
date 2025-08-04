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
        'trade_log': [],
        'daily_summary': {},
        'returns': [],
        'realized_pl': 0.0,
        'unrealized_pl': 0.0
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

def plot_candlestick_chart(df, pred_df=None, trade_log=None):
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

    if trade_log:
        trade_df = pd.DataFrame(trade_log)
        for i, row in trade_df.iterrows():
            color = 'green' if row['Type'] == 'BUY' else 'red'
            fig.add_trace(go.Scatter(
                x=[row['Datetime']],
                y=[row['Price']],
                mode='markers+text',
                marker=dict(color=color, size=10),
                name=row['Type'],
                text=[f"{row['Type']}\n{row['Shares']} @ ${row['Price']:.2f}"],
                textposition="top center"
            ))

    fig.update_layout(
        title='Candlestick Chart with Trades',
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
            'Signal': 'BUY',
            'Symbol': st.session_state.get('symbol', 'N/A')
        })

    elif signal == 0 and p['shares'] > 0:
        total_gain = p['shares'] * price
        profit = total_gain - (p['shares'] * p['last_price'])
        p['cash'] += total_gain
        portfolio_value = p['cash']
        trade_msg = f"SELL {p['shares']} @ ${price:.2f}"
        p['history'].append(trade_msg)
        p['realized_pl'] += profit
        p['trade_log'].append({
            'Datetime': now,
            'Type': 'SELL',
            'Shares': p['shares'],
            'Price': price,
            'Total Value': total_gain,
            'Cash After': p['cash'],
            'Portfolio Value After': portfolio_value,
            'Signal': 'SELL',
            'Symbol': st.session_state.get('symbol', 'N/A')
        })
        p['shares'] = 0
        p['last_price'] = price

    # Unrealized P/L
    p['unrealized_pl'] = p['shares'] * (price - p['last_price'])

    # Track returns
    start_value = 10000.0
    curr_value = p['cash'] + p['shares'] * price
    cumulative_return = (curr_value - start_value) / start_value
    p['returns'].append({'Datetime': now, 'Return': cumulative_return})
    date_key = now.split(" ")[0]
    p['daily_summary'][date_key] = p['daily_summary'].get(date_key, 0) + 1
