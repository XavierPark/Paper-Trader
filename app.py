import os, json
import streamlit as st
import pandas as pd
from datetime import datetime
from core.data_feed import ReplayFeed
from core.broker_sim import BrokerSim
from core.risk import RiskConfig, RiskEngine
from core.metrics import load_trades, equity_curve
from strategies.rsi_mean_reversion import RSIMeanReversion

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

st.set_page_config(page_title="StockApp", layout="wide")
st.title("📈 StockApp — Paper Trading (Privacy‑First)")

# Sidebar config
with st.sidebar:
    st.header("Settings")
    tickers = st.multiselect("Universe", ["AAPL","MSFT","NVDA","SPY","TSLA","AMZN"], default=["AAPL","MSFT","NVDA","SPY"])
    starting_cash = st.number_input("Starting Cash ($)", 100.0, 1_000_000.0, 1000.0, step=100.0)
    buy_th = st.slider("RSI Buy Threshold", 10, 40, 30)
    sell_th = st.slider("RSI Sell Threshold", 60, 90, 70)
    per_trade_risk = st.slider("Per‑Trade Risk %", 0.1, 5.0, 1.0) / 100.0
    stop_loss_pct = st.slider("Stop Loss %", 0.5, 10.0, 3.0) / 100.0
    run = st.button("Next Tick ▶️")
    reset = st.button("Reset Session 🔄")

# Session state
if "feed" not in st.session_state or reset:
    st.session_state.feed = ReplayFeed(tickers=tickers, period="5d", interval="1m")
    st.session_state.feed.load()
    st.session_state.broker = BrokerSim(cash=starting_cash)
    st.session_state.strategy = RSIMeanReversion(buy_th=buy_th, sell_th=sell_th)
    st.session_state.risk = RiskEngine(RiskConfig(per_trade_risk_pct=per_trade_risk, stop_loss_pct=stop_loss_pct))
    st.session_state.equity = starting_cash

feed = st.session_state.feed
broker = st.session_state.broker
strat = st.session_state.strategy
risk = st.session_state.risk

# Tick processing
if run:
    if feed.has_next():
        tick_ts, tick_batch = feed.next_tick()
        for tkr, hist in tick_batch.items():
            if hist.empty:
                continue
            sig = strat.generate(hist)
            if sig.action in ("buy","sell"):
                price = float(hist['close'].iloc[-1])
                # Position sizing (fractional)
                equity_now = broker.cash + sum(pos.qty*pos.avg_price for pos in broker.positions.values())
                qty = risk.size_fractional(equity=equity_now, price=price)
                if qty <= 0:
                    continue
                reason_json = json.dumps(sig.reason)
                broker.execute_market(
                    ts=tick_ts, ticker=tkr, side=sig.action, qty=qty, price=price,
                    strategy=strat.name, confidence=sig.confidence, reason_json=reason_json
                )
    else:
        st.info("No more ticks in replay. Reset to start over.")

# UI Layout
col1, col2 = st.columns([2,1])

with col1:
    st.subheader("Equity & Trades")
    trades = load_trades(broker.db_path)
    if not trades.empty:
        ec = equity_curve(starting_cash, trades)
        st.line_chart(ec.set_index("ts"))
    st.dataframe(trades.tail(20))

with col2:
    st.subheader("Positions / Cash")
    pos_rows = [{"ticker": t, "qty": p.qty, "avg_price": p.avg_price} for t, p in broker.positions.items()]
    pos_df = pd.DataFrame(pos_rows) if pos_rows else pd.DataFrame(columns=["ticker","qty","avg_price"])
    st.dataframe(pos_df)
    st.metric(label="Cash", value=f"$ {broker.cash:,.2f}")

st.markdown("---")
st.subheader("Live Trade Reasons (last 10)")
trades = load_trades(broker.db_path)
if not trades.empty:
    last = trades.tail(10).copy()
    st.dataframe(last[["ts","ticker","side","qty","price","strategy","confidence","reason_json"]])

st.caption("Educational prototype. Not investment advice.")
