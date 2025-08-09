import streamlit as st
from datetime import datetime, timedelta
import pandas as pd
import time

from configparser import ConfigParser
import yaml

# Core imports
from core.data_feed import ReplayFeed
from core.broker_sim import BrokerSim
from core.risk import RiskEngine
from core.logger import Logger
from core.metrics import Metrics
from core.controller import Controller

# Strategies
from strategies.rsi_mean_reversion import RSIMeanReversion
from strategies.macd_trend import MACDTrend

# --- Load Config ---
with open("config/defaults.yaml", "r") as f:
    config = yaml.safe_load(f)

# --- Streamlit Page Config ---
st.set_page_config(page_title="AI Stock Replay Trader", layout="wide")

st.title("📈 AI Stock Trading Simulator (Replay Mode)")
st.write("This simulates historical market data candle-by-candle so the AI trades as if live.")

# --- Sidebar Controls ---
st.sidebar.header("Simulation Settings")

tickers = st.sidebar.multiselect(
    "Select tickers:",
    config["market"]["universe"],
    default=config["market"]["universe"]
)

days_back = st.sidebar.slider("Days of history to replay", 1, 10, 5)
start_date = datetime.now() - timedelta(days=days_back)
end_date = datetime.now()

refresh_seconds = st.sidebar.slider("Replay speed (seconds per candle)", 1, 10, config["ui"]["refresh_seconds"])

st.sidebar.markdown("---")
st.sidebar.write("**Strategies Enabled:**")
use_rsi = st.sidebar.checkbox("RSI Mean Reversion", value=True)
use_macd = st.sidebar.checkbox("MACD Trend", value=True)

# --- Initialize Systems ---
feed = ReplayFeed(tickers, start_date, end_date, interval=config["market"]["resolution"])
feed.load_data()

broker = BrokerSim(starting_cash=config["portfolio"]["starting_cash"], allow_fractional=config["portfolio"]["allow_fractional"])
risk_engine = RiskEngine(config)
metrics = Metrics()
logger = Logger(config["logging"]["db_path"], export_csv=config["logging"]["export_csv"], export_parquet=config["logging"]["export_parquet"])

strategies = []
if use_rsi:
    for t in tickers:
        strategies.append(RSIMeanReversion(ticker=t))
if use_macd:
    for t in tickers:
        strategies.append(MACDTrend(ticker=t))

controller = Controller(strategies, broker, risk_engine, metrics, logger, config)

# --- UI Placeholders ---
equity_chart = st.empty()
verdict_table_placeholder = st.empty()
trade_log_placeholder = st.empty()

trade_log_df = pd.DataFrame(columns=["Time", "Ticker", "Side", "Qty", "Price", "Strategy", "Confidence", "Reason"])

# --- Run Simulation ---
st.write(f"▶ Running Replay from {start_date.date()} to {end_date.date()} ...")

while True:
    next_candle = feed.get_next_candle()
    if not next_candle:
        break

    timestamp, candles = next_candle
    controller.process_candles(timestamp, candles)

    # Update verdict table in real-time
    verdicts = metrics.get_realtime_verdicts()
    verdict_df = pd.DataFrame([
        {
            "Ticker": t,
            "Verdict": v["verdict"],
            "Win Rate %": v["win_rate"],
            "Avg Conf": v["avg_conf"],
            "Trend": v["trend"],
            "Notes": v["notes"]
        }
        for t, v in verdicts.items()
    ])
    verdict_table_placeholder.dataframe(verdict_df)

    # Update equity curve
    eq_df = pd.DataFrame(metrics.equity_curve)
    equity_chart.line_chart(eq_df.set_index("ts"))

    # Update trade log
    if broker.trade_history:
        last_trade = broker.trade_history[-1]
        trade_log_df.loc[len(trade_log_df)] = [
            timestamp, last_trade["ticker"], last_trade["side"],
            last_trade["qty"], last_trade["price"],
            "N/A",  # Strategy placeholder
            0.0,    # Confidence placeholder
            "Reason N/A"
        ]
        trade_log_placeholder.dataframe(trade_log_df)

    time.sleep(refresh_seconds)

# --- Final Summary ---
st.header("📊 Final Per-Stock Verdicts")
final_summary = metrics.get_final_summary()
final_df = pd.DataFrame([
    {
        "Ticker": t,
        "Verdict": v["verdict"],
        "Win Rate %": v["win_rate"],
        "Avg Conf": v["avg_conf"],
        "Trend": v["trend"],
        "Notes": v["notes"]
    }
    for t, v in final_summary.items()
])
st.dataframe(final_df)

st.download_button("Download Trade Log (CSV)", trade_log_df.to_csv(index=False), "trade_log.csv")
