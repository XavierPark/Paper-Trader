# core/metrics.py
import os
import pandas as pd
import sqlite3

COLUMNS = ["ts","ticker","side","qty","price","strategy","confidence","reason_json","data_mode"]

def load_trades(db_path: str) -> pd.DataFrame:
    # Ensure parent directory exists
    parent = os.path.dirname(db_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    # If DB file doesn't exist yet, return empty
    if not os.path.exists(db_path):
        return pd.DataFrame(columns=COLUMNS)

    try:
        con = sqlite3.connect(db_path)
        df = pd.read_sql_query("SELECT * FROM trades", con, parse_dates=["ts"])
        con.close()
    except Exception:
        return pd.DataFrame(columns=COLUMNS)

    if not df.empty:
        df = df.sort_values("ts")
    return df

def equity_curve(starting_cash: float, trades: pd.DataFrame) -> pd.DataFrame:
    """
    Simple running cash curve based on executed trades.
    (Does not mark open positions to market—good enough for a basic chart.)
    """
    cash = starting_cash
    rows = []
    for _, r in trades.iterrows():
        notional = float(r["qty"]) * float(r["price"])
        cash += -notional if r["side"] == "buy" else notional
        rows.append({"ts": r["ts"], "cash": cash})
    return pd.DataFrame(rows, columns=["ts","cash"])
