import pandas as pd
import sqlite3

def load_trades(db_path: str) -> pd.DataFrame:
    con = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM trades", con, parse_dates=["ts"])
    con.close()
    if not df.empty:
        df = df.sort_values("ts")
    return df

def equity_curve(starting_cash: float, trades: pd.DataFrame) -> pd.DataFrame:
    cash = starting_cash
    rows = []
    for _, r in trades.iterrows():
        notional = r.qty * r.price
        cash += -notional if r.side == "buy" else notional
        rows.append({"ts": r.ts, "cash": cash})
    ec = pd.DataFrame(rows)
    return ec
