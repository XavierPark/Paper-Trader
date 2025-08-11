# core/metrics.py
import os
import pandas as pd
import sqlite3

COLUMNS = ["ts","ticker","side","qty","price","strategy","confidence","reason_json","data_mode"]

def load_trades(db_path: str) -> pd.DataFrame:
    # Ensure parent directory exists (or use current dir if none)
    parent = os.path.dirname(db_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    # If DB file doesn't exist yet, return an empty, well-shaped DataFrame
    if not os.path.exists(db_path):
        return pd.DataFrame(columns=COLUMNS)

    try:
        con = sqlite3.connect(db_path)
        # If table isn't created yet, this will fail — catch and return empty
        df = pd.read_sql_query("SELECT * FROM trades", con, parse_dates=["ts"])
        con.close()
    except Exception:
        return pd.DataFrame(columns=COLUMNS)

    if not df.empty:
        df = df.sort_values("ts")
    return df
