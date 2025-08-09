import sqlite3
import pandas as pd
import os

class Logger:
    def __init__(self, db_path, export_csv=True, export_parquet=True):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.export_csv = export_csv
        self.export_parquet = export_parquet

    def log_trade(self, trade_dict):
        df = pd.DataFrame([trade_dict])
        df.to_sql("trades", self.conn, if_exists="append", index=False)
        if self.export_csv:
            df.to_csv("data/trades.csv", mode="a", index=False, header=not os.path.exists("data/trades.csv"))
        if self.export_parquet:
            df.to_parquet("data/trades.parquet", append=True)

    def log_summary(self, summary_dict):
        df = pd.DataFrame([summary_dict])
        df.to_sql("daily_summary", self.conn, if_exists="append", index=False)
