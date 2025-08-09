import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta


class ReplayFeed:
    def __init__(self, tickers, start_date, end_date, interval="1m"):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        self.data = {}
        self.pointer = 0

    def load_data(self):
        for ticker in self.tickers:
            df = yf.download(
                ticker,
                start=self.start_date,
                end=self.end_date,
                interval=self.interval,
                progress=False
            )
            df = df.reset_index()
            df["Datetime"] = pd.to_datetime(df["Datetime"])
            self.data[ticker] = df
        self.pointer = 0

    def get_next_candle(self):
        """Yield the next timestamp's candles for all tickers"""
        if not self.data:
            raise ValueError("Data not loaded. Call load_data() first.")

        try:
            timestamp = self.data[self.tickers[0]].iloc[self.pointer]["Datetime"]
        except IndexError:
            return None  # End of data

        candles = {}
        for ticker in self.tickers:
            row = self.data[ticker].iloc[self.pointer]
            candles[ticker] = {
                "open": row["Open"],
                "high": row["High"],
                "low": row["Low"],
                "close": row["Close"],
                "volume": row["Volume"],
                "timestamp": timestamp
            }

        self.pointer += 1
        return timestamp, candles
