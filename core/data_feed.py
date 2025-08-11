import yfinance as yf
import pandas as pd

class ReplayFeed:
    """Streams historical intraday candles as if live."""
    def __init__(self, tickers, period="5d", interval="1m"):
        self.tickers = tickers
        self.period = period
        self.interval = interval
        self.cache = {}
        self.index = 0
        self.timeline = None

    def load(self):
        frames = []
        for t in self.tickers:
            df = yf.download(t, period=self.period, interval=self.interval, progress=False)
            if df.empty:
                continue
            df = df.rename(columns=str.lower)
            df['ticker'] = t
            df.index = pd.to_datetime(df.index).tz_localize(None)
            frames.append(df)
            self.cache[t] = df
        if not frames:
            raise RuntimeError("No data loaded from yfinance. Try different tickers/period/interval.")
        self.timeline = frames[0].index.to_list()
        self.index = 0

    def has_next(self):
        return self.timeline is not None and self.index < len(self.timeline)

    def next_tick(self):
        if not self.has_next():
            return None
        ts = self.timeline[self.index]
        self.index += 1
        batch = {}
        for t, df in self.cache.items():
            batch[t] = df.loc[:ts].copy()
        return ts, batch
