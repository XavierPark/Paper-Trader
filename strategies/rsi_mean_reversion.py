import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from .base import Strategy, Signal

class RSIMeanReversion(Strategy):
    name = "rsi_mean_reversion"

    def __init__(self, buy_th=30, sell_th=70, length=14):
        self.buy_th = buy_th
        self.sell_th = sell_th
        self.length = length

    def generate(self, history: pd.DataFrame) -> Signal:
        if history is None or history.empty:
            return Signal(ticker="", action="hold", confidence=0.0, reason={})
        close = history['close']
        rsi = RSIIndicator(close, window=self.length).rsi()
        rsi_val = float(rsi.iloc[-1]) if len(rsi) else np.nan
        ticker = history['ticker'].iloc[-1]
        reason = {
            "triggers": [],
            "indicators": {"rsi": rsi_val},
            "notes": "Mean reversion RSI"
        }
        if np.isnan(rsi_val):
            return Signal(ticker=ticker, action="hold", confidence=0.0, reason=reason)
        if rsi_val <= self.buy_th:
            reason["triggers"].append(f"RSI<={self.buy_th}")
            return Signal(ticker=ticker, action="buy", confidence=0.7, reason=reason)
        if rsi_val >= self.sell_th:
            reason["triggers"].append(f"RSI>={self.sell_th}")
            return Signal(ticker=ticker, action="sell", confidence=0.7, reason=reason)
        return Signal(ticker=ticker, action="hold", confidence=0.2, reason=reason)
