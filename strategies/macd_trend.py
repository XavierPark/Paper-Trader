import pandas as pd
import ta
from core.schemas import TradeReason
from strategies.base import Strategy, Signal

class MACDTrend(Strategy):
    def __init__(self, ticker, fast=12, slow=26, signal=9):
        self.name = "MACD Trend"
        self.ticker = ticker
        self.fast = fast
        self.slow = slow
        self.signal_period = signal
        self.history = pd.DataFrame(columns=["close"])

    def generate_signals(self, candle: dict):
        # Append new close to history
        self.history.loc[len(self.history)] = [candle["close"]]
        if len(self.history) < self.slow:
            return None

        macd_ind = ta.trend.MACD(
            self.history["close"],
            window_slow=self.slow,
            window_fast=self.fast,
            window_sign=self.signal_period
        )

        macd_val = macd_ind.macd().iloc[-1]
        macd_signal = macd_ind.macd_signal().iloc[-1]

        action = "hold"
        confidence = 0.0
        if macd_val > macd_signal:
            action = "buy"
            confidence = min(1.0, (macd_val - macd_signal) / abs(macd_signal) if macd_signal != 0 else 0.5)
        elif macd_val < macd_signal:
            action = "sell"
            confidence = min(1.0, (macd_signal - macd_val) / abs(macd_signal) if macd_signal != 0 else 0.5)

        if action != "hold":
            reason = TradeReason(
                triggers=[f"MACD {macd_val:.4f} vs Signal {macd_signal:.4f}"],
                indicators={"macd": round(macd_val, 4), "trend": "up" if action == "buy" else "down"},
                notes="MACD crossover trend signal"
            )
            return Signal(
                ticker=self.ticker,
                action=action,
                confidence=confidence,
                reason=reason,
                strategy=self.name
            )
        return None
