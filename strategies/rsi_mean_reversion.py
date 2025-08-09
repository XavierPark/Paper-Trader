import pandas as pd
import ta
from core.schemas import TradeReason
from strategies.base import Strategy, Signal

class RSIMeanReversion(Strategy):
    def __init__(self, ticker, window=14, buy_level=30, sell_level=70):
        self.name = "RSI Mean Reversion"
        self.ticker = ticker
        self.window = window
        self.buy_level = buy_level
        self.sell_level = sell_level
        self.history = pd.DataFrame(columns=["close"])

    def generate_signals(self, candle: dict):
        # Append new close to history
        self.history.loc[len(self.history)] = [candle["close"]]
        if len(self.history) < self.window:
            return None

        # Calculate RSI
        rsi = ta.momentum.RSIIndicator(self.history["close"], window=self.window).rsi().iloc[-1]

        action = "hold"
        confidence = 0.0
        if rsi < self.buy_level:
            action = "buy"
            confidence = min(1.0, (self.buy_level - rsi) / self.buy_level)
        elif rsi > self.sell_level:
            action = "sell"
            confidence = min(1.0, (rsi - self.sell_level) / (100 - self.sell_level))

        if action != "hold":
            reason = TradeReason(
                triggers=[f"RSI {rsi:.2f}"],
                indicators={"rsi": round(rsi, 2), "trend": "down" if action == "buy" else "up"},
                notes="RSI mean reversion signal"
            )
            return Signal(
                ticker=self.ticker,
                action=action,
                confidence=confidence,
                reason=reason,
                strategy=self.name
            )
        return None
