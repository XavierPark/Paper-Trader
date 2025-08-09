import pandas as pd
import numpy as np

class Metrics:
    def __init__(self):
        self.equity_curve = []
        self.benchmark_curve = []
        self.perfect_curve = []
        self.per_stock_stats = {}

    def update_equity(self, timestamp, equity, benchmark_price, start_price, perfect_price):
        self.equity_curve.append({"ts": timestamp, "equity": equity})
        bh_return = (benchmark_price - start_price) / start_price
        perfect_return = (perfect_price - start_price) / start_price
        self.benchmark_curve.append({"ts": timestamp, "return": bh_return})
        self.perfect_curve.append({"ts": timestamp, "return": perfect_return})

    def update_per_stock(self, ticker, win, confidence, indicators):
        if ticker not in self.per_stock_stats:
            self.per_stock_stats[ticker] = {
                "trades": 0, "wins": 0, "avg_conf": 0.0, "indicators": {}
            }
        stats = self.per_stock_stats[ticker]
        stats["trades"] += 1
        stats["wins"] += 1 if win else 0
        stats["avg_conf"] = ((stats["avg_conf"] * (stats["trades"] - 1)) + confidence) / stats["trades"]
        stats["indicators"] = indicators

    def get_realtime_verdicts(self):
        verdicts = {}
        for ticker, stats in self.per_stock_stats.items():
            win_rate = stats["wins"] / stats["trades"] if stats["trades"] > 0 else 0
            avg_conf = stats["avg_conf"]
            ind = stats["indicators"]

            if win_rate > 0.6 and ind.get("trend") == "up":
                verdict = "✅ Buy"
            elif 0.4 <= win_rate <= 0.6:
                verdict = "⚠ Hold"
            else:
                verdict = "❌ Avoid"

            verdicts[ticker] = {
                "verdict": verdict,
                "win_rate": round(win_rate * 100, 1),
                "avg_conf": round(avg_conf, 2),
                "trend": ind.get("trend", "N/A"),
                "notes": f"RSI {ind.get('rsi')}, MACD {ind.get('macd')}"
            }
        return verdicts

    def get_final_summary(self):
        return self.get_realtime_verdicts()
