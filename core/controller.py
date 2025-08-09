from datetime import datetime
from core.schemas import TradeReason, Trade
from typing import List

class Controller:
    def __init__(self, strategies, broker, risk_engine, metrics, logger, config):
        self.strategies = strategies  # list of strategy instances
        self.broker = broker
        self.risk_engine = risk_engine
        self.metrics = metrics
        self.logger = logger
        self.config = config
        self.equity_start = config["portfolio"]["starting_cash"]

    def process_candles(self, timestamp, candles):
        """Get signals from all strategies, merge them, and execute."""
        if self.risk_engine.trading_halted:
            return

        for ticker, candle in candles.items():
            merged_signal = None
            top_confidence = 0
            reason_used = None

            # Collect and merge signals from strategies
            for strat in self.strategies:
                signal = strat.generate_signals(candles[ticker])
                if signal and signal.confidence > top_confidence:
                    merged_signal = signal
                    top_confidence = signal.confidence
                    reason_used = signal.reason

            if not merged_signal:
                continue

            # Risk sizing
            current_equity = self.broker.portfolio_value({t: c["close"] for t, c in candles.items()})
            qty = self.risk_engine.position_size(current_equity, candle["close"])

            if qty <= 0:
                continue

            # Execute trade
            trade_side = merged_signal.action
            executed = self.broker.execute_trade(ticker, trade_side, qty, candle["close"])
            if executed:
                trade_obj = Trade(
                    ts=timestamp,
                    ticker=ticker,
                    side=trade_side,
                    qty=qty,
                    price=candle["close"],
                    strategy=merged_signal.strategy,
                    confidence=merged_signal.confidence,
                    reason=reason_used,
                    data_mode=self.config["market"]["data_mode"],
                    pnl_realized=None
                )
                self.logger.log_trade(trade_obj.dict())

                # Update metrics for this ticker
                win = trade_side == "buy" and candle["close"] > reason_used.indicators.get("entry_price", candle["close"])
                self.metrics.update_per_stock(
                    ticker,
                    win,
                    merged_signal.confidence,
                    reason_used.indicators
                )

        # Update portfolio equity curve and benchmarks
        prices_now = {t: c["close"] for t, c in candles.items()}
        self.metrics.update_equity(
            timestamp,
            self.broker.portfolio_value(prices_now),
            list(prices_now.values())[0],  # benchmark: first ticker's price
            self.equity_start,
            max(prices_now.values())       # perfect trading placeholder
        )
