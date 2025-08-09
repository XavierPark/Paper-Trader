import math

class RiskEngine:
    def __init__(self, config):
        self.per_trade_risk_pct = config["risk"]["per_trade_risk_pct"]
        self.max_daily_drawdown_pct = config["risk"]["max_daily_drawdown_pct"]
        self.max_open_positions = config["risk"]["max_open_positions"]
        self.stop_loss_pct = config["risk"]["stop_loss_pct"]
        self.trailing_stop_pct = config["risk"]["trailing_stop_pct"]

        self.starting_equity = config["portfolio"]["starting_cash"]
        self.daily_loss_limit = self.starting_equity * self.max_daily_drawdown_pct
        self.equity_peak = self.starting_equity
        self.trading_halted = False

    def position_size(self, equity, price):
        """Size trade based on risk percentage and price."""
        dollar_risk = equity * self.per_trade_risk_pct
        qty = dollar_risk / (price * self.stop_loss_pct)
        return qty

    def check_drawdown(self, current_equity):
        """Halt trading if daily drawdown exceeded."""
        loss_today = self.starting_equity - current_equity
        if loss_today >= self.daily_loss_limit:
            self.trading_halted = True
            return False
        return True

    def update_equity_peak(self, current_equity):
        self.equity_peak = max(self.equity_peak, current_equity)
