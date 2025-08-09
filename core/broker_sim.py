class BrokerSim:
    def __init__(self, starting_cash=1000.0, allow_fractional=True):
        self.cash = starting_cash
        self.allow_fractional = allow_fractional
        self.positions = {}  # {ticker: {"qty": x, "avg_price": y}}
        self.trade_history = []

    def execute_trade(self, ticker, side, qty, price):
        cost = qty * price

        if side == "buy":
            if cost > self.cash:
                return False  # Not enough funds
            self.cash -= cost
            if ticker in self.positions:
                pos = self.positions[ticker]
                total_qty = pos["qty"] + qty
                pos["avg_price"] = ((pos["qty"] * pos["avg_price"]) + cost) / total_qty
                pos["qty"] = total_qty
            else:
                self.positions[ticker] = {"qty": qty, "avg_price": price}

        elif side == "sell":
            if ticker not in self.positions or self.positions[ticker]["qty"] < qty:
                return False  # Not enough shares
            self.cash += cost
            self.positions[ticker]["qty"] -= qty
            if self.positions[ticker]["qty"] == 0:
                del self.positions[ticker]

        self.trade_history.append({
            "ticker": ticker,
            "side": side,
            "qty": qty,
            "price": price
        })
        return True

    def portfolio_value(self, prices):
        value = self.cash
        for ticker, pos in self.positions.items():
            if ticker in prices:
                value += pos["qty"] * prices[ticker]
        return value
