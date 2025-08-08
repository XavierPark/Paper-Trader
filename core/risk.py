from __future__ import annotations
from dataclasses import dataclass

@dataclass
class RiskConfig:
    per_trade_risk_pct: float = 0.01
    stop_loss_pct: float = 0.03
    trailing_stop_pct: float = 0.02
    max_open_positions: int = 5

class RiskEngine:
    def __init__(self, cfg: RiskConfig):
        self.cfg = cfg

    def dollar_risk(self, equity: float) -> float:
        return equity * self.cfg.per_trade_risk_pct

    def size_fractional(self, equity: float, price: float) -> float:
        # Simple: risk $ / (stop % * price) then translate to qty
        risk_dollars = self.dollar_risk(equity)
        stop_distance = max(price * self.cfg.stop_loss_pct, 0.01)
        position_dollars = risk_dollars / (stop_distance / price)
        qty = position_dollars / price
        return max(qty, 0.0)
