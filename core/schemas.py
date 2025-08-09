from pydantic import BaseModel
from typing import List, Literal, Optional
from datetime import datetime


class TradeReason(BaseModel):
    triggers: List[str]  # e.g., ["RSI<30", "50SMA cross"]
    indicators: dict     # e.g., {"rsi": 28.4, "sma50": 182.1}
    sentiment: Optional[float] = None  # -1..1
    regime: Optional[str] = None  # trending|range|volatile
    notes: Optional[str] = None


class Trade(BaseModel):
    ts: datetime
    ticker: str
    side: Literal["buy", "sell"]
    qty: float
    price: float
    strategy: str
    confidence: float
    reason: TradeReason
    data_mode: Literal["live", "replay", "static"]
    pnl_realized: Optional[float] = None


class Order(BaseModel):
    ts: datetime
    ticker: str
    side: Literal["buy", "sell"]
    qty: float
    limit_price: Optional[float]
    status: Literal["open", "filled", "cancelled"]
    strategy: str
    confidence: float
    reason: TradeReason
