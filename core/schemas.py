from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Literal, Optional, List, Dict
from datetime import datetime

class TradeReason(BaseModel):
    triggers: List[str] = Field(default_factory=list)
    indicators: Dict[str, float] = Field(default_factory=dict)
    sentiment: Optional[float] = None
    regime: Optional[str] = None
    notes: Optional[str] = None

class Trade(BaseModel):
    ts: datetime
    ticker: str
    side: Literal["buy","sell"]
    qty: float
    price: float
    strategy: str
    confidence: float
    reason: TradeReason
    data_mode: Literal["live","replay","static"] = "replay"

class Position(BaseModel):
    ticker: str
    qty: float = 0.0
    avg_price: float = 0.0

class Order(BaseModel):
    ts: datetime
    ticker: str
    side: Literal["buy","sell"]
    qty: float
    limit_price: float
    strategy: str
    confidence: float
    reason: TradeReason
