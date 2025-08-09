from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, Optional
from core.schemas import TradeReason

@dataclass
class Signal:
    ticker: str
    action: Literal["buy", "sell", "hold"]
    confidence: float
    reason: TradeReason
    strategy: str

class Strategy(ABC):
    name: str

    @abstractmethod
    def generate_signals(self, candle: dict) -> Optional[Signal]:
        pass
