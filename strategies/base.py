from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, Dict, Any
import pandas as pd

@dataclass
class Signal:
    ticker: str
    action: Literal["buy","sell","hold"] = "hold"
    confidence: float = 0.0
    reason: Dict[str, Any] | None = None

class Strategy(ABC):
    name: str = "base"

    @abstractmethod
    def generate(self, history: pd.DataFrame) -> Signal:
        ...
