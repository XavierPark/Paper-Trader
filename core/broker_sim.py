from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List
from datetime import datetime
import sqlite3, os

@dataclass
class TradeExec:
    ts: datetime
    ticker: str
    side: str
    qty: float
    price: float
    strategy: str
    confidence: float
    reason_json: str
    data_mode: str = "replay"

@dataclass
class Position:
    qty: float = 0.0
    avg_price: float = 0.0

@dataclass
class BrokerSim:
    cash: float
    allow_fractional: bool = True
    db_path: str = "data/sqlite/stockapp.db"
    positions: Dict[str, Position] = field(default_factory=dict)
    trades: List[TradeExec] = field(default_factory=list)

    def _ensure_db(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        con = sqlite3.connect(self.db_path)
        cur = con.cursor()
        cur.execute("""                CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT, ticker TEXT, side TEXT, qty REAL, price REAL,
                strategy TEXT, confidence REAL, reason_json TEXT, data_mode TEXT
            )""")
        cur.execute("""                CREATE TABLE IF NOT EXISTS positions (
                ticker TEXT PRIMARY KEY, qty REAL, avg_price REAL
            )""")
        con.commit()
        con.close()

    def _write_trade(self, t: TradeExec):
        self._ensure_db()
        con = sqlite3.connect(self.db_path)
        cur = con.cursor()
        cur.execute("INSERT INTO trades(ts,ticker,side,qty,price,strategy,confidence,reason_json,data_mode) VALUES (?,?,?,?,?,?,?,?,?)",
                    (t.ts.isoformat(), t.ticker, t.side, t.qty, t.price, t.strategy, t.confidence, t.reason_json, t.data_mode))
        con.commit()
        con.close()

    def _update_position(self, ticker: str, side: str, qty: float, price: float):
        pos = self.positions.get(ticker, Position())
        if side == "buy":
            total_cost = pos.avg_price * pos.qty + price * qty
            new_qty = pos.qty + qty
            pos.avg_price = total_cost / new_qty if new_qty > 0 else 0.0
            pos.qty = new_qty
        elif side == "sell":
            pos.qty -= qty
            if pos.qty <= 0:
                pos.qty = 0.0
                pos.avg_price = 0.0
        self.positions[ticker] = pos

    def execute_market(self, ts: datetime, ticker: str, side: str, qty: float, price: float,
                       strategy: str, confidence: float, reason_json: str, data_mode: str = "replay"):
        notional = qty * price
        if side == "buy" and notional > self.cash:
            return False  # insufficient cash
        if side == "buy":
            self.cash -= notional
        else:
            # allow selling up to held qty (no shorting here)
            held = self.positions.get(ticker, Position()).qty
            qty = min(qty, held)
            notional = qty * price
            self.cash += notional
        trade = TradeExec(ts, ticker, side, qty, price, strategy, confidence, reason_json, data_mode)
        self.trades.append(trade)
        self._update_position(ticker, side, qty, price)
        self._write_trade(trade)
        return True
