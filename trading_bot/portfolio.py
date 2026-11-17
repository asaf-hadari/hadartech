"""Virtual portfolio + order decisions. Shared by the backtester and the paper trader
so that what is tested is exactly what runs live."""
from dataclasses import dataclass, field, asdict
import math


@dataclass
class Position:
    symbol: str
    shares: float
    entry_price: float
    entry_date: str
    stop: float
    highest: float


@dataclass
class Portfolio:
    cash: float
    positions: dict = field(default_factory=dict)
    trades: list = field(default_factory=list)

    def equity(self, prices):
        return self.cash + sum(p.shares * prices.get(s, p.entry_price)
                               for s, p in self.positions.items())

    def _cost(self, value, costs):
        return max(costs.min_commission, value * costs.commission_pct)

    def buy(self, symbol, price, shares, date, stop, costs):
        fill = price * (1 + costs.slippage_pct)
        shares = math.floor(shares)
        if shares <= 0:
            return False
        value = fill * shares
        fee = self._cost(value, costs)
        if value + fee > self.cash:
            shares = math.floor((self.cash - costs.min_commission) / (fill * (1 + costs.commission_pct)))
            if shares <= 0:
                return False
            value = fill * shares
            fee = self._cost(value, costs)
        self.cash -= value + fee
        self.positions[symbol] = Position(symbol, shares, fill, str(date), stop, fill)
        self.trades.append({"date": str(date), "symbol": symbol, "side": "BUY",
                            "shares": shares, "price": round(fill, 4), "fee": round(fee, 2)})
        return True

    def sell(self, symbol, price, date, costs, reason=""):
        pos = self.positions.pop(symbol, None)
        if pos is None:
            return None
        fill = price * (1 - costs.slippage_pct)
        value = fill * pos.shares
        fee = self._cost(value, costs)
        self.cash += value - fee
        pnl = value - fee - pos.shares * pos.entry_price
        self.trades.append({"date": str(date), "symbol": symbol, "side": "SELL",
                            "shares": pos.shares, "price": round(fill, 4), "fee": round(fee, 2),
                            "pnl": round(pnl, 2),
                            "ret_pct": round(100 * (fill / pos.entry_price - 1), 2),
                            "reason": reason})
        return pnl

    def to_dict(self):
        return {"cash": self.cash,
                "positions": {s: asdict(p) for s, p in self.positions.items()},
                "trades": self.trades}

    @classmethod
    def from_dict(cls, d):
        return cls(d["cash"], {s: Position(**p) for s, p in d["positions"].items()},
                   d.get("trades", []))


def update_stops(pf, closes, atrs, mult):
    """Ratchet each trailing stop upward. Returns symbols whose stop was hit."""
    hit = []
    for s, pos in pf.positions.items():
        c, a = closes.get(s), atrs.get(s)
        if c is None or not math.isfinite(c):
            continue
        pos.highest = max(pos.highest, c)
        if a is not None and math.isfinite(a):
            pos.stop = max(pos.stop, pos.highest - mult * a)
        if c < pos.stop:
            hit.append(s)
    return hit


def plan_orders(pf, row, params, equity):
    """Given today's signal row (dict-of-dicts), decide exits and ranked entries.

    row keys: entry, trend_exit, score, atr, close  -> {symbol: value}
    Returns (exits: list[(symbol, reason)], entries: list[(symbol, shares, stop)])
    """
    exits = []
    stop_hits = update_stops(pf, row["close"], row["atr"], params.atr_stop_mult)
    for s in pf.positions:
        if s in stop_hits:
            exits.append((s, "trailing_stop"))
        elif row["trend_exit"].get(s, False):
            exits.append((s, "trend_reversal"))

    exiting = {s for s, _ in exits}
    slots = params.max_positions - (len(pf.positions) - len(exiting))
    entries = []
    if slots > 0:
        cands = [s for s, ok in row["entry"].items()
                 if ok and s not in pf.positions and math.isfinite(row["score"].get(s, float("nan")))]
        cands.sort(key=lambda s: row["score"][s], reverse=True)
        for s in cands[:slots]:
            price, a = row["close"][s], row["atr"].get(s)
            if not (price and a and math.isfinite(a) and a > 0):
                continue
            stop_dist = params.atr_stop_mult * a
            by_alloc = (equity / params.max_positions) / price
            by_risk = (equity * params.risk_per_trade) / stop_dist
            entries.append((s, min(by_alloc, by_risk), price - stop_dist))
    return exits, entries
