"""Virtual (paper) trading on live market data.

Each `step()`:
  1. fetches fresh daily bars + live prices,
  2. sells immediately any position whose live price broke its trailing stop,
  3. once per completed daily bar, runs the strategy and executes virtual
     orders at the live price (same logic as the backtest),
  4. records equity, and
  5. periodically re-optimises while the live average monthly return is below
     the target - the "keeps improving itself" loop.
"""
from datetime import datetime, time as dtime
import json
import math
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from .backtest import monthly_returns
from .config import StrategyParams
from .portfolio import Portfolio, plan_orders
from .strategy import compute_signals

MARKET_HOURS = {  # timezone, close time (daily bar is final after this)
    "nasdaq100": ("America/New_York", dtime(16, 5)),
    "nasdaq_all": ("America/New_York", dtime(16, 5)),
    "ta35": ("Asia/Jerusalem", dtime(17, 35)),
}


class PaperTrader:
    def __init__(self, cfg, fetch_data, fetch_prices, state_dir="bot_state",
                 reoptimize_every_days=7, reopt_iters=150, log=print):
        self.cfg, self.fetch_data, self.fetch_prices, self.log = cfg, fetch_data, fetch_prices, log
        self.dir = Path(state_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.dir / "paper_state.json"
        self.trades_csv = self.dir / "paper_trades.csv"
        self.reopt_every = reoptimize_every_days
        self.reopt_iters = reopt_iters
        self._load()

    # ---------- persistence ----------
    def _load(self):
        if self.state_path.exists():
            s = json.loads(self.state_path.read_text())
            self.pf = Portfolio.from_dict(s["portfolio"])
            self.equity_log = s.get("equity_log", [])
            self.last_bar = s.get("last_bar")
            self.last_reopt = s.get("last_reopt")
            self.params = StrategyParams.from_dict(s["params"])
        else:
            self.pf = Portfolio(cash=self.cfg.initial_capital)
            self.equity_log, self.last_bar, self.last_reopt = [], None, None
            champ = self.dir / "champion.json"
            self.params = (StrategyParams.from_dict(json.loads(champ.read_text())["params"])
                           if champ.exists() else self.cfg.params)

    def _save(self):
        self.state_path.write_text(json.dumps({
            "portfolio": self.pf.to_dict(), "equity_log": self.equity_log[-5000:],
            "last_bar": self.last_bar, "last_reopt": self.last_reopt,
            "params": self.params.to_dict()}, indent=2))
        if self.pf.trades:
            pd.DataFrame(self.pf.trades).to_csv(self.trades_csv, index=False)

    # ---------- helpers ----------
    def _completed_bar_index(self, data, now):
        tz, close_t = MARKET_HOURS.get(self.cfg.market, ("America/New_York", dtime(16, 5)))
        local = now.astimezone(ZoneInfo(tz))
        last = data.dates[-1].date()
        if last == local.date() and local.time() < close_t:
            return len(data.dates) - 2  # today's bar still forming
        return len(data.dates) - 1

    def live_monthly_avg(self):
        if len(self.equity_log) < 2:
            return None
        eq = pd.Series({pd.Timestamp(e["time"]): e["equity"] for e in self.equity_log})
        eq = eq.groupby(eq.index.normalize()).last()
        mr = monthly_returns(eq)
        return float(mr.mean()) if len(mr) else None

    # ---------- main step ----------
    def step(self, now=None):
        now = now or datetime.now(ZoneInfo("UTC"))
        data = self.fetch_data()
        held = list(self.pf.positions)
        prices = self.fetch_prices(list(set(held)))
        last_close = data.close.iloc[-1].to_dict()
        px = lambda s: prices.get(s) or last_close.get(s)

        # 1) intraday stop check on live prices
        for s in held:
            p, pos = px(s), self.pf.positions[s]
            if p and p < pos.stop:
                self.pf.sell(s, p, now.date(), self.cfg.costs, "trailing_stop_live")
                self.log(f"SELL {s} @ {p:.2f} (live stop {pos.stop:.2f})")

        # 2) strategy decisions once per completed bar
        t = self._completed_bar_index(data, now)
        bar_date = str(data.dates[t].date())
        if bar_date != self.last_bar:
            sig = compute_signals(data, self.params)
            row = {
                "entry": {s: True for s, v in sig.entry.iloc[t].items() if v},
                "trend_exit": sig.trend_exit.iloc[t].to_dict(),
                "score": sig.score.iloc[t].to_dict(),
                "atr": sig.atr.iloc[t].to_dict(),
                "close": data.close.iloc[t].to_dict(),
            }
            equity = self.pf.equity({s: px(s) for s in self.pf.positions})
            exits, entries = plan_orders(self.pf, row, self.params, equity)
            want = list({s for s, _ in exits} | {s for s, _, _ in entries})
            prices.update(self.fetch_prices(want) if want else {})
            for s, reason in exits:
                self.pf.sell(s, px(s), now.date(), self.cfg.costs, reason)
                self.log(f"SELL {s} @ {px(s):.2f} ({reason})")
            for s, shares, stop in entries:
                if self.pf.buy(s, px(s), shares, now.date(), stop, self.cfg.costs):
                    self.log(f"BUY  {s} x{math.floor(shares)} @ {px(s):.2f} stop {stop:.2f}")
            self.last_bar = bar_date

        # 3) record equity
        eq = self.pf.equity({s: px(s) for s in self.pf.positions})
        self.equity_log.append({"time": now.isoformat(timespec="seconds"), "equity": round(eq, 2)})
        self._save()
        return eq

    # ---------- self-improvement while live ----------
    def maybe_reoptimize(self, data, now=None):
        from .optimizer import SelfImprover
        now = now or datetime.now(ZoneInfo("UTC"))
        if self.last_reopt and (now - datetime.fromisoformat(self.last_reopt)).days < self.reopt_every:
            return False
        avg = self.live_monthly_avg()
        target = self.cfg.target_monthly_return
        if avg is not None and avg >= target:
            self.log(f"Live avg monthly {avg:.2%} >= target {target:.0%}: keeping parameters.")
            self.last_reopt = now.isoformat()
            self._save()
            return False
        self.log(f"Live avg monthly {'n/a' if avg is None else f'{avg:.2%}'} < target "
                 f"{target:.0%}: re-optimising...")
        si = SelfImprover(data, self.cfg, state_dir=self.dir, log=self.log)
        new_p, report = si.run(max_iters=self.reopt_iters, patience=self.reopt_iters // 2)
        changed = new_p != self.params
        self.params = new_p
        self.last_reopt = now.isoformat()
        self._save()
        self.log("Parameters updated." if changed else "No better parameters found.")
        return changed

    def status(self):
        prices = self.fetch_prices(list(self.pf.positions)) if self.pf.positions else {}
        eq = self.pf.equity(prices)
        return {
            "equity": eq, "cash": self.pf.cash,
            "return_pct": 100 * (eq / self.cfg.initial_capital - 1),
            "live_avg_monthly": self.live_monthly_avg(),
            "positions": {s: {"shares": p.shares, "entry": p.entry_price, "stop": round(p.stop, 2),
                              "last": prices.get(s)} for s, p in self.pf.positions.items()},
            "closed_trades": sum(1 for t in self.pf.trades if t["side"] == "SELL"),
            "params": self.params.to_dict(),
        }
