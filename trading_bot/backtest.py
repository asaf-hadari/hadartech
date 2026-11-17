"""Event-driven daily backtester (signals at close t, fills at open t+1)."""
from dataclasses import dataclass
import math
import numpy as np
import pandas as pd

from .config import BrokerCosts
from .portfolio import Portfolio, plan_orders
from .strategy import compute_signals


@dataclass
class BacktestResult:
    equity: pd.Series
    trades: pd.DataFrame
    metrics: dict


def build_row(sig_np, t, symbols, held):
    entry_idx = np.flatnonzero(sig_np["entry"][t])
    need = set(entry_idx.tolist()) | {symbols.index(s) for s in held}
    return {
        "entry": {symbols[j]: True for j in entry_idx},
        "trend_exit": {symbols[j]: bool(sig_np["trend_exit"][t, j]) for j in need},
        "score": {symbols[j]: float(sig_np["score"][t, j]) for j in need},
        "atr": {symbols[j]: float(sig_np["atr"][t, j]) for j in need},
        "close": {symbols[j]: float(sig_np["close"][t, j]) for j in need},
    }


def run_backtest(data, params, initial_capital=100_000.0, costs=None, start=None):
    costs = costs or BrokerCosts()
    sig = compute_signals(data, params)
    symbols = data.symbols
    sig_np = {"entry": sig.entry.to_numpy(bool), "trend_exit": sig.trend_exit.to_numpy(bool),
              "score": sig.score.to_numpy(float), "atr": sig.atr.to_numpy(float),
              "close": data.close.to_numpy(float)}
    opens = data.open.to_numpy(float)
    closes = sig_np["close"]
    dates = data.dates
    col = {s: i for i, s in enumerate(symbols)}

    warmup = max(params.slow_ema, params.momentum_lookback, 2 * params.adx_period) + 1
    t0 = warmup if start is None else max(warmup, dates.searchsorted(pd.Timestamp(start)))
    pf = Portfolio(cash=initial_capital)
    pending_exits, pending_entries = [], []
    equity = np.full(len(dates), np.nan)

    for t in range(t0, len(dates)):
        d = dates[t].date()
        for s, reason in pending_exits:
            px = opens[t, col[s]]
            pf.sell(s, px if math.isfinite(px) else closes[t, col[s]], d, costs, reason)
        for s, shares, stop in pending_entries:
            px = opens[t, col[s]]
            if math.isfinite(px):
                pf.buy(s, px, shares, d, stop, costs)
        held_val = sum(p.shares * closes[t, col[s]] for s, p in pf.positions.items())
        equity[t] = pf.cash + held_val
        row = build_row(sig_np, t, symbols, pf.positions.keys())
        pending_exits, pending_entries = plan_orders(pf, row, params, equity[t])

    eq = pd.Series(equity, index=dates).dropna()
    trades = pd.DataFrame(pf.trades)
    return BacktestResult(eq, trades, compute_metrics(eq, trades))


def monthly_returns(equity):
    m = equity.resample("ME").last()
    first = equity.iloc[0] if len(equity) else np.nan
    return pd.concat([pd.Series([first], index=[m.index[0] - pd.offsets.MonthEnd(1)]), m]
                     ).pct_change().dropna() if len(m) else pd.Series(dtype=float)


def compute_metrics(equity, trades):
    if len(equity) < 2:
        return {}
    mr = monthly_returns(equity)
    daily = equity.pct_change().dropna()
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    total = equity.iloc[-1] / equity.iloc[0] - 1
    dd = equity / equity.cummax() - 1
    sells = trades[trades["side"] == "SELL"] if len(trades) else pd.DataFrame()
    return {
        "total_return": total,
        "cagr": (1 + total) ** (1 / years) - 1 if years > 0 and total > -1 else np.nan,
        "avg_monthly_return": mr.mean(),
        "median_monthly_return": mr.median(),
        "geo_monthly_return": (1 + total) ** (1 / max(len(mr), 1)) - 1,
        "worst_month": mr.min(),
        "best_month": mr.max(),
        "positive_months_pct": (mr > 0).mean(),
        "months": len(mr),
        "max_drawdown": dd.min(),
        "sharpe": daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else 0.0,
        "trades": len(sells),
        "win_rate": (sells["pnl"] > 0).mean() if len(sells) else np.nan,
        "final_equity": equity.iloc[-1],
    }


def format_metrics(m):
    pct = lambda x: f"{100 * x:6.2f}%"
    keys = [("Total return", "total_return", pct), ("CAGR", "cagr", pct),
            ("Avg monthly", "avg_monthly_return", pct),
            ("Geometric monthly", "geo_monthly_return", pct),
            ("Median monthly", "median_monthly_return", pct),
            ("Best / worst month", None, None),
            ("Positive months", "positive_months_pct", pct),
            ("Max drawdown", "max_drawdown", pct),
            ("Sharpe", "sharpe", lambda x: f"{x:6.2f}"),
            ("Closed trades", "trades", lambda x: f"{x:6d}"),
            ("Win rate", "win_rate", pct),
            ("Final equity", "final_equity", lambda x: f"{x:,.0f}")]
    lines = []
    for label, k, f in keys:
        if k is None:
            lines.append(f"  {label:<20} {pct(m['best_month'])} / {pct(m['worst_month'])}")
        elif k in m and m[k] is not None and not (isinstance(m[k], float) and math.isnan(m[k])):
            lines.append(f"  {label:<20} {f(m[k])}")
    return "\n".join(lines)
