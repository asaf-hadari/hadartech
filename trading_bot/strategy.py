"""Trend-following strategy: signals are computed on bar t and executed on bar t+1.

Entry  : fast EMA > slow EMA, close > slow EMA, ADX >= adx_min, +DI > -DI,
         positive momentum, and enough market breadth.
Ranking: candidates are ranked by risk-adjusted momentum.
Exit   : close below an ATR trailing stop, or fast EMA crosses below slow EMA.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd

from . import indicators as ind


@dataclass
class Signals:
    entry: pd.DataFrame      # bool
    trend_exit: pd.DataFrame  # bool
    score: pd.DataFrame
    atr: pd.DataFrame
    close: pd.DataFrame


def compute_signals(data, p):
    c, h, l = data.close, data.high, data.low
    fast, slow = ind.ema(c, p.fast_ema), ind.ema(c, max(p.slow_ema, p.fast_ema + 5))
    adx, pdi, mdi = ind.adx(h, l, c, p.adx_period)
    atr = ind.atr(h, l, c, p.atr_period)
    score = ind.momentum_score(c, p.momentum_lookback)

    above = (c > slow)
    breadth = above.sum(axis=1) / slow.notna().sum(axis=1).replace(0, np.nan)
    market_ok = (breadth >= p.breadth_min)

    entry = (fast > slow) & above & (adx >= p.adx_min) & (pdi > mdi) & (score > 0)
    entry = entry & market_ok.values[:, None]
    trend_exit = fast < slow
    return Signals(entry.fillna(False), trend_exit.fillna(False), score, atr, c)
