"""Vectorised indicators on wide frames (dates x symbols)."""
import numpy as np
import pandas as pd


def ema(df, n):
    return df.ewm(span=n, adjust=False, min_periods=n).mean()


def wilder(df, n):
    return df.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def true_range(high, low, close):
    prev = close.shift(1)
    tr = np.fmax(np.fmax(high - low, (high - prev).abs()), (low - prev).abs())
    return pd.DataFrame(tr, index=close.index, columns=close.columns)


def atr(high, low, close, n=14):
    return wilder(true_range(high, low, close), n)


def adx(high, low, close, n=14):
    up = high.diff()
    down = -low.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    tr = wilder(true_range(high, low, close), n)
    plus_di = 100 * wilder(plus_dm, n) / tr
    minus_di = 100 * wilder(minus_dm, n) / tr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return wilder(dx, n), plus_di, minus_di


def momentum_score(close, lookback):
    """Return over the lookback divided by its volatility (risk-adjusted momentum)."""
    ret = close / close.shift(lookback) - 1
    vol = close.pct_change().rolling(lookback).std() * np.sqrt(lookback)
    return ret / vol.replace(0, np.nan)
