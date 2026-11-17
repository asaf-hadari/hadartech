"""Market data: Yahoo Finance (real) and a synthetic generator (offline testing only)."""
from pathlib import Path
import hashlib
import time
import numpy as np
import pandas as pd

FIELDS = ["Open", "High", "Low", "Close", "Volume"]


class MarketData:
    """Wide frames, index = dates, columns = symbols."""

    def __init__(self, frames):
        self.open = frames["Open"]
        self.high = frames["High"]
        self.low = frames["Low"]
        self.close = frames["Close"]
        self.volume = frames["Volume"]

    @property
    def symbols(self):
        return list(self.close.columns)

    @property
    def dates(self):
        return self.close.index

    def slice(self, start=None, end=None):
        f = {k: getattr(self, k.lower()).loc[start:end] for k in FIELDS}
        return MarketData(f)

    def select(self, symbols):
        return MarketData({k: getattr(self, k.lower())[symbols] for k in FIELDS})

    def liquid(self, min_price=5.0, min_dollar_volume=5e6, lookback=60):
        """Drop penny stocks and illiquid names (important for nasdaq_all)."""
        c = self.close.tail(lookback)
        dv = (c * self.volume.tail(lookback)).median()
        keep = [s for s in self.symbols
                if c[s].iloc[-1] >= min_price and dv[s] >= min_dollar_volume]
        return self.select(keep)

    def save(self, path):
        pd.concat({k: getattr(self, k.lower()) for k in FIELDS}, axis=1).to_pickle(path)

    @classmethod
    def load(cls, path):
        df = pd.read_pickle(path)
        return cls({k: df[k] for k in FIELDS})


def _clean(frames, min_history=250):
    close = frames["Close"]
    good = close.columns[close.notna().sum() >= min_history]
    out = {k: frames[k][good] for k in FIELDS}
    out["Close"] = out["Close"].ffill()
    for k in ("Open", "High", "Low"):
        out[k] = out[k].fillna(out["Close"])
    out["Volume"] = out["Volume"].fillna(0)
    return out


def download_yahoo(symbols, period="5y", interval="1d", cache_dir="bot_state/cache",
                   max_age_hours=6, batch=200):
    """Download daily bars for many symbols via yfinance, with a disk cache."""
    import yfinance as yf

    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1(",".join(sorted(symbols)).encode()).hexdigest()[:12]
    key = cache / f"{digest}_{period}_{interval}.pkl"
    if key.exists() and (time.time() - key.stat().st_mtime) < max_age_hours * 3600:
        return MarketData.load(key)

    parts = []
    for i in range(0, len(symbols), batch):
        chunk = symbols[i:i + batch]
        df = yf.download(chunk, period=period, interval=interval, auto_adjust=True,
                         group_by="column", progress=False, threads=True)
        if df.empty:
            continue
        if not isinstance(df.columns, pd.MultiIndex):  # single symbol
            df.columns = pd.MultiIndex.from_product([df.columns, chunk])
        parts.append(df)
    if not parts:
        raise RuntimeError("No data downloaded - check your internet connection / symbols")
    df = pd.concat(parts, axis=1)
    frames = {k: df[k] for k in FIELDS}
    frames["Close"].index = pd.to_datetime(frames["Close"].index).tz_localize(None)
    for k in FIELDS:
        frames[k].index = frames["Close"].index
    md = MarketData(_clean(frames))
    md.save(key)
    return md


def live_prices(symbols):
    """Latest traded price per symbol (Yahoo, ~real time for US, may be delayed)."""
    import yfinance as yf

    prices = {}
    tickers = yf.Tickers(" ".join(symbols))
    for s in symbols:
        try:
            p = tickers.tickers[s].fast_info["last_price"]
            if p and np.isfinite(p):
                prices[s] = float(p)
        except Exception:
            pass
    return prices


def synthetic(n_symbols=40, n_days=1500, seed=7, start="2019-01-01"):
    """Random-walk prices with persistent trend regimes. ONLY for offline tests -
    results on this data say nothing about real markets."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_days)
    syms = [f"SYN{i:03d}" for i in range(n_symbols)]
    market = rng.normal(0.0003, 0.01, n_days)
    closes = np.empty((n_days, n_symbols))
    for j in range(n_symbols):
        drift = np.zeros(n_days)
        d, i = 0.0, 0
        while i < n_days:
            length = rng.integers(40, 200)
            d = rng.normal(0.0004, 0.0015)
            drift[i:i + length] = d
            i += length
        r = drift + 0.8 * market + rng.normal(0, 0.015, n_days)
        closes[:, j] = 50 * np.exp(np.cumsum(r))
    close = pd.DataFrame(closes, index=dates, columns=syms)
    gap = rng.normal(0, 0.004, close.shape)
    open_ = close.shift(1).bfill() * (1 + gap)
    spread = np.abs(rng.normal(0, 0.008, close.shape))
    high = np.maximum(open_, close) * (1 + spread)
    low = np.minimum(open_, close) * (1 - spread)
    vol = pd.DataFrame(rng.integers(1e5, 5e6, close.shape), index=dates, columns=syms)
    return MarketData({"Open": open_, "High": high, "Low": low, "Close": close,
                       "Volume": vol.astype(float)})
