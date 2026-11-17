"""Stock universes: NASDAQ-100, the full NASDAQ listing, and Tel Aviv TA-35.

The static lists are a snapshot and index membership changes over time;
edit them or pass --symbols-file to use your own list.
"""
from pathlib import Path
import io

NASDAQ_100 = """
AAPL MSFT NVDA AMZN META GOOGL GOOG AVGO TSLA COST NFLX TMUS ASML CSCO AMD PEP
LIN ADBE ISRG INTU QCOM TXN AMGN BKNG AMAT CMCSA HON PANW ADP VRTX GILD ADI MU
SBUX LRCX MELI KLAC INTC MDLZ CTAS PYPL REGN SNPS CDNS CRWD MAR ORLY CEG FTNT
MRVL CSX DASH ADSK ABNB ROP PDD WDAY NXPI CHTR PCAR TTD MNST AEP PAYX FANG KDP
CPRT ODFL ROST FAST KHC DDOG MCHP EA VRSK CTSH EXC XEL GEHC LULU IDXX AZN CCEP
BKR TEAM CSGP ZS ANSS ON DXCM TTWO CDW WBD BIIB GFS MDB ILMN ARM APP PLTR
MSTR AXON SHOP
""".split()

# Tel Aviv 35 (Yahoo Finance tickers use the .TA suffix).
TA_35 = """
LUMI POLI DSCT MZTF FIBI TEVA NICE ESLT ICL AZRG BEZQ ORA TSEM NVMI CAMT ENLT
PHOE HARL CLIS MGDL NWMD DLEKG ALHE MLSR BIG AMOT SPEN STRS SAE ORL OPCE ENOG
NXSN ARPT DIMRI
""".split()
TA_35 = [f"{s}.TA" for s in TA_35]

BENCHMARKS = {"nasdaq100": "QQQ", "nasdaq_all": "QQQ", "ta35": "TA35.TA"}

NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"


def fetch_all_nasdaq():
    """Download every common stock listed on NASDAQ (no ETFs or test issues)."""
    import pandas as pd
    import requests

    text = requests.get(NASDAQ_LISTED_URL, timeout=30).text
    df = pd.read_csv(io.StringIO(text), sep="|")
    df = df[df["Symbol"].notna() & ~df["Symbol"].str.startswith("File Creation")]
    df = df[(df["ETF"] == "N") & (df["Test Issue"] == "N")]
    df = df[df["Financial Status"].isin(["N"])]  # normal status only
    syms = df["Symbol"].astype(str)
    syms = syms[~syms.str.contains(r"[\.\$]")]
    return sorted(syms.tolist())


def get_universe(market, symbols_file=None):
    if symbols_file:
        return [s.strip().upper() for s in Path(symbols_file).read_text().split() if s.strip()]
    market = market.lower()
    if market == "nasdaq100":
        return list(dict.fromkeys(NASDAQ_100))
    if market == "nasdaq_all":
        return fetch_all_nasdaq()
    if market in ("ta35", "ta30", "tase"):
        return list(TA_35)
    raise ValueError(f"Unknown market '{market}'. Use nasdaq100 | nasdaq_all | ta35")
