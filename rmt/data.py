"""
Data pipeline for yahoo finance data.

Collects
"""

import datetime
import json
from pathlib import Path
import pandas as pd
import yfinance as yf

DOWNLOAD_START = "2000-9-01"
WINDOW_START = "2001-01-03"  # first trading day of 2001
RANK_START, RANK_END = "2001-01-01", "2004-12-31"
N_TARGET = 300
FFILL_LIMIT = 5  # maximum acceptable days to forward fill
COVERAGE_MIN = 0.995  # minimum acceptable coverage over the window


DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def download_panels(tickers, start):
    """Download Adj and Raw Close and Volume."""
    raw = yf.download(tickers, start=start, auto_adjust=False)
    if raw is None or raw.empty:
        raise RuntimeError("Yahoo Data Download Failed")
    adj_close = raw["Adj Close"]
    close = raw["Close"]
    volume = raw["Volume"]
    return adj_close, close, volume


def download_risk_free(start):
    """13 week t-bill yield for later reference"""
    irx = download_panels(["^IRX"], start)[1]["^IRX"]
    return (irx.ffill() / 100 / 252).rename("rf_daily")


def select_tickers(adj_close, close, volume):
    """Filter the universe of stocks to N_TARGET tickers with highest liquidity"""
    counts = {}
    in_window = adj_close.index >= WINDOW_START

    tickers = [t for t in adj_close.columns if adj_close[t].notna().any()]
    counts["has_data"] = len(tickers)

    tickers = [t for t in tickers if adj_close.loc[:WINDOW_START, t].notna().any()]
    counts["listed_by_window_start"] = len(tickers)

    coverage = adj_close.loc[in_window, tickers].notna().mean()
    tickers = [t for t in tickers if coverage[t] >= COVERAGE_MIN]
    counts["coverage"] = len(tickers)

    # fills using ffill, then drops incomplete tickers
    filled = adj_close[tickers].ffill(limit=FFILL_LIMIT)
    complete = filled.loc[in_window].notna().all()
    tickers = [t for t in tickers if complete[t]]
    counts["complete_after_ffill"] = len(tickers)

    # sorts by dollar volume over window
    dollar_volume = close[tickers] * volume[tickers].loc[RANK_START:RANK_END].mean()
    ranked = dollar_volume.sort_values(ascending=False).index.tolist()

    selected = sorted(ranked[:N_TARGET])
    counts["selected"] = len(selected)
    return selected, counts


def compute_returns(adj_close, tickers):
    prices = adj_close[tickers].ffill(limit=FFILL_LIMIT).loc[WINDOW_START:]
    returns = prices.pct_change().iloc[1:]  # first row is NaN's
    assert not returns.isna().any().any(), "Returns Matrix is broken/has gaps"
    return returns
