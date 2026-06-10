"""
Data pipeline for yahoo finance data.

Collects daily prices for the current S&P 500 members, keeps the 300
most liquid (ranked on 2001-2004 dollar volume, unadjusted close so future
splits don't leak in), saves daily returns and the t-bill rate to data.
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
    dollar_volume = (close[tickers] * volume[tickers]).loc[RANK_START:RANK_END].mean()
    ranked = dollar_volume.sort_values(ascending=False).index.tolist()

    selected = sorted(ranked[:N_TARGET])
    counts["selected"] = len(selected)
    return selected, counts


def compute_returns(adj_close, tickers):
    prices = adj_close[tickers].ffill(limit=FFILL_LIMIT).loc[WINDOW_START:]
    returns = prices.pct_change().iloc[1:]  # first row is NaN's
    assert not returns.isna().any().any(), "Returns Matrix is broken/has gaps"
    return returns


def run_pipeline(universe_path, data_dir=DATA_DIR):
    data_dir.mkdir(exist_ok=True)
    universe = pd.read_csv(universe_path)

    spy_close = download_panels(["SPY"], DOWNLOAD_START)[1]["SPY"]
    calendar = spy_close.dropna().index

    adj_close, close, volume = download_panels(
        universe["symbol"].tolist(), DOWNLOAD_START
    )

    adj_close = adj_close.reindex(calendar)
    close = close.reindex(calendar)
    volume = volume.reindex(calendar)
    rf = download_risk_free(DOWNLOAD_START).reindex(calendar).ffill()

    tickers, counts = select_tickers(adj_close, close, volume)
    filter_counts = {"universe": len(universe)}
    filter_counts.update(counts)
    returns = compute_returns(adj_close, tickers)

    adj_close.to_csv(data_dir / "adj_close.csv.gz")
    close.to_csv(data_dir / "close.csv.gz")
    volume.to_csv(data_dir / "volume.csv.gz")
    returns.to_csv(data_dir / "returns.csv.gz")
    rf.reindex(returns.index).to_csv(data_dir / "rf.csv")

    info = {
        "date_downloaded": str(datetime.date.today()),
        "filter_counts": filter_counts,
        "n_stocks": int(returns.shape[1]),
        "n_days": int(returns.shape[0]),
        "first_day": str(returns.index[0])[:10],
        "last_day": str(returns.index[-1])[:10],
        "returns_over_50pct": int((returns.abs() > 0.5).sum().sum()),
    }
    with open(data_dir / "info.json", "w") as f:
        json.dump(info, f, indent=2)
    print(info)


if __name__ == "__main__":
    run_pipeline(Path(__file__).resolve().parent.parent / "universe_sp500.csv")
