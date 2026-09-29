# monthly rebalanced min variance backtest, every estimator x window length.
# covariance is always estimated on the T days before the rebalance date so there is no lookahead

import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

# one BLAS thread per worker, the process pool provides the parallelism
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from rmt.data import RANK_START, RANK_END
from rmt.estimators import sample_cov, clipped_cov, lw_cov, rie_cov

BACKTEST_START = "2005-02-01"


def long_only_min_var(cov, w_start=None):
    S = cov.values
    N = len(S)
    w0 = np.full(N, 1.0 / N) if w_start is None else np.clip(w_start, 0, 1)
    res = minimize(
        lambda w: w @ S @ w,
        w0,
        jac=lambda w: 2 * S @ w,
        method="SLSQP",
        bounds=[(0, 1)] * N,
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
        # default ftol is too loose for daily variances (~1e-4), stops ~15% above the optimum
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    if not res.success:
        raise RuntimeError(res.message)
    return pd.Series(res.x, index=cov.index)


def run_backtest(
    returns, estimator, T, start=BACKTEST_START, long_only=False, equal_weight=False
):
    idx = returns.index
    rebs = idx[idx >= start]
    rebs = rebs[~rebs.to_period("M").duplicated()]
    positions = [idx.get_loc(d) for d in rebs]
    assert positions[0] - T >= 0, "not enough history before the first rebalance"

    daily = []
    stats = []
    w_prev = None
    for i, p in enumerate(positions):
        window = returns.iloc[p - T : p]
        cov = estimator(window)
        if equal_weight:
            w = pd.Series(1.0 / len(cov), index=cov.index)
        elif long_only:
            w = long_only_min_var(cov, None if w_prev is None else w_prev.values)
        else:
            # closed form
            w = np.linalg.solve(cov.values, np.ones(len(cov)))
            w = pd.Series(w / w.sum(), index=cov.index)

        turnover = np.nan if w_prev is None else 0.5 * (w - w_prev).abs().sum()
        predicted_vol = np.sqrt(max(w @ cov.values @ w, 0) * 252)
        stats.append((idx[p], turnover, predicted_vol))

        end = positions[i + 1] if i + 1 < len(positions) else len(idx)
        hold = returns.iloc[p:end]

        # Buy and hold between monthly rebalances
        growth = (1 + hold).cumprod()
        value = growth.mul(w, axis=1).sum(axis=1)
        period_returns = value.pct_change()
        period_returns.iloc[0] = value.iloc[0] - 1
        daily.append(period_returns)

        # Drifted weights immediately before the next rebalance
        w_prev = w * growth.iloc[-1] / value.iloc[-1]

    daily = pd.concat(daily)
    stats = pd.DataFrame(stats, columns=["date", "turnover", "predicted_vol"])
    return daily, stats.set_index("date")


ESTIMATORS = {
    "sample": sample_cov,
    "clipped": clipped_cov,
    "clipped_raw": lambda w: clipped_cov(w, sigma2=1),
    "lw": lw_cov,
    "rie": rie_cov,
    # weights are forced equal, the cov is only used for the predicted vol
    "one_over_n": sample_cov,
}
WINDOWS = [252, 504, 1008]

returns = pd.read_csv(root / "data/returns.csv.gz", index_col=0, parse_dates=True)
rf = pd.read_csv(root / "data/rf.csv", index_col=0, parse_dates=True)["rf_daily"]


def summary(col, daily):
    excess = daily - rf.reindex(daily.index)
    return (
        f"{col:19s} ann vol = {daily.std() * np.sqrt(252):6.1%}  "
        f"sharpe = {excess.mean() / excess.std() * np.sqrt(252):5.2f}  "
    )


def run_config(key):
    name, T, long_only = key
    daily, stats = run_backtest(
        returns,
        ESTIMATORS[name],
        T,
        long_only=long_only,
        equal_weight=name == "one_over_n",
    )
    col = f"{name}_{T}" + ("_lo" if long_only else "")
    stats["strategy"] = col
    line = summary(col, daily) + (
        f"mean turnover = {stats['turnover'].mean():5.2f}  "
        f"mean predicted vol = {stats['predicted_vol'].mean():6.1%}"
    )
    return col, daily, stats, line


if __name__ == "__main__":
    keys = [
        (name, T, long_only)
        for long_only in [False, True]
        for T in WINDOWS
        for name in ESTIMATORS
        # 1/N is already long only, and its weights don't depend on T
        if not (name == "one_over_n" and (long_only or T != WINDOWS[0]))
    ]
    all_returns = {}
    all_stats = []
    # each config is independent, run them across cores
    with ProcessPoolExecutor() as ex:
        for col, daily, stats, line in ex.map(run_config, keys):
            print(line, flush=True)
            all_returns[col] = daily
            all_stats.append(stats)

    # buy and hold, starting weights proportional to 2001-2004 dollar volume
    close = pd.read_csv(root / "data/close.csv.gz", index_col=0, parse_dates=True)
    volume = pd.read_csv(root / "data/volume.csv.gz", index_col=0, parse_dates=True)
    w0 = (
        (close[returns.columns] * volume[returns.columns])
        .loc[RANK_START:RANK_END]
        .mean()
    )
    hold = returns.loc[returns.index >= BACKTEST_START]
    value = (1 + hold).cumprod() @ (w0 / w0.sum())
    daily = value.pct_change()
    daily.iloc[0] = value.iloc[0] - 1
    all_returns["index_proxy"] = daily
    print(summary("index_proxy", daily) + "(buy and hold, no rebalances)")

    pd.DataFrame(all_returns).to_csv(root / "data/backtest_returns.csv")
    pd.concat(all_stats).to_csv(root / "data/backtest_stats.csv")
    print("wrote data/backtest_returns.csv and data/backtest_stats.csv")
