# writes the results table into README.md between the table markers,
# from the outputs of run_backtest.py

import re
from pathlib import Path

import numpy as np
import pandas as pd

root = Path(__file__).resolve().parent.parent

# (column in backtest_returns.csv, strategy label, window label)
ROWS = [
    ("sample_504", "sample min-var", "504d"),
    ("clipped_504", "clipped min-var", "504d"),
    ("lw_504", "Ledoit-Wolf min-var", "504d"),
    ("rie_504", "RIE min-var", "504d"),
    ("rie_1008", "RIE min-var", "1008d"),
    ("one_over_n_252", "equal weight (1/N)", "—"),
    ("index_proxy", "buy & hold (index proxy)", "—"),
]
# rows that compete for the bolded best vol and turnover
MIN_VAR = [col for col, _, _ in ROWS if col not in ("one_over_n_252", "index_proxy")]

returns = pd.read_csv(root / "data/backtest_returns.csv", index_col=0, parse_dates=True)
stats = pd.read_csv(root / "data/backtest_stats.csv")
rf = pd.read_csv(root / "data/rf.csv", index_col=0, parse_dates=True)["rf_daily"]

cols = [col for col, _, _ in ROWS]
vol = returns[cols].std() * np.sqrt(252)
excess = returns[cols].sub(rf.reindex(returns.index), axis=0)
sharpe = excess.mean() / excess.std() * np.sqrt(252)
# buy and hold never rebalances, so it has no turnover rows
turnover = stats.groupby("strategy")["turnover"].mean()

best_vol = vol[MIN_VAR].idxmin()
best_turnover = turnover[MIN_VAR].idxmin()


def bold(text, on):
    return f"**{text}**" if on else text


lines = [
    "| strategy | window | ann. vol | Sharpe | mean monthly turnover |",
    "|---|---|---|---|---|",
]
for col, label, window in ROWS:
    t = f"{turnover[col]:.2f}" if col in turnover else "—"
    lines.append(
        f"| {label} | {window} "
        f"| {bold(f'{vol[col]:.1%}', col == best_vol)} "
        f"| {sharpe[col]:.2f} "
        f"| {bold(t, col == best_turnover)} |"
    )
table = "\n".join(lines)

readme_path = root / "README.md"
readme = readme_path.read_text()
pattern = r"(<!-- table:start -->\n).*?\n?(<!-- table:end -->)"
if not re.search(pattern, readme, flags=re.S):
    raise SystemExit("README.md is missing the <!-- table:start --> / <!-- table:end --> markers")
readme = re.sub(pattern, lambda m: m.group(1) + table + "\n" + m.group(2), readme, flags=re.S)
readme_path.write_text(readme)
print(table)
print("updated README.md")
