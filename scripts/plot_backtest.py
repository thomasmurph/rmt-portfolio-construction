from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

root = Path(__file__).resolve().parent.parent

ESTIMATORS = {
    "sample": "sample",
    "clipped": "clipped",
    "lw": "Ledoit-Wolf",
    "rie": "RIE",
}
WINDOWS = [252, 504, 1008]

returns = pd.read_csv(root / "data/backtest_returns.csv", index_col=0, parse_dates=True)
vol = returns.std() * np.sqrt(252) * 100

fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True, layout="constrained")
width = 0.2
for ax, suffix, title in [
    (axes[0], "", "unconstrained"),
    (axes[1], "_lo", "long-only"),
]:
    ax.set_title(title)
    for j, (est, label) in enumerate(ESTIMATORS.items()):
        x = np.arange(3) + (j - 1.5) * width
        ax.bar(x, [vol[f"{est}_{T}{suffix}"] for T in WINDOWS], width, label=label)
    ax.axhline(vol["one_over_n_252"], color="gray", ls="--", label="equal weight")
    ax.axhline(vol["index_proxy"], color="gray", ls=":", label="buy & hold")
    ax.set_xticks(range(3), [f"{T} days" for T in WINDOWS])
axes[0].set_ylim(0, 23)  # the sample_252 bar runs off the top
axes[0].text(
    -0.3,
    11,
    f"{vol['sample_252']:,.0f}%",
    rotation=90,
    ha="center",
    va="center",
    color="w",
)
axes[0].set_ylabel("realized vol (%, annualized)")
fig.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center", ncol=6)
(root / "figs").mkdir(exist_ok=True)
fig.savefig(root / "figs/backtest_vol.png", dpi=200)
print("wrote figs/backtest_vol.png")

# trailing vol through time
show = {
    "rie_1008": "RIE min-var (4y)",
    "sample_504": "sample min-var (2y)",
    "one_over_n_252": "equal weight",
    "index_proxy": "buy & hold",
}
rolling = returns[list(show)].rolling(126).std() * np.sqrt(252) * 100
ax = rolling.rename(columns=show).plot(
    figsize=(9, 4), ylabel="trailing 126-day vol (%, annualized)"
)
ax.figure.tight_layout()
ax.figure.savefig(root / "figs/rolling_vol.png", dpi=200)
print("wrote figs/rolling_vol.png")
