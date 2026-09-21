"""
Backtest the rule-based and learned strategies against a buy-and-hold benchmark.

Trains StrategyLearner on the in-sample period, evaluates all strategies on both
the in-sample and out-of-sample periods, prints performance statistics, and saves
comparison charts to the output directory.
"""

import argparse
import datetime as dt
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import pandas as pd

from ManualStrategy import ManualStrategy
from marketsimcode import calculate_portfolio_stats, compute_portvals_from_trades
from StrategyLearner import StrategyLearner

IN_SAMPLE = (dt.datetime(2008, 1, 1), dt.datetime(2009, 12, 31))
OUT_OF_SAMPLE = (dt.datetime(2010, 1, 1), dt.datetime(2011, 12, 31))
START_VALUE = 100000
COMMISSION = 9.95
IMPACT = 0.005


def benchmark_trades(index, shares=1000):
    """Buy and hold: buy `shares` on the first day and hold to the end."""
    trades = pd.DataFrame(0.0, index=index, columns=["Position"])
    trades.iloc[0, 0] = shares
    return trades


STYLE = {
    "Benchmark": {"color": "#8a8f98", "linewidth": 1.6, "linestyle": "--"},
    "Manual Strategy": {"color": "#e8833a", "linewidth": 1.6},
    "Strategy Learner": {"color": "#2f6fdb", "linewidth": 2.4},
}


def style_axes(ax):
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.grid(True, axis="y", alpha=0.25)
    ax.tick_params(labelsize=9)


def plot_performance(portvals, trades, title, path):
    """Plot normalized portfolio values, marking the learner's long/short entries."""
    fig, ax = plt.subplots(figsize=(11, 5.5))
    normalized = {name: v["total_value"] / v["total_value"].iloc[0] for name, v in portvals.items()}

    for name, series in normalized.items():
        ret = series.iloc[-1] - 1
        ax.plot(series.index, series, label=f"{name} ({ret:+.1%})", **STYLE[name])

    learner, benchmark = normalized["Strategy Learner"], normalized["Benchmark"]
    ax.fill_between(learner.index, learner, benchmark, where=learner >= benchmark,
                    color=STYLE["Strategy Learner"]["color"], alpha=0.08, interpolate=True)

    holdings = trades["Strategy Learner"]["Position"].cumsum()
    previous = holdings.shift(1).fillna(0)
    longs = holdings.index[(holdings == 1000) & (previous != 1000)]
    shorts = holdings.index[(holdings == -1000) & (previous != -1000)]
    ax.scatter(longs, learner[longs], marker="^", s=70, color="#1f9d55", zorder=5, label="Learner long entry")
    ax.scatter(shorts, learner[shorts], marker="v", s=70, color="#d64545", zorder=5, label="Learner short entry")

    ax.axhline(1.0, color="black", linewidth=0.6, alpha=0.4)
    ax.set_title(title, fontsize=13, fontweight="bold", loc="left")
    ax.set_ylabel("Portfolio value (normalized)")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_summary(results, symbol, path):
    """Grouped bar chart of cumulative return per strategy for each period."""
    periods = list(results)
    names = list(STYLE)
    width = 0.26
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, name in enumerate(names):
        values = [results[p].loc[name, "Cumulative Return"] for p in periods]
        xs = [j + (i - 1) * width for j in range(len(periods))]
        bars = ax.bar(xs, values, width, label=name, color=STYLE[name]["color"])
        ax.bar_label(bars, labels=[f"{v:+.1%}" for v in values], fontsize=8, padding=2)
    ax.set_xticks(range(len(periods)), periods)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title(f"{symbol}: cumulative return vs. buy-and-hold", fontsize=13, fontweight="bold", loc="left")
    ax.legend(frameon=False, fontsize=9)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def evaluate(symbol, period, learner, outdir, label):
    sd, ed = period
    trades = {
        "Manual Strategy": ManualStrategy(impact=IMPACT, commission=COMMISSION).testPolicy(symbol, sd, ed, START_VALUE),
        "Strategy Learner": learner.testPolicy(symbol, sd, ed, START_VALUE),
    }
    trades = {"Benchmark": benchmark_trades(trades["Manual Strategy"].index), **trades}
    portvals = {
        name: compute_portvals_from_trades(t, symbol, START_VALUE, COMMISSION, IMPACT)
        for name, t in trades.items()
    }

    stats = pd.DataFrame([calculate_portfolio_stats(v, name) for name, v in portvals.items()]).set_index("Portfolio")
    stats["Trades"] = [int((t["Position"] != 0).sum()) for t in trades.values()]
    print(f"\n{label} ({sd.date()} to {ed.date()})")
    print(stats.to_string(float_format=lambda x: f"{x:.4f}"))

    filename = label.lower().replace(" ", "_").replace("-", "_") + ".png"
    title = f"{symbol}: {label} ({sd.year}–{ed.year})"
    plot_performance(portvals, trades, title, os.path.join(outdir, filename))
    return stats


def impact_study(symbol, outdir):
    """Show how market impact changes the learner's trading frequency and returns (in-sample)."""
    sd, ed = IN_SAMPLE
    rows = []
    for impact in [0.0, 0.005, 0.01, 0.02, 0.04]:
        learner = StrategyLearner(impact=impact, commission=0.0)
        learner.add_evidence(symbol, sd, ed, START_VALUE)
        trades = learner.testPolicy(symbol, sd, ed, START_VALUE)
        portvals = compute_portvals_from_trades(trades, symbol, START_VALUE, 0.0, impact)
        stats = calculate_portfolio_stats(portvals)
        rows.append({"Impact": impact, "Trades": int((trades["Position"] != 0).sum()),
                     "Cumulative Return": stats["Cumulative Return"]})
    results = pd.DataFrame(rows).set_index("Impact")
    print("\nMarket impact study (in-sample, no commission)")
    print(results.to_string(float_format=lambda x: f"{x:.4f}"))

    labels = [f"{x:.1%}" for x in results.index]
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.bar(labels, results["Trades"], color="#2f6fdb", alpha=0.35, label="Trades")
    ax1.set_xlabel("Market impact per trade")
    ax1.set_ylabel("Number of trades")
    ax2 = ax1.twinx()
    ax2.plot(labels, results["Cumulative Return"], color="#2f6fdb", marker="o", linewidth=2.2, label="Cumulative return")
    ax2.set_ylabel("Cumulative return")
    ax2.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax2.spines["top"].set_visible(False)
    ax1.set_title(f"{symbol}: Strategy Learner sensitivity to market impact", fontsize=13, fontweight="bold", loc="left")
    style_axes(ax1)
    fig.legend(frameon=False, fontsize=9, loc="upper right", bbox_to_anchor=(0.88, 0.88))
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "impact_study.png"), dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="JPM")
    parser.add_argument("--outdir", default="images")
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    learner = StrategyLearner(impact=IMPACT, commission=COMMISSION)
    learner.add_evidence(args.symbol, *IN_SAMPLE, START_VALUE)

    results = {
        "In-sample": evaluate(args.symbol, IN_SAMPLE, learner, args.outdir, "In-sample"),
        "Out-of-sample": evaluate(args.symbol, OUT_OF_SAMPLE, learner, args.outdir, "Out-of-sample"),
    }
    plot_summary(results, args.symbol, os.path.join(args.outdir, "summary.png"))
    impact_study(args.symbol, args.outdir)
    print(f"\nCharts saved to {args.outdir}/")


if __name__ == "__main__":
    main()
