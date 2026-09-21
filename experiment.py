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


def plot_performance(portvals, trades, title, path):
    """Plot normalized portfolio values and mark the learner's long/short entries."""
    colors = {"Benchmark": "tab:purple", "Manual Strategy": "tab:red", "Strategy Learner": "tab:green"}
    fig, ax = plt.subplots(figsize=(12, 6))
    for name, values in portvals.items():
        normalized = values["total_value"] / values["total_value"].iloc[0]
        ax.plot(normalized.index, normalized, label=name, color=colors.get(name))

    holdings = trades["Manual Strategy"]["Position"].cumsum()
    previous = holdings.shift(1).fillna(0)
    for date in holdings.index[(holdings == 1000) & (previous != 1000)]:
        ax.axvline(date, color="tab:blue", linestyle="--", linewidth=0.8, alpha=0.6)
    for date in holdings.index[(holdings == -1000) & (previous != -1000)]:
        ax.axvline(date, color="black", linestyle="--", linewidth=0.8, alpha=0.6)

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Normalized Portfolio Value")
    ax.grid(True, alpha=0.3)
    ax.legend(title="Manual entries: blue = long, black = short", title_fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
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
    plot_performance(portvals, trades, f"{symbol}: {label}", os.path.join(outdir, filename))
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

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.bar(results.index.astype(str), results["Trades"], color="tab:blue", alpha=0.6)
    ax1.set_xlabel("Market impact")
    ax1.set_ylabel("Number of trades", color="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(results.index.astype(str), results["Cumulative Return"], color="tab:red", marker="o")
    ax2.set_ylabel("Cumulative return", color="tab:red")
    ax1.set_title(f"{symbol}: Strategy Learner sensitivity to market impact")
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

    evaluate(args.symbol, IN_SAMPLE, learner, args.outdir, "In-sample")
    evaluate(args.symbol, OUT_OF_SAMPLE, learner, args.outdir, "Out-of-sample")
    impact_study(args.symbol, args.outdir)
    print(f"\nCharts saved to {args.outdir}/")


if __name__ == "__main__":
    main()
