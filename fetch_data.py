"""Download daily adjusted close prices from Yahoo Finance into the data directory."""

import argparse
import os

import yfinance as yf

from util import data_dir


def fetch(symbols, start, end, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for symbol in symbols:
        df = yf.download(symbol, start=start, end=end, auto_adjust=False, progress=False, multi_level_index=False)
        if df.empty:
            raise RuntimeError(f"No data returned for {symbol}")
        df.index.name = "Date"
        path = os.path.join(out_dir, f"{symbol}.csv")
        df[["Open", "High", "Low", "Close", "Adj Close", "Volume"]].to_csv(path)
        print(f"Wrote {len(df)} rows to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("symbols", nargs="*", default=["SPY", "JPM"])
    parser.add_argument("--start", default="2007-01-01")
    parser.add_argument("--end", default="2012-12-31")
    parser.add_argument("--out", default=data_dir())
    args = parser.parse_args()
    fetch(args.symbols, args.start, args.end, args.out)
