# Python Trading Repo Starter

A clean, MIT-licensed starter repo for algorithmic **trading/backtesting** in Python. Includes a simple moving-average-crossover strategy, risk management, metrics (Sharpe, max drawdown, CAGR), and plots — ready to push to GitHub.

---

### 📁 Structure

```
trading-repo///
├─ README.md
├─ requirements.txt
├─ .gitignore
├─ LICENSE
├─ src/
│  ├─ main.py
│  ├─ data.py
│  ├─ strategy.py
│  ├─ backtester.py
│  └─ plotter.py
└─ examples/
   └─ run_example.sh
```

---

### 🚀 Quickstart

```bash
# 1) Create venv (optional but recommended)
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\\Scripts\\activate

# 2) Install deps
pip install -r requirements.txt

# 3) Run a backtest
python src/main.py --ticker AAPL --start 2018-01-01 --end 2025-08-01 \
  --fast 20 --slow 50 --risk 0.01 --initial 10000 --fee 0.0005
```

---

### 📦 requirements.txt

```txt
pandas>=2.0
numpy>=1.24
yfinance>=0.2
matplotlib>=3.7
```

---

## 🧰 .gitignore

```gitignore
# Environments
.venv/
__pycache__/
*.pyc

# OS
.DS_Store
Thumbs.db

# Notebooks/plots
*.png
*.csv
```

---

## 📜 LICENSE (MIT)

```text
MIT License

Copyright (c) 2025 YOUR_NAME

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 🧱 src/data.py

```python
from __future__ import annotations
import pandas as pd
import yfinance as yf


def load_ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download daily OHLCV from Yahoo Finance and return clean df."""
    data = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if data.empty:
        raise ValueError(f"No data for {ticker} between {start} and {end}.")
    data = data.rename(columns=str.lower)
    data.index.name = "date"
    return data[["open", "high", "low", "close", "volume"]]
```

---

## 🧠 src/strategy.py

```python
from __future__ import annotations
import pandas as pd

class MovingAverageCrossover:
    """Simple long-only SMA crossover with position sizing by risk per trade.

    Rules:
      - Go long when fast SMA crosses above slow SMA.
      - Exit when fast SMA crosses below slow SMA.
      - Risk management via fixed % of equity and ATR-based stop.
    """

    def __init__(self, fast: int = 20, slow: int = 50, atr_window: int = 14):
        assert fast < slow, "fast must be < slow"
        self.fast = fast
        self.slow = slow
        self.atr_window = atr_window

    @staticmethod
    def _atr(df: pd.DataFrame, n: int) -> pd.Series:
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift()).abs()
        low_close = (df["low"] - df["close"].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(n, min_periods=n).mean()

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df[f"sma_{self.fast}"] = df["close"].rolling(self.fast, min_periods=self.fast).mean()
        df[f"sma_{self.slow}"] = df["close"].rolling(self.slow, min_periods=self.slow).mean()
        df["atr"] = self._atr(df, self.atr_window)

        # Signals: 1 for long, 0 for flat
        cross_up = (df[f"sma_{self.fast}"] > df[f"sma_{self.slow}"]) & (
            df[f"sma_{self.fast}"].shift() <= df[f"sma_{self.slow}"].shift()
        )
        cross_dn = (df[f"sma_{self.fast}"] < df[f"sma_{self.slow}"]) & (
            df[f"sma_{self.fast}"].shift() >= df[f"sma_{self.slow}"].shift()
        )

        position = 0
        pos_list = []
        for up, dn in zip(cross_up, cross_dn):
            if up:
                position = 1
            elif dn:
                position = 0
            pos_list.append(position)

        df["signal"] = pd.Series(pos_list, index=df.index).fillna(method="ffill").fillna(0)
        return df
```

---

## 📈 src/backtester.py

```python
from __future__ import annotations
import numpy as np
import pandas as pd

class Backtester:
    def __init__(self, initial_cash: float = 10_000.0, fee: float = 0.0005, risk_pct: float = 0.01):
        self.initial_cash = float(initial_cash)
        self.fee = float(fee)
        self.risk_pct = float(risk_pct)

    @staticmethod
    def _safe_pct_change(series: pd.Series) -> pd.Series:
        return series.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "signal" not in df:
            raise ValueError("DataFrame must contain a 'signal' column (0/1).")

        returns = self._safe_pct_change(df["close"])  # daily returns
        df["ret"] = returns

        # Position sizing (naive): when in position, exposure = min(1, risk-based cap)
        # Use ATR stop distance to compute size so that risk per trade ≈ risk_pct of equity.
        atr = df["atr"].replace(0, np.nan)
        stop_dist = 2.0 * atr  # 2x ATR stop
        # shares = (risk_pct * equity) / stop_dist; convert to exposure by shares*price / equity
        exposure = (self.risk_pct / (stop_dist / df["close"]))
        exposure = exposure.clip(lower=0, upper=1).fillna(0)
        df["exposure"] = df["signal"] * exposure

        # Apply fees on position changes (turnover)
        turnover = (df["exposure"] - df["exposure"].shift()).abs().fillna(0)
        fee_drag = turnover * self.fee

        # Equity curve
        strat_ret = df["exposure"] * df["ret"] - fee_drag
        df["equity"] = (1 + strat_ret).cumprod() * self.initial_cash

        # Bench (buy & hold)
        df["buy_hold"] = (1 + returns).cumprod() * self.initial_cash

        # Metrics
        daily = strat_ret
        ann_factor = 252
        cagr = (df["equity"].iloc[-1] / self.initial_cash) ** (ann_factor / len(df)) - 1
        sharpe = np.sqrt(ann_factor) * daily.mean() / (daily.std(ddof=0) + 1e-12)

        # Max drawdown
        roll_max = df["equity"].cummax()
        dd = df["equity"] / roll_max - 1
        max_dd = dd.min()

        # Win rate on days with exposure > 0
        mask = df["exposure"] > 0
        pnl_days = (daily[mask] > 0).sum()
        total_days = mask.sum()
        win_rate = (pnl_days / total_days) if total_days > 0 else 0.0

        df.attrs["metrics"] = {
            "CAGR": float(cagr),
            "Sharpe": float(sharpe),
            "MaxDrawdown": float(max_dd),
            "WinRate": float(win_rate),
            "FinalEquity": float(df["equity"].iloc[-1]),
        }
        return df
```

---

## 🖥️ src/plotter.py

```python
from __future__ import annotations
import matplotlib.pyplot as plt
import pandas as pd


def plot_equity(df: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 5))
    df[["equity", "buy_hold"]].dropna().plot(ax=plt.gca())
    plt.title("Strategy vs Buy & Hold")
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.tight_layout()
    plt.show()


def plot_price_signals(df: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 5))
    df["close"].plot(ax=plt.gca(), label="Close")
    sma_cols = [c for c in df.columns if c.startswith("sma_")]
    for c in sma_cols:
        df[c].plot(ax=plt.gca(), alpha=0.8)
    # Mark positions
    long_points = df.index[df["signal"].diff().fillna(0) > 0]
    exit_points = df.index[df["signal"].diff().fillna(0) < 0]
    plt.scatter(long_points, df.loc[long_points, "close"], marker="^")
    plt.scatter(exit_points, df.loc[exit_points, "close"], marker="v")
    plt.title("Price with Signals")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.tight_layout()
    plt.show()
```

---

## 🏃 src/main.py

```python
from __future__ import annotations
import argparse
from data import load_ohlcv
from strategy import MovingAverageCrossover
from backtester import Backtester
from plotter import plot_equity, plot_price_signals


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SMA Crossover Backtester")
    p.add_argument("--ticker", type=str, required=True, help="Ticker, e.g. AAPL")
    p.add_argument("--start", type=str, required=True, help="YYYY-MM-DD")
    p.add_argument("--end", type=str, required=True, help="YYYY-MM-DD")
    p.add_argument("--fast", type=int, default=20)
    p.add_argument("--slow", type=int, default=50)
    p.add_argument("--risk", type=float, default=0.01, help="Risk % of equity per trade")
    p.add_argument("--initial", type=float, default=10000.0)
    p.add_argument("--fee", type=float, default=0.0005, help="One-way fee as decimal")
    p.add_argument("--no-plots", action="store_true", help="Disable matplotlib windows")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # 1) Data
    df = load_ohlcv(args.ticker, args.start, args.end)

    # 2) Strategy signals
    strat = MovingAverageCrossover(fast=args.fast, slow=args.slow)
    df = strat.generate(df)

    # 3) Backtest
    bt = Backtester(initial_cash=args.initial, fee=args.fee, risk_pct=args.risk)
    result = bt.run(df)
    metrics = result.attrs.get("metrics", {})

    # 4) Print metrics
    print("\n=== RESULTS ===")
    for k, v in metrics.items():
        if k in {"CAGR", "Sharpe", "MaxDrawdown", "WinRate"}:
            print(f"{k:>12}: {v:.4f}")
        else:
            print(f"{k:>12}: {v:,.2f}")

    # 5) Plots
    if not args.no_plots:
        plot_equity(result)
        plot_price_signals(result)


if __name__ == "__main__":
    main()
```

---

## 🧪 examples/run\_example.sh

```bash
#!/usr/bin/env bash
python src/main.py --ticker MSFT --start 2015-01-01 --end 2025-08-01 \
  --fast 10 --slow 40 --risk 0.01 --initial 10000 --fee 0.0003
```

---

## ✍️ How to Use This in Your GitHub

1. Create a new repository on GitHub (empty).
2. Locally, make a folder `trading-repo` and copy the above structure/files.
3. Initialize and push:

   ```bash
   git init
   git add .
   git commit -m "feat: trading backtester starter"
   git branch -M main
   git remote add origin https://github.com/<YOUR_USERNAME>/<REPO_NAME>.git
   git push -u origin main
   ```

---

## 🔧 Next Ideas (easy enhancements)

* Add **walk-forward optimization** for SMA windows.
* Add **position shorting** and **stop-loss / take-profit** orders.
* Save results to CSV and **export plots** to PNG.
* Try other indicators: RSI, MACD, Bollinger Bands.
* Wrap a minimal **Streamlit UI** for interactive runs.
