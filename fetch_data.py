import yfinance as yf
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_FILE = BASE_DIR / "All_Stocks_Data.csv"


def fetch_stock_data(ticker, start="2020-01-01", end="2024-01-01"):
    stock = yf.download(ticker, start=start, end=end, progress=False)

    if stock.empty:
        raise ValueError(
            f"No data returned for ticker '{ticker}'. Check the ticker symbol, date range, and internet connection."
        )

    # yfinance may return MultiIndex columns; this flattens to a single Close series.
    if isinstance(stock.columns, pd.MultiIndex):
        if ("Close", ticker) in stock.columns:
            close_series = stock[("Close", ticker)]
        elif ("Adj Close", ticker) in stock.columns:
            close_series = stock[("Adj Close", ticker)]
        else:
            close_series = stock.xs("Close", axis=1, level=0).iloc[:, 0]
    else:
        if "Close" in stock.columns:
            close_series = stock["Close"]
        elif "Adj Close" in stock.columns:
            close_series = stock["Adj Close"]
        else:
            raise ValueError(
                "Could not find a Close or Adj Close column in downloaded data."
            )

    result = close_series.to_frame(name="Close")
    result.index.name = "Date"
    return result


def save_default_dataset(ticker="AAPL"):
    df = fetch_stock_data(ticker)
    df.to_csv(DEFAULT_OUTPUT_FILE)
    print(f"[OK] Data saved to {DEFAULT_OUTPUT_FILE}")
    print(df.head())


if __name__ == "__main__":
    save_default_dataset()
