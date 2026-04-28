from datetime import timedelta
from typing import Optional

import pandas as pd
import yfinance as yf


def _get_close_prices(price_data, ticker):
    if price_data is None or price_data.empty:
        return None

    if isinstance(price_data.columns, pd.MultiIndex):
        if ("Close", ticker) in price_data.columns:
            return price_data[("Close", ticker)].dropna()

        if "Close" in price_data.columns.get_level_values(0):
            close_data = price_data["Close"]
            if ticker in close_data.columns:
                return close_data[ticker].dropna()
            if len(close_data.columns) == 1:
                return close_data.iloc[:, 0].dropna()

    if "Close" in price_data.columns:
        return price_data["Close"].dropna()

    return None


def get_forward_returns(ticker, filing_date):
    try:
        if not isinstance(ticker, str) or ticker.strip() == "":
            print("Error: ticker must be a non-empty string.")
            return None

        ticker = ticker.strip().upper()
        filing_date_timestamp = pd.to_datetime(filing_date)
        if pd.isna(filing_date_timestamp):
            print("Error: filing_date is invalid.")
            return None

        start_date = filing_date_timestamp.date()
        end_date = start_date + timedelta(days=60)

        stock_data = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            auto_adjust=True,
            progress=False,
        )
        spy_data = yf.download(
            "SPY",
            start=start_date,
            end=end_date,
            auto_adjust=True,
            progress=False,
        )

        stock_close_prices = _get_close_prices(stock_data, ticker)
        spy_close_prices = _get_close_prices(spy_data, "SPY")

        if stock_close_prices is None or stock_close_prices.empty:
            print(f"Error: no usable Close price data found for {ticker}.")
            return None

        if spy_close_prices is None or spy_close_prices.empty:
            print("Error: no usable Close price data found for SPY.")
            return None

        if len(stock_close_prices) < 31:
            print(f"Error: fewer than 31 trading days found for {ticker}.")
            return None

        if len(spy_close_prices) < 31:
            print("Error: fewer than 31 trading days found for SPY.")
            return None

        stock_price_day0 = stock_close_prices.iloc[0]
        stock_price_day30 = stock_close_prices.iloc[30]
        spy_price_day0 = spy_close_prices.iloc[0]
        spy_price_day30 = spy_close_prices.iloc[30]

        stock_return = (stock_price_day30 - stock_price_day0) / stock_price_day0
        market_return = (spy_price_day30 - spy_price_day0) / spy_price_day0
        car = stock_return - market_return

        if car > 0:
            actual_direction = "UP"
        elif car < 0:
            actual_direction = "DOWN"
        else:
            actual_direction = None

        return {
            "ticker": ticker,
            "filing_date": filing_date,
            "stock_return": stock_return,
            "market_return": market_return,
            "CAR": car,
            "actual_direction": actual_direction,
        }

    except Exception as error:
        print(f"Error: could not calculate forward returns for {ticker}: {error}")
        return None


def get_company_name(ticker: str) -> Optional[str]:
    try:
        info = yf.Ticker(ticker.strip().upper()).info
    except Exception:
        return None

    for key in ("shortName", "longName", "displayName"):
        value = info.get(key)
        if value:
            return value
    return None
