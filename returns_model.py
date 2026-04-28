from datetime import timedelta

import pandas as pd
import yfinance as yf


def _get_close_prices(price_data, ticker):
    """
    Pull the Close column out of yfinance data.

    yfinance can return either normal columns or multi-level columns depending
    on the version/settings, so this helper keeps that detail in one place.
    """
    if price_data is None or price_data.empty:
        return None

    # Some yfinance versions return multi-level columns like ("Close", "AAPL").
    if isinstance(price_data.columns, pd.MultiIndex):
        if ("Close", ticker) in price_data.columns:
            return price_data[("Close", ticker)].dropna()

        if "Close" in price_data.columns.get_level_values(0):
            close_data = price_data["Close"]
            if ticker in close_data.columns:
                return close_data[ticker].dropna()
            if len(close_data.columns) == 1:
                return close_data.iloc[:, 0].dropna()

    # Most single-ticker downloads have columns like Open, High, Low, Close.
    if "Close" in price_data.columns:
        return price_data["Close"].dropna()

    return None


def get_forward_returns(ticker, filing_date):
    """
    Calculate a stock's 30-trading-day forward return and abnormal return.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol, such as "AAPL".
    filing_date : str or datetime
        10-K filing date, such as "2023-10-27".

    Returns
    -------
    dict or None
        Dictionary with stock return, market return, CAR, and actual direction.
        Returns None if the data cannot be downloaded or is incomplete.
    """
    try:
        # Make sure the ticker is a clean uppercase string.
        if not isinstance(ticker, str) or ticker.strip() == "":
            print("Error: ticker must be a non-empty string.")
            return None

        ticker = ticker.strip().upper()

        # Convert the filing date to a pandas Timestamp so date math is easy.
        filing_date_timestamp = pd.to_datetime(filing_date)
        if pd.isna(filing_date_timestamp):
            print("Error: filing_date is invalid.")
            return None

        start_date = filing_date_timestamp.date()
        # Use a 60-calendar-day window as a safety buffer.
        # We still calculate returns using only trading-day observations 0 and 30.
        end_date = start_date + timedelta(days=60)

        # Download adjusted close data for the stock and for SPY.
        # With auto_adjust=True, the Close column is already adjusted.
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

        # If the ticker is invalid or yfinance finds no data, this will catch it.
        if stock_close_prices is None or stock_close_prices.empty:
            print(f"Error: no usable Close price data found for {ticker}.")
            return None

        if spy_close_prices is None or spy_close_prices.empty:
            print("Error: no usable Close price data found for SPY.")
            return None

        # Need at least 31 observations: day 0 through day 30.
        if len(stock_close_prices) < 31:
            print(f"Error: fewer than 31 trading days found for {ticker}.")
            return None

        if len(spy_close_prices) < 31:
            print("Error: fewer than 31 trading days found for SPY.")
            return None

        # If filing_date is not a trading day, index 0 is the next trading day.
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
        # This catches yfinance download failures and any unexpected data issues.
        print(f"Error: could not calculate forward returns for {ticker}: {error}")
        return None

