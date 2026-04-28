from pathlib import Path

import pandas as pd

from .backtest import run_backtest, run_backtest_for_records
from .config import (
    DATA_DIR,
    DEFAULT_FILING_LIMIT,
    DEFAULT_START_DATE,
    DEFAULT_TICKERS,
    DICTIONARY_PATH,
    ITEM1_OUTPUT_PATH,
)
from .filings import download_filings, extract_item1, get_latest_item1_for_ticker
from .returns import get_company_name
from .sentiment import score_single_filing


def save_extracted_item1_data(extracted_item_1_data, output_path: Path = ITEM1_OUTPUT_PATH) -> pd.DataFrame:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    extracted_df = pd.DataFrame(extracted_item_1_data)
    extracted_df.to_csv(output_path, index=False)
    return extracted_df


def run_pipeline(
    tickers=None,
    start_date: str = DEFAULT_START_DATE,
    filing_limit: int = DEFAULT_FILING_LIMIT,
    dictionary_path: Path = DICTIONARY_PATH,
):
    if tickers is None:
        tickers = DEFAULT_TICKERS

    if not tickers:
        raise ValueError("Ticker list is empty. Add at least one ticker in main.py.")

    download_filings(tickers, start_date=start_date, limit=filing_limit)
    extracted_item_1_data = extract_item1()
    extracted_df = save_extracted_item1_data(extracted_item_1_data)

    backtest_results_df, backtest_summary = run_backtest(
        extracted_csv_path=ITEM1_OUTPUT_PATH,
        dictionary_path=dictionary_path,
    )

    return extracted_df, backtest_results_df, backtest_summary


def analyze_latest_filing_for_ticker(ticker: str, dictionary_path: Path = DICTIONARY_PATH) -> dict:
    normalized_ticker = ticker.strip().upper()
    if not normalized_ticker:
        raise ValueError("Ticker cannot be blank.")

    latest_filing = get_latest_item1_for_ticker(normalized_ticker)
    if latest_filing is None:
        raise ValueError(f"Could not retrieve a 10-K filing for {normalized_ticker}.")

    scored_filing = score_single_filing(latest_filing, dictionary_path=dictionary_path)
    scored_filing["company_name"] = get_company_name(normalized_ticker) or normalized_ticker
    scored_filing["preview_text"] = " ".join(scored_filing["item_1_content"].split()[:300])
    return scored_filing


def run_ticker_backtest(
    ticker: str,
    filing_limit: int = DEFAULT_FILING_LIMIT,
    dictionary_path: Path = DICTIONARY_PATH,
):
    normalized_ticker = ticker.strip().upper()
    if not normalized_ticker:
        raise ValueError("Ticker cannot be blank.")

    download_filings([normalized_ticker], limit=filing_limit)
    ticker_filings = extract_item1(ticker_filter=normalized_ticker)
    if not ticker_filings:
        raise ValueError(f"Could not retrieve filings for {normalized_ticker}.")

    backtest_results_df, backtest_summary = run_backtest_for_records(
        ticker_filings,
        dictionary_path=dictionary_path,
    )
    return ticker_filings, backtest_results_df, backtest_summary
