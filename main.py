from pathlib import Path

import pandas as pd

from Pull_Item_1 import download_filings, extract_item1
from backtest import run_backtest


MY_FIRM = ["CVX"]
COMPETITORS = []
TICKERS = MY_FIRM + COMPETITORS
START_DATE = "2021-01-01"
FILING_LIMIT = 3
DATA_DIR = Path("data")
ITEM1_OUTPUT_PATH = DATA_DIR / "item1_extracted.csv"
DICTIONARY_PATH = "LM_MasterDictionary_1993-2021.csv"


def save_extracted_item1_data(extracted_item_1_data, output_path: Path = ITEM1_OUTPUT_PATH) -> pd.DataFrame:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    extracted_df = pd.DataFrame(extracted_item_1_data)
    extracted_df.to_csv(output_path, index=False)
    return extracted_df


def run_pipeline(
    tickers=None,
    start_date: str = START_DATE,
    filing_limit: int = FILING_LIMIT,
    dictionary_path: str = DICTIONARY_PATH,
):
    if tickers is None:
        tickers = TICKERS

    if not tickers:
        raise ValueError("Ticker list is empty. Add at least one ticker in main.py.")

    download_filings(tickers, start_date=start_date, limit=filing_limit)
    extracted_item_1_data = extract_item1()
    extracted_df = save_extracted_item1_data(extracted_item_1_data)

    backtest_results_df, backtest_summary = run_backtest(
        extracted_csv_path=str(ITEM1_OUTPUT_PATH),
        dictionary_path=dictionary_path,
        output_dir=str(DATA_DIR),
    )

    return extracted_df, backtest_results_df, backtest_summary


if __name__ == "__main__":
    extracted_df, backtest_results_df, backtest_summary = run_pipeline()

    print(f"Saved {len(extracted_df)} extracted filings to {ITEM1_OUTPUT_PATH}")
    if not backtest_results_df.empty:
        print(backtest_results_df[["ticker", "filing_date", "compound", "predicted_direction", "actual_direction", "correct"]])
    print(backtest_summary)
