import json
from pathlib import Path

import pandas as pd

from .config import (
    BACKTEST_RESULTS_PATH,
    BACKTEST_SUMMARY_PATH,
    DICTIONARY_PATH,
    ITEM1_OUTPUT_PATH,
    RETURNS_OUTPUT_PATH,
    SENTIMENT_OUTPUT_PATH,
)
from .returns import get_forward_returns
from .sentiment import score_extracted_filings, score_filing_records


def build_returns_dataset(sentiment_df: pd.DataFrame) -> pd.DataFrame:
    returns_rows = []

    for row in sentiment_df.itertuples(index=False):
        returns_result = get_forward_returns(row.ticker, row.filing_date)
        if returns_result is not None and returns_result["actual_direction"] is not None:
            returns_rows.append(returns_result)

    return pd.DataFrame(returns_rows)


def compute_backtest_accuracy(sentiment_df: pd.DataFrame, returns_df: pd.DataFrame):
    sentiment_df = sentiment_df.copy()
    returns_df = returns_df.copy()

    sentiment_df["filing_date"] = pd.to_datetime(sentiment_df["filing_date"]).dt.strftime("%Y-%m-%d")
    returns_df["filing_date"] = pd.to_datetime(returns_df["filing_date"]).dt.strftime("%Y-%m-%d")

    merged_df = pd.merge(sentiment_df, returns_df, on=["ticker", "filing_date"], how="inner")
    active_signals = merged_df[merged_df["predicted_direction"] != "NEUTRAL"].copy()
    active_signals["correct"] = (
        active_signals["predicted_direction"] == active_signals["actual_direction"]
    ).astype(int)

    n_obs = len(active_signals)
    overall_accuracy = active_signals["correct"].mean() if n_obs > 0 else 0.0

    up_calls = active_signals[active_signals["predicted_direction"] == "UP"]
    down_calls = active_signals[active_signals["predicted_direction"] == "DOWN"]

    up_accuracy = up_calls["correct"].mean() if len(up_calls) > 0 else 0.0
    down_accuracy = down_calls["correct"].mean() if len(down_calls) > 0 else 0.0

    summary_stats = {
        "overall_accuracy": round(float(overall_accuracy), 4),
        "up_accuracy": round(float(up_accuracy), 4),
        "down_accuracy": round(float(down_accuracy), 4),
        "n_observations": int(n_obs),
    }

    return active_signals, summary_stats


def run_backtest(
    extracted_csv_path: Path = ITEM1_OUTPUT_PATH,
    dictionary_path: Path = DICTIONARY_PATH,
    sentiment_output_path: Path = SENTIMENT_OUTPUT_PATH,
    returns_output_path: Path = RETURNS_OUTPUT_PATH,
    backtest_results_path: Path = BACKTEST_RESULTS_PATH,
    backtest_summary_path: Path = BACKTEST_SUMMARY_PATH,
):
    sentiment_df = score_extracted_filings(
        extracted_csv_path=extracted_csv_path,
        dictionary_path=dictionary_path,
        output_csv_path=sentiment_output_path,
    )

    returns_df = build_returns_dataset(sentiment_df)
    returns_output_path.parent.mkdir(parents=True, exist_ok=True)
    returns_df.to_csv(returns_output_path, index=False)

    active_signals, summary_stats = compute_backtest_accuracy(sentiment_df, returns_df)
    backtest_results_path.parent.mkdir(parents=True, exist_ok=True)
    active_signals.to_csv(backtest_results_path, index=False)

    with open(backtest_summary_path, "w", encoding="utf-8") as summary_file:
        json.dump(summary_stats, summary_file, indent=2)

    return active_signals, summary_stats


def run_backtest_for_records(filing_records: list[dict], dictionary_path: Path = DICTIONARY_PATH):
    sentiment_df = score_filing_records(filing_records, dictionary_path=dictionary_path)
    returns_df = build_returns_dataset(sentiment_df)
    active_signals, summary_stats = compute_backtest_accuracy(sentiment_df, returns_df)
    return active_signals, summary_stats
