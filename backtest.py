import json
from pathlib import Path

import pandas as pd

from Compare_w_vader import score_extracted_filings
from returns_model import get_forward_returns


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
    extracted_csv_path: str = "data/item1_extracted.csv",
    dictionary_path: str = "LM_MasterDictionary_1993-2021.csv",
    output_dir: str = "data",
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    sentiment_df = score_extracted_filings(
        extracted_csv_path=extracted_csv_path,
        dictionary_path=dictionary_path,
        output_csv_path=str(output_path / "sentiment_scores.csv"),
    )

    returns_df = build_returns_dataset(sentiment_df)
    returns_df.to_csv(output_path / "returns_data.csv", index=False)

    active_signals, summary_stats = compute_backtest_accuracy(sentiment_df, returns_df)
    active_signals.to_csv(output_path / "backtest_results.csv", index=False)

    with open(output_path / "backtest_summary.json", "w", encoding="utf-8") as summary_file:
        json.dump(summary_stats, summary_file, indent=2)

    return active_signals, summary_stats
