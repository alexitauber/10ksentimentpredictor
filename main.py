from src.fin377_project.config import ITEM1_OUTPUT_PATH
from src.fin377_project.pipeline import run_pipeline


if __name__ == "__main__":
    extracted_df, backtest_results_df, backtest_summary = run_pipeline()

    print(f"Saved {len(extracted_df)} extracted filings to {ITEM1_OUTPUT_PATH}")
    if not backtest_results_df.empty:
        print(backtest_results_df[["ticker", "filing_date", "compound", "predicted_direction", "actual_direction", "correct"]])
    print(backtest_summary)
