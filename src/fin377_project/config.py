from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
FILINGS_DIR = PROJECT_ROOT / "sec-edgar-filings"


def _resolve_dictionary_path() -> Path:
    candidate_paths = [
        PROJECT_ROOT / "LM_MasterDictionary_1993-2021.csv",
        PROJECT_ROOT / "LM_MasterDictionary_Filtered.csv",
    ]

    for candidate in candidate_paths:
        if candidate.exists():
            return candidate

    return candidate_paths[0]


DICTIONARY_PATH = _resolve_dictionary_path()

ITEM1_OUTPUT_PATH = DATA_DIR / "item1_extracted.csv"
SENTIMENT_OUTPUT_PATH = DATA_DIR / "sentiment_scores.csv"
RETURNS_OUTPUT_PATH = DATA_DIR / "returns_data.csv"
BACKTEST_RESULTS_PATH = DATA_DIR / "backtest_results.csv"
BACKTEST_SUMMARY_PATH = DATA_DIR / "backtest_summary.json"

DEFAULT_TICKERS = ["CVX"]
DEFAULT_START_DATE = "2021-01-01"
DEFAULT_FILING_LIMIT = 3

SEC_COMPANY_NAME = "Lehigh University"
SEC_EMAIL = "ait227@lehigh.edu"

POSITIVE_THRESHOLD = 0.05
NEGATIVE_THRESHOLD = -0.05
