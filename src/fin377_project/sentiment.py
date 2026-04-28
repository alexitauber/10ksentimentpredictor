import re
from pathlib import Path
from typing import Optional

import pandas as pd

from .config import DICTIONARY_PATH, NEGATIVE_THRESHOLD, POSITIVE_THRESHOLD, SENTIMENT_OUTPUT_PATH

TOKEN_PATTERN = re.compile(r"\b[A-Z][A-Z'-]*\b")


def load_dictionary(dictionary_path: Path = DICTIONARY_PATH) -> dict[str, int]:
    dictionary_df = pd.read_csv(dictionary_path)
    dictionary_df = dictionary_df.dropna(subset=["Word"])

    sentiment_dict = pd.Series(
        dictionary_df["Pos"].values + dictionary_df["Neg"].values,
        index=dictionary_df["Word"].astype(str).str.upper(),
    ).to_dict()

    return {word: int(score) for word, score in sentiment_dict.items()}


def predict_direction(compound: float) -> str:
    if compound > POSITIVE_THRESHOLD:
        return "UP"
    if compound < NEGATIVE_THRESHOLD:
        return "DOWN"
    return "NEUTRAL"


def sentiment_to_confidence(compound: float) -> Optional[int]:
    direction = predict_direction(compound)
    if direction == "NEUTRAL":
        return None
    return max(50, min(100, round(50 + abs(compound) * 50)))


def calculate_sentiment_score(text: str, sentiment_dict: dict[str, int]) -> dict:
    words = TOKEN_PATTERN.findall(text.upper())
    total_words = len(words)

    positive_count = 0
    negative_count = 0
    matched_words = []

    for word in words:
        score = sentiment_dict.get(word, 0)
        if score > 0:
            positive_count += 1
            matched_words.append(word)
        elif score < 0:
            negative_count += 1
            matched_words.append(word)

    matched_count = positive_count + negative_count
    compound = (positive_count - negative_count) / matched_count if matched_count else 0.0

    if total_words:
        pos_ratio = positive_count / total_words
        neg_ratio = negative_count / total_words
        neu_ratio = max(0.0, 1 - pos_ratio - neg_ratio)
    else:
        pos_ratio = 0.0
        neg_ratio = 0.0
        neu_ratio = 0.0

    return {
        "compound": round(compound, 6),
        "pos": round(pos_ratio, 6),
        "neg": round(neg_ratio, 6),
        "neu": round(neu_ratio, 6),
        "positive_word_count": positive_count,
        "negative_word_count": negative_count,
        "matched_word_count": matched_count,
        "total_word_count": total_words,
        "predicted_direction": predict_direction(compound),
        "confidence": sentiment_to_confidence(compound),
        "matched_words_preview": ", ".join(matched_words[:25]),
    }


def score_extracted_filings(
    extracted_csv_path: Path,
    dictionary_path: Path = DICTIONARY_PATH,
    output_csv_path: Path = SENTIMENT_OUTPUT_PATH,
) -> pd.DataFrame:
    extracted_df = pd.read_csv(extracted_csv_path)
    sentiment_dict = load_dictionary(dictionary_path)

    scored_rows = []
    for row in extracted_df.itertuples(index=False):
        score = calculate_sentiment_score(row.item_1_content, sentiment_dict)
        scored_rows.append(
            {
                "ticker": row.ticker,
                "filing_date": row.filing_date,
                "accession_number": row.accession_number,
                **score,
            }
        )

    scored_df = pd.DataFrame(scored_rows)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    scored_df.to_csv(output_csv_path, index=False)
    return scored_df


def score_filing_records(
    filing_records: list[dict],
    dictionary_path: Path = DICTIONARY_PATH,
) -> pd.DataFrame:
    sentiment_dict = load_dictionary(dictionary_path)

    scored_rows = []
    for record in filing_records:
        score = calculate_sentiment_score(record["item_1_content"], sentiment_dict)
        scored_rows.append(
            {
                "ticker": record["ticker"],
                "filing_date": record["filing_date"],
                "accession_number": record["accession_number"],
                **score,
            }
        )

    return pd.DataFrame(scored_rows)


def score_single_filing(
    filing_record: dict,
    dictionary_path: Path = DICTIONARY_PATH,
) -> dict:
    sentiment_dict = load_dictionary(dictionary_path)
    score = calculate_sentiment_score(filing_record["item_1_content"], sentiment_dict)
    return {
        "ticker": filing_record["ticker"],
        "filing_date": filing_record["filing_date"],
        "accession_number": filing_record["accession_number"],
        "item_1_content": filing_record["item_1_content"],
        **score,
    }
