import re
from pathlib import Path

import pandas as pd

POSITIVE_THRESHOLD = 0.05
NEGATIVE_THRESHOLD = -0.05
TOKEN_PATTERN = re.compile(r"\b[A-Z][A-Z'-]*\b")


def load_dictionary(dictionary_path: str) -> dict[str, int]:
    dictionary_df = pd.read_csv(dictionary_path)
    dictionary_df = dictionary_df.dropna(subset=["Word"])

    sentiment_dict = pd.Series(
        dictionary_df["Pos"].values + dictionary_df["Neg"].values,
        index=dictionary_df["Word"].astype(str).str.upper(),
    ).to_dict()

    return {word: int(score) for word, score in sentiment_dict.items()}


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

    if matched_count:
        compound = (positive_count - negative_count) / matched_count
    else:
        compound = 0.0

    if total_words:
        pos_ratio = positive_count / total_words
        neg_ratio = negative_count / total_words
        neu_ratio = max(0.0, 1 - pos_ratio - neg_ratio)
    else:
        pos_ratio = 0.0
        neg_ratio = 0.0
        neu_ratio = 0.0

    if compound > POSITIVE_THRESHOLD:
        predicted_direction = "UP"
    elif compound < NEGATIVE_THRESHOLD:
        predicted_direction = "DOWN"
    else:
        predicted_direction = "NEUTRAL"

    return {
        "compound": round(compound, 6),
        "pos": round(pos_ratio, 6),
        "neg": round(neg_ratio, 6),
        "neu": round(neu_ratio, 6),
        "positive_word_count": positive_count,
        "negative_word_count": negative_count,
        "matched_word_count": matched_count,
        "total_word_count": total_words,
        "predicted_direction": predicted_direction,
        "matched_words_preview": ", ".join(matched_words[:25]),
    }


def score_extracted_filings(
    extracted_csv_path: str = "data/item1_extracted.csv",
    dictionary_path: str = "LM_MasterDictionary_1993-2021.csv",
    output_csv_path: str = "data/sentiment_scores.csv",
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
    Path(output_csv_path).parent.mkdir(parents=True, exist_ok=True)
    scored_df.to_csv(output_csv_path, index=False)
    return scored_df
