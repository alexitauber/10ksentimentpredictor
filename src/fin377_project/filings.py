import os
import re
from pathlib import Path
from typing import Optional

from bs4 import BeautifulSoup
from sec_edgar_downloader import Downloader

from .config import FILINGS_DIR, SEC_COMPANY_NAME, SEC_EMAIL

ITEM_1_START_PATTERN = re.compile(r"\bitem\s+1\b(?!\s*[ab])", re.IGNORECASE)
ITEM_1_END_PATTERN = re.compile(r"\bitem\s+(1a|1b|2|3)\b", re.IGNORECASE)
FILED_AS_OF_DATE_PATTERN = re.compile(r"FILED AS OF DATE:\s*(\d{8})")


def _build_downloader(download_dir: Path = FILINGS_DIR) -> Downloader:
    return Downloader(SEC_COMPANY_NAME, SEC_EMAIL, str(download_dir))


def download_filings(
    tickers: list[str],
    start_date: Optional[str] = None,
    limit: int = 3,
    download_dir: Path = FILINGS_DIR,
) -> None:
    download_dir.mkdir(parents=True, exist_ok=True)
    downloader = _build_downloader(download_dir)

    for ticker in tickers:
        print(f"Downloading {ticker}...")
        download_kwargs = {"limit": limit, "download_details": True}
        if start_date:
            download_kwargs["after"] = start_date
        downloader.get("10-K", ticker, **download_kwargs)

    print("Done.")


def _extract_filing_date(filing_dir: str) -> Optional[str]:
    submission_path = os.path.join(filing_dir, "full-submission.txt")
    if not os.path.exists(submission_path):
        return None

    with open(submission_path, "r", encoding="utf-8", errors="ignore") as submission_file:
        submission_text = submission_file.read()

    match = FILED_AS_OF_DATE_PATTERN.search(submission_text)
    if not match:
        return None

    filing_date = match.group(1)
    return f"{filing_date[:4]}-{filing_date[4:6]}-{filing_date[6:]}"


def _clean_extracted_text(text: str) -> str:
    lines = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        lowered = line.lower()
        if lowered == "table of contents":
            continue

        if set(line) <= {"_", "-", " "}:
            continue

        if re.fullmatch(r"\d+", line):
            continue

        if re.fullmatch(r"page\s+\d+", lowered):
            continue

        line = re.sub(r"\s+", " ", line)
        lines.append(line)

    return "\n".join(lines)


def _extract_item_1_from_text(text: str) -> str:
    normalized_text = text.replace("\xa0", " ")
    normalized_text = normalized_text.replace("\r", "\n")
    normalized_text = re.sub(r"\n{2,}", "\n", normalized_text).strip()

    best_section = ""

    for start_match in ITEM_1_START_PATTERN.finditer(normalized_text):
        end_match = ITEM_1_END_PATTERN.search(normalized_text, start_match.end())
        if not end_match:
            continue

        candidate = normalized_text[start_match.start():end_match.start()].strip()
        candidate = _clean_extracted_text(candidate)

        if len(candidate) > len(best_section):
            best_section = candidate

    return best_section


def extract_item1(base_dir: Path = FILINGS_DIR, ticker_filter: Optional[str] = None) -> list[dict]:
    extracted_item_1_data = []
    ticker_filter = ticker_filter.upper() if ticker_filter else None

    for root, _, files in os.walk(base_dir):
        for file_name in files:
            if file_name != "primary-document.html":
                continue

            file_path = os.path.join(root, file_name)
            path_parts = file_path.split(os.sep)
            ticker = path_parts[-4]

            if ticker_filter and ticker != ticker_filter:
                continue

            accession_number = path_parts[-2]
            filing_date = _extract_filing_date(root)

            with open(file_path, "r", encoding="utf-8", errors="ignore") as filing_file:
                html_content = filing_file.read()

            soup = BeautifulSoup(html_content, "html.parser")
            for table in soup.find_all("table"):
                table.decompose()

            text = soup.get_text(separator="\n")
            item_1_content = _extract_item_1_from_text(text)

            if item_1_content and filing_date:
                extracted_item_1_data.append(
                    {
                        "ticker": ticker,
                        "filing_date": filing_date,
                        "accession_number": accession_number,
                        "item_1_content": item_1_content,
                    }
                )

    extracted_item_1_data.sort(key=lambda row: (row["ticker"], row["filing_date"]))
    print(f"Extracted Item 1 content from {len(extracted_item_1_data)} filings.")
    return extracted_item_1_data


def get_latest_item1_for_ticker(ticker: str, filings_dir: Path = FILINGS_DIR) -> Optional[dict]:
    ticker = ticker.strip().upper()
    download_filings([ticker], limit=1, download_dir=filings_dir)
    ticker_filings = extract_item1(base_dir=filings_dir, ticker_filter=ticker)
    if not ticker_filings:
        return None

    return max(ticker_filings, key=lambda row: row["filing_date"])
