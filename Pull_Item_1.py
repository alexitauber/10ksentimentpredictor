import os
import re
from typing import Optional

from bs4 import BeautifulSoup
from sec_edgar_downloader import Downloader


def download_filings(tickers: list, start_date: str, limit: int = 3) -> None:
    dl = Downloader("Lehigh University", "ait227@lehigh.edu")
    for t in tickers:
        print(f"Downloading {t}...")
        dl.get("10-K", t, limit=limit, after=start_date, download_details=True)
    print("Done.")


ITEM_1_START_PATTERN = re.compile(r"\bitem\s+1\b(?!\s*[ab])", re.IGNORECASE)
ITEM_1_END_PATTERN = re.compile(r"\bitem\s+(1a|1b|2|3)\b", re.IGNORECASE)
FILED_AS_OF_DATE_PATTERN = re.compile(r"FILED AS OF DATE:\s*(\d{8})")


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


def extract_item1(base_dir: str = 'sec-edgar-filings') -> list:
    extracted_item_1_data = []

    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file == 'primary-document.html':
                file_path = os.path.join(root, file)

                path_parts = file_path.split(os.sep)
                ticker = path_parts[-4]
                accession_number = path_parts[-2]
                filing_date = _extract_filing_date(root)

                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    html_content = f.read()

                soup = BeautifulSoup(html_content, 'html.parser')
                for table in soup.find_all("table"):
                    table.decompose()

                text = soup.get_text(separator='\n')
                item_1_content = _extract_item_1_from_text(text)

                if item_1_content and filing_date:
                    extracted_item_1_data.append({
                        'ticker': ticker,
                        'filing_date': filing_date,
                        'accession_number': accession_number,
                        'item_1_content': item_1_content
                    })

    print(f"Extracted Item 1 content from {len(extracted_item_1_data)} filings.")
    return extracted_item_1_data


# --- run ---
