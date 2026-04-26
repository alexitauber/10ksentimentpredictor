import os
import re
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


def _extract_item_1_from_text(text: str) -> str:
    normalized_text = text.replace("\xa0", " ")
    normalized_text = re.sub(r"\s+", " ", normalized_text).strip()

    best_section = ""

    for start_match in ITEM_1_START_PATTERN.finditer(normalized_text):
        end_match = ITEM_1_END_PATTERN.search(normalized_text, start_match.end())
        if not end_match:
            continue

        candidate = normalized_text[start_match.start():end_match.start()].strip()
        candidate = re.sub(r"\s+", " ", candidate)

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

                with open(file_path, 'r', encoding='utf-8') as f:
                    html_content = f.read()

                soup = BeautifulSoup(html_content, 'html.parser')
                text = soup.get_text(separator=' ')
                item_1_content = _extract_item_1_from_text(text)

                if item_1_content:
                    extracted_item_1_data.append({
                        'ticker': ticker,
                        'accession_number': accession_number,
                        'item_1_content': item_1_content
                    })

    print(f"Extracted Item 1 content from {len(extracted_item_1_data)} filings.")
    return extracted_item_1_data


# --- run ---
