from Pull_Item_1 import download_filings, extract_item1

my_firm = ["CVX"]
competitors = []
tickers = my_firm + competitors

download_filings(tickers, start_date="2021-01-01", limit=3)
extracted_item_1_data = extract_item1()

import pandas as pd

pd.DataFrame(extracted_item_1_data).to_csv('data/item1_extracted.csv', index=False)