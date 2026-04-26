# FIN377_Final_Project

## 10-K Sentiment Stock Predictor
This project analyzes the language used in publicly traded companies' annual reports (10-K filings) to predict the directional movement of their stock price following the filing date.
### Overview
Every public company is required to file a 10-K annual report with the SEC. Item 1 of the 10-K, the Business Description section, contains management's narrative description of the company's operations, strategy, and outlook. The language used in this section — whether optimistic or cautious — can reflect management's underlying sentiment about the company's future prospects.
This tool extracts Item 1 text from the most recent 10-K filing for any given ticker, scores it using VADER (Valence Aware Dictionary and sEntiment Reasoner), and uses the resulting sentiment score to predict whether the stock is likely to trend upward or downward in the 30 days following the filing date.

## How It Works

Data Collection — The app fetches the most recent 10-K filing directly from the SEC EDGAR database using the company's ticker symbol
Text Extraction — Item 1 of the filing is isolated and cleaned, stripping HTML formatting, tables, and boilerplate legal language
Sentiment Scoring — VADER analyzes the cleaned text and produces a compound sentiment score ranging from -1 (most negative) to +1 (most positive)
Prediction — A threshold rule converts the sentiment score into a directional signal: positive sentiment predicts upward movement, negative sentiment predicts downward movement
Validation — The model's historical accuracy is displayed alongside the prediction, computed by backtesting the same rules-based approach across a sample of S&P 500 filings from the past three years

### Tech Stack

Streamlit — dashboard and user interface
sec-edgar-downloader — fetching 10-K filings from SEC EDGAR
NLTK / VADER — sentiment analysis
yfinance — historical stock price data for backtesting
pandas / scikit-learn — data processing and validation

### Limitations
VADER is a general-purpose sentiment tool trained on social media text, not financial or legal language. Certain words common in 10-K filings may be scored incorrectly in a financial context. A natural extension of this project would be to swap VADER for the Loughran-McDonald Financial Sentiment Dictionary, which is purpose-built for SEC filings. Additionally, this tool is intended for academic and educational purposes only and should not be used as the basis for real investment decisions.


# Psuedocode
# ============================================================
# STEP 1: DEFINE SAMPLE UNIVERSE
# ============================================================

Define list of 20-30 tickers (diverse across sectors)
    → e.g. AAPL, MSFT, JPM, XOM, JNJ, WMT, BA, GS, AMZN, TSLA...

Define date range for backtest:
    → 2021 - 2023 (3 years, ~60-90 filing observations total)

Define sentiment thresholds:
    → POSITIVE_THRESHOLD = 0.05
    → NEGATIVE_THRESHOLD = -0.05

Define forward return window:
    → 30 trading days after filing date


# ============================================================
# STEP 2: FETCH AND PARSE 10-K FILINGS
# ============================================================

For each ticker in universe:
    For each year in date range:

        Use sec-edgar-downloader:
            → downloader.get("10-K", ticker, after=year-start, before=year-end)
            → Locate downloaded filing file on disk

        Parse filing text:
            → Read raw .txt or .htm file
            → Search for "Item 1" or "Item 1. Business" header
            → Extract text until "Item 1A" header is found
            → Strip all HTML tags, whitespace artifacts, page numbers
            → Store clean text

        Record filing date from document metadata

        If filing not found or Item 1 extraction fails:
            → Log error for this (ticker, year)
            → Skip and continue


# ============================================================
# STEP 3: SCORE SENTIMENT WITH VADER
# ============================================================

Initialize VADER SentimentIntensityAnalyzer

For each (ticker, filing_date, item1_text):

    Chunk text into ~500 word paragraphs:
        → Split on paragraph breaks
        → Group into chunks of ~500 words

    For each chunk:
        → scores = analyzer.polarity_scores(chunk)
        → store chunk compound score

    Final compound = average of all chunk compound scores
    Also store: pos, neg, neu averages across chunks

    Store: (ticker, filing_date, compound, pos, neg, neu)


# ============================================================
# STEP 4: APPLY THRESHOLD RULE
# ============================================================

For each (ticker, filing_date, compound):

    If compound > POSITIVE_THRESHOLD:
        → predicted_direction = "UP"
    Elif compound < NEGATIVE_THRESHOLD:
        → predicted_direction = "DOWN"
    Else:
        → predicted_direction = "NEUTRAL"

    Store predicted_direction alongside sentiment scores


# ============================================================
# STEP 5: FETCH ACTUAL FORWARD RETURNS
# ============================================================

For each (ticker, filing_date):

    Use yfinance to pull adjusted close prices:
        → Start: filing_date
        → End: filing_date + 45 calendar days (covers 30 trading days)

    Compute 30-day forward return:
        → return = (price_day30 - price_day0) / price_day0

    Fetch SPY return for same window:
        → market_return = (SPY_day30 - SPY_day0) / SPY_day0

    Compute abnormal return:
        → CAR = forward_return - market_return

    Assign actual direction:
        → If CAR > 0: actual_direction = "UP"
        → If CAR < 0: actual_direction = "DOWN"
        → If CAR == 0: drop row (extremely rare)

    Store: (ticker, filing_date, CAR, actual_direction)


# ============================================================
# STEP 6: COMPUTE BACKTEST ACCURACY
# ============================================================

Merge sentiment table and returns table on (ticker, filing_date)

For each row:
    → correct = 1 if predicted_direction == actual_direction else 0
    → Note: NEUTRAL predictions are excluded from accuracy calc
      (or counted as incorrect — pick one and document it)

Compute overall accuracy:
    → accuracy = sum(correct) / total non-neutral predictions

Compute accuracy by sector:
    → Group by sector, compute accuracy per group

Compute accuracy by direction:
    → Of all UP predictions, how many were correct?
    → Of all DOWN predictions, how many were correct?

Store summary stats:
    → overall_accuracy, up_accuracy, down_accuracy, n_observations

Save full backtest results to CSV:
    → backtest_results.csv
    → Columns: [ticker, filing_date, compound, pos, neg, neu,
                predicted_direction, actual_direction, CAR, correct]

Save summary stats to JSON:
    → backtest_summary.json
    → { overall_accuracy, up_accuracy, down_accuracy, n_observations }


# ============================================================
# THESE TWO FILES ARE ALL THE DASHBOARD NEEDS AT RUNTIME
# backtest_results.csv
# backtest_summary.json
# ============================================================

Phase 2: Streamlit Dashboard (live inference)
# ============================================================
# STEP 1: APP STARTUP
# ============================================================

Import: streamlit, sec-edgar-downloader, nltk vader,
        yfinance, pandas, json

Initialize VADER SentimentIntensityAnalyzer

Load pre-computed files:
    → backtest_results = pd.read_csv('backtest_results.csv')
    → backtest_summary = json.load('backtest_summary.json')

Define thresholds (must match Phase 1):
    → POSITIVE_THRESHOLD = 0.05
    → NEGATIVE_THRESHOLD = -0.05


# ============================================================
# STEP 2: RENDER STATIC UI ELEMENTS
# ============================================================

Display:
    → App title: "10-K Sentiment Stock Predictor"
    → One-line description
    → Sidebar: "How it works" expander with brief methodology note
    → Sidebar: overall backtest accuracy stat
        e.g. "Historically accurate on X% of S&P 500 filings (2021-2023)"


# ============================================================
# STEP 3: USER INPUT
# ============================================================

Render input section:
    → st.text_input: "Enter a ticker symbol"
    → st.button: "Analyze"

On button click:
    → Uppercase and strip whitespace from ticker input
    → Validate ticker via yfinance:
        ticker_info = yf.Ticker(input).info
        If 'shortName' not in ticker_info:
            → st.error("Ticker not found. Please check and try again.")
            → Stop execution

    → Display: "Analyzing {ticker} — {company name}..."
    → Show loading spinner while steps 4-7 run


# ============================================================
# STEP 4: FETCH LATEST 10-K
# ============================================================

Use sec-edgar-downloader:
    → downloader.get("10-K", ticker, limit=1)
    → Locate file on disk

Parse filing:
    → Read raw file
    → Search for "Item 1" / "Item 1. Business" header
    → Extract text until "Item 1A" header
    → Strip HTML, tables, boilerplate

Extract filing date from document metadata

If fetch or parse fails:
    → st.error("Could not retrieve 10-K for this ticker.")
    → Suggest user try again or check SEC EDGAR manually
    → Stop execution

Display:
    → "10-K filing found — filed on {filing_date}"
    → Expandable preview: first 300 words of Item 1 text


# ============================================================
# STEP 5: SCORE SENTIMENT
# ============================================================

Chunk Item 1 text into ~500 word paragraphs
Score each chunk with VADER
Compute averages: compound, pos, neg, neu

Display sentiment breakdown:
    → Compound score with colored gauge or metric
        Green if compound > POSITIVE_THRESHOLD
        Red if compound < NEGATIVE_THRESHOLD
        Gray if neutral
    → pos / neg / neu percentages as a horizontal bar chart


# ============================================================
# STEP 6: APPLY THRESHOLD RULE AND PREDICT
# ============================================================

If compound > POSITIVE_THRESHOLD:
    → prediction = "▲ UP"
    → color = green
    → confidence = maps compound score to a 50-100% scale
        e.g. confidence = 50 + (compound / 1.0) * 50

Elif compound < NEGATIVE_THRESHOLD:
    → prediction = "▼ DOWN"
    → color = red
    → confidence = 50 + (abs(compound) / 1.0) * 50

Else:
    → prediction = "→ NEUTRAL"
    → color = gray
    → confidence = None
    → Display note: "Sentiment too weak to make a directional call"


# ============================================================
# STEP 7: DISPLAY RESULTS
# ============================================================

Main result panel:
    → Large colored prediction label (UP / DOWN / NEUTRAL)
    → Confidence percentage if not neutral
    → st.progress bar showing confidence

Sentiment detail panel:
    → Compound score: {value}
    → Positive language: {pos}%
    → Negative language: {neg}%
    → Neutral language: {neu}%

Historical context panel:
    → Check if ticker exists in backtest_results.csv
    → If yes: plot compound score over past filings as line chart
              overlay actual stock return for each year
    → If no: show message "No historical backtest data for this ticker"

Backtest stats panel:
    → Overall accuracy: {overall_accuracy}%
    → Based on {n_observations} filings from 2021-2023
    → Accuracy on UP calls: {up_accuracy}%
    → Accuracy on DOWN calls: {down_accuracy}%


# ============================================================
# STEP 8: FOOTER
# ============================================================

Display disclaimer:
    → "This application is an academic project built for educational
       purposes. Predictions are based solely on textual sentiment
       analysis of SEC filings and do not constitute financial advice.