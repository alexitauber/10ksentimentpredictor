import json
from pathlib import Path
import sys

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.fin377_project.config import BACKTEST_RESULTS_PATH, BACKTEST_SUMMARY_PATH
from src.fin377_project.pipeline import analyze_latest_filing_for_ticker, run_ticker_backtest
from src.fin377_project.sentiment import NEGATIVE_THRESHOLD, POSITIVE_THRESHOLD


st.set_page_config(
    page_title="10-K Sentiment Stock Predictor",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_backtest_results() -> pd.DataFrame:
    if not BACKTEST_RESULTS_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(BACKTEST_RESULTS_PATH)


@st.cache_data(show_spinner=False)
def load_backtest_summary() -> dict:
    if not BACKTEST_SUMMARY_PATH.exists():
        return {}
    with open(BACKTEST_SUMMARY_PATH, "r", encoding="utf-8") as summary_file:
        return json.load(summary_file)


def render_prediction_banner(direction: str, confidence):
    if direction == "UP":
        label = "UP"
        color = "#15803d"
    elif direction == "DOWN":
        label = "DOWN"
        color = "#b91c1c"
    else:
        label = "NEUTRAL"
        color = "#475569"

    st.markdown(
        f"""
        <div style="padding: 20px 24px; border: 1px solid #dbe4ee; border-radius: 8px; background: #f8fafc;">
            <div style="font-size: 12px; text-transform: uppercase; color: #64748b; letter-spacing: 0;">Projection</div>
            <div style="font-size: 40px; font-weight: 700; color: {color}; margin-top: 8px;">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if confidence is not None:
        st.metric("Confidence", f"{confidence}%")
        st.progress(confidence / 100)
    else:
        st.caption("Sentiment is too weak to make a directional call.")


def render_sentiment_bars(result: dict):
    sentiment_pairs = [
        ("Positive language", result["pos"]),
        ("Negative language", result["neg"]),
        ("Neutral language", result["neu"]),
    ]

    for label, value in sentiment_pairs:
        st.write(f"{label}: {value:.1%}")
        st.progress(float(value))


def render_historical_context(backtest_results: pd.DataFrame, ticker: str):
    ticker_history = backtest_results[backtest_results["ticker"] == ticker].copy()
    if ticker_history.empty:
        st.info("No historical backtest data for this ticker yet.")
        return

    ticker_history["filing_date"] = pd.to_datetime(ticker_history["filing_date"])
    ticker_history = ticker_history.sort_values("filing_date")
    chart_df = ticker_history.set_index("filing_date")[["compound", "CAR"]]

    st.line_chart(chart_df)
    st.dataframe(
        ticker_history[
            ["filing_date", "compound", "predicted_direction", "actual_direction", "CAR", "correct"]
        ].assign(filing_date=lambda df: df["filing_date"].dt.strftime("%Y-%m-%d")),
        use_container_width=True,
        hide_index=True,
    )


def get_ticker_backtest_context(saved_backtest_results: pd.DataFrame, ticker: str):
    ticker = ticker.strip().upper()
    saved_ticker_results = saved_backtest_results[saved_backtest_results["ticker"] == ticker].copy()
    if not saved_ticker_results.empty:
        n_obs = len(saved_ticker_results)
        summary = {
            "overall_accuracy": float(saved_ticker_results["correct"].mean()) if n_obs else 0.0,
            "up_accuracy": float(
                saved_ticker_results.loc[saved_ticker_results["predicted_direction"] == "UP", "correct"].mean()
            )
            if (saved_ticker_results["predicted_direction"] == "UP").any()
            else 0.0,
            "down_accuracy": float(
                saved_ticker_results.loc[saved_ticker_results["predicted_direction"] == "DOWN", "correct"].mean()
            )
            if (saved_ticker_results["predicted_direction"] == "DOWN").any()
            else 0.0,
            "n_observations": int(n_obs),
        }
        return saved_ticker_results, summary, False

    ticker_results, ticker_summary = run_ticker_backtest(ticker)[1:]
    return ticker_results, ticker_summary, True


def main():
    backtest_results = load_backtest_results()
    backtest_summary = load_backtest_summary()

    st.title("10-K Sentiment Stock Predictor")
    st.write(
        "Analyze the latest Item 1 section from a company's 10-K and turn its language into a directional stock projection."
    )

    with st.sidebar:
        st.subheader("Backtest Snapshot")
        if backtest_summary:
            st.metric("Overall accuracy", f"{backtest_summary.get('overall_accuracy', 0):.1%}")
            st.caption(f"Based on {backtest_summary.get('n_observations', 0)} filings")
            st.write(f"UP accuracy: {backtest_summary.get('up_accuracy', 0):.1%}")
            st.write(f"DOWN accuracy: {backtest_summary.get('down_accuracy', 0):.1%}")
        else:
            st.warning("Run `main.py` first to generate the historical backtest files.")

        with st.expander("How it works", expanded=False):
            st.write(
                "The app downloads the latest 10-K, extracts Item 1, scores the text with the "
                "Loughran-McDonald financial dictionary, applies sentiment thresholds, and compares "
                "the live projection against the historical backtest data when available."
            )
            st.write(f"Positive threshold: {POSITIVE_THRESHOLD}")
            st.write(f"Negative threshold: {NEGATIVE_THRESHOLD}")

    ticker_input = st.text_input("Enter a ticker symbol", value="CVX")
    analyze_clicked = st.button("Analyze", type="primary")

    if not analyze_clicked:
        return

    ticker = ticker_input.strip().upper()
    if not ticker:
        st.error("Please enter a ticker symbol.")
        return

    try:
        with st.spinner(f"Analyzing {ticker}..."):
            result = analyze_latest_filing_for_ticker(ticker)
            ticker_backtest_results, ticker_backtest_summary, generated_for_ticker = get_ticker_backtest_context(
                backtest_results,
                ticker,
            )
    except Exception as error:
        st.error(f"Could not analyze {ticker}: {error}")
        return

    st.success(f"10-K filing found for {result['company_name']} and filed on {result['filing_date']}.")
    if generated_for_ticker:
        st.info(f"Generated a fresh backtest for {ticker} from its downloaded filings because it was not already in the saved history.")

    with st.expander("Item 1 preview", expanded=False):
        st.write(result["preview_text"])

    left_col, right_col = st.columns([1.1, 0.9])

    with left_col:
        render_prediction_banner(result["predicted_direction"], result["confidence"])

        st.subheader("Sentiment Detail")
        detail_cols = st.columns(4)
        detail_cols[0].metric("Compound", f"{result['compound']:.3f}")
        detail_cols[1].metric("Positive", f"{result['pos']:.1%}")
        detail_cols[2].metric("Negative", f"{result['neg']:.1%}")
        detail_cols[3].metric("Neutral", f"{result['neu']:.1%}")
        render_sentiment_bars(result)

    with right_col:
        st.subheader("Backtest Stats")
        if ticker_backtest_summary:
            stats_cols = st.columns(2)
            stats_cols[0].metric("Overall accuracy", f"{ticker_backtest_summary.get('overall_accuracy', 0):.1%}")
            stats_cols[1].metric("Observations", ticker_backtest_summary.get("n_observations", 0))
            stats_cols[0].metric("UP calls", f"{ticker_backtest_summary.get('up_accuracy', 0):.1%}")
            stats_cols[1].metric("DOWN calls", f"{ticker_backtest_summary.get('down_accuracy', 0):.1%}")
        else:
            st.info("Historical backtest stats are not available yet.")

        st.subheader("Latest Filing")
        st.write(f"Matched dictionary words: {result['matched_word_count']}")
        st.write(f"Positive words: {result['positive_word_count']}")
        st.write(f"Negative words: {result['negative_word_count']}")
        if result["matched_words_preview"]:
            st.caption(f"Example matched words: {result['matched_words_preview']}")

    st.subheader("Historical Context")
    render_historical_context(ticker_backtest_results, ticker)

    st.caption(
        "This application is an academic project built for educational purposes. "
        "Predictions are based solely on textual sentiment analysis of SEC filings and do not constitute financial advice."
    )


if __name__ == "__main__":
    main()
