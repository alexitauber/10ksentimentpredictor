import json
from pathlib import Path
import sys

import altair as alt
import pandas as pd
import streamlit as st
import yfinance as yf

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


def apply_theme():
    st.markdown(
        """
        <style>
            .stApp {
                background: #ffffff;
                color: #0f172a;
            }
            .block-container {
                padding-top: 2rem;
                padding-bottom: 2rem;
                max-width: 1180px;
            }
            h1, h2, h3, p, label, .stMarkdown, .stCaption, .stMetricLabel, .stMetricValue {
                color: #0f172a !important;
            }
            [data-testid="stSidebar"] {
                background: #f8fafc;
                border-right: 1px solid #e2e8f0;
                display: none;
            }
            [data-testid="stTextInputRootElement"] input {
                background: #ffffff;
                color: #0f172a;
                border: 1px solid #cbd5e1;
            }
            [data-testid="stDataFrame"] {
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                overflow: hidden;
            }
            div[data-testid="stMetric"] {
                background: #ffffff;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                padding: 0.9rem 1rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
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
        <div style="padding: 20px 24px; border: 1px solid #e2e8f0; border-radius: 8px; background: #ffffff;">
            <div style="font-size: 12px; text-transform: uppercase; color: #475569; letter-spacing: 0.04em;">Projection</div>
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


def get_directional_sentiment_mix(result: dict) -> dict[str, float]:
    positive_count = result.get("positive_word_count", 0)
    negative_count = result.get("negative_word_count", 0)
    matched_count = positive_count + negative_count

    if matched_count == 0:
        return {"positive": 0.0, "negative": 0.0}

    return {
        "positive": positive_count / matched_count,
        "negative": negative_count / matched_count,
    }


def render_sentiment_bars(result: dict):
    sentiment_mix = get_directional_sentiment_mix(result)
    sentiment_pairs = [
        ("Positive language", sentiment_mix["positive"], "#15803d"),
        ("Negative language", sentiment_mix["negative"], "#b91c1c"),
    ]

    for label, value, color in sentiment_pairs:
        st.markdown(
            f"""
            <div style="margin: 0.85rem 0 0.35rem 0; display: flex; justify-content: space-between; font-size: 0.95rem;">
                <span style="color: #0f172a;">{label}</span>
                <span style="color: #334155; font-weight: 600;">{value:.1%}</span>
            </div>
            <div style="width: 100%; height: 10px; background: #e2e8f0; border-radius: 999px; overflow: hidden;">
                <div style="width: {value * 100:.1f}%; height: 100%; background: {color};"></div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_compact_metric(label: str, value: str | int):
    st.markdown(
        f"""
        <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 0.9rem 1rem;">
            <div style="font-size: 0.82rem; color: #475569; margin-bottom: 0.3rem;">{label}</div>
            <div style="font-size: 1.35rem; font-weight: 600; color: #0f172a; line-height: 1.15;">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def prepare_ticker_history(backtest_results: pd.DataFrame, ticker: str) -> pd.DataFrame:
    ticker_history = backtest_results[backtest_results["ticker"] == ticker].copy()
    if ticker_history.empty:
        return ticker_history

    ticker_history["filing_date"] = pd.to_datetime(ticker_history["filing_date"])
    dedupe_columns = [column for column in ["accession_number", "filing_date", "predicted_direction", "CAR"] if column in ticker_history.columns]
    if dedupe_columns:
        ticker_history = ticker_history.drop_duplicates(subset=dedupe_columns)

    return ticker_history.sort_values("filing_date")


@st.cache_data(show_spinner=False)
def load_price_history(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    price_history = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        interval="1d",
        auto_adjust=True,
        progress=False,
    )
    if price_history is None or price_history.empty:
        return pd.DataFrame()

    if isinstance(price_history.columns, pd.MultiIndex):
        close_frame = price_history["Close"].copy()
        if ticker in close_frame.columns:
            close_series = close_frame[ticker]
        else:
            close_series = close_frame.iloc[:, 0]
    else:
        close_series = price_history["Close"]

    daily_history = (
        close_series.dropna()
        .rename("close")
        .reset_index()
        .rename(columns={"Date": "date"})
        .sort_values("date")
    )
    daily_history["daily_return"] = daily_history["close"].pct_change()
    daily_history["daily_return_label"] = daily_history["daily_return"].map(
        lambda value: "N/A" if pd.isna(value) else f"{value:.2%}"
    )
    return daily_history


def render_price_prediction_chart(ticker_history: pd.DataFrame, ticker: str):
    if ticker_history.empty:
        st.info("No historical backtest data for this ticker yet.")
        return

    start_date = (ticker_history["filing_date"].min() - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
    end_date = (ticker_history["filing_date"].max() + pd.Timedelta(days=75)).strftime("%Y-%m-%d")
    price_history = load_price_history(ticker, start_date, end_date)

    if price_history.empty:
        st.info("Price history is not available for this ticker right now.")
        return

    price_lookup = price_history.set_index("date")["close"]
    trading_dates = price_lookup.index

    filing_markers = ticker_history[["filing_date", "predicted_direction", "actual_direction", "correct", "compound", "CAR"]].copy()
    filing_markers["date"] = filing_markers["filing_date"].apply(
        lambda filing_date: trading_dates[trading_dates.searchsorted(filing_date, side="left")]
        if trading_dates.searchsorted(filing_date, side="left") < len(trading_dates)
        else pd.NaT
    )
    filing_markers = filing_markers.dropna(subset=["date"]).copy()
    filing_markers["price"] = filing_markers["date"].map(price_lookup)
    filing_markers["status"] = filing_markers["correct"].map({1: "Correct", 0: "Incorrect"})

    price_line = (
        alt.Chart(price_history)
        .mark_line(color="#2563eb", strokeWidth=3)
        .encode(
            x=alt.X(
                "date:T",
                title="Date",
                axis=alt.Axis(
                    format="%Y",
                    labelAngle=0,
                    tickCount=alt.TimeIntervalStep(interval="year", step=1),
                ),
            ),
            y=alt.Y("close:Q", title="Adjusted Close Price"),
            tooltip=[
                alt.Tooltip("date:T", title="Date"),
                alt.Tooltip("close:Q", title="Close", format=".2f"),
                alt.Tooltip("daily_return_label:N", title="Daily return"),
            ],
        )
    )

    daily_points = (
        alt.Chart(price_history)
        .mark_circle(color="#2563eb", opacity=0.35, size=28)
        .encode(
            x=alt.X(
                "date:T",
                title="Date",
                axis=alt.Axis(
                    format="%Y",
                    labelAngle=0,
                    tickCount=alt.TimeIntervalStep(interval="year", step=1),
                ),
            ),
            y=alt.Y("close:Q", title="Adjusted Close Price"),
            tooltip=[
                alt.Tooltip("date:T", title="Date"),
                alt.Tooltip("close:Q", title="Close", format=".2f"),
                alt.Tooltip("daily_return_label:N", title="Daily return"),
            ],
        )
    )

    filing_rules = (
        alt.Chart(filing_markers)
        .mark_rule(strokeDash=[4, 4], strokeWidth=1.5)
        .encode(
            x="date:T",
            color=alt.Color(
                "predicted_direction:N",
                scale=alt.Scale(domain=["UP", "DOWN", "NEUTRAL"], range=["#15803d", "#b91c1c", "#64748b"]),
                title="Prediction",
            ),
            tooltip=[
                alt.Tooltip("filing_date:T", title="10-K date"),
                alt.Tooltip("predicted_direction:N", title="Predicted"),
                alt.Tooltip("actual_direction:N", title="Actual"),
                alt.Tooltip("compound:Q", title="Positivity ratio", format=".3f"),
                alt.Tooltip("CAR:Q", title="30-day CAR", format=".2%"),
                alt.Tooltip("status:N", title="Accuracy"),
            ],
        )
    )

    signal_points = (
        alt.Chart(filing_markers)
        .mark_point(filled=True, size=240)
        .encode(
            x="date:T",
            y="price:Q",
            shape=alt.Shape(
                "predicted_direction:N",
                scale=alt.Scale(domain=["UP", "DOWN", "NEUTRAL"], range=["triangle-up", "triangle-down", "circle"]),
                title="Prediction",
            ),
            color=alt.Color(
                "predicted_direction:N",
                scale=alt.Scale(domain=["UP", "DOWN", "NEUTRAL"], range=["#15803d", "#b91c1c", "#64748b"]),
                title="Prediction",
            ),
            tooltip=[
                alt.Tooltip("filing_date:T", title="10-K date"),
                alt.Tooltip("price:Q", title="Price on filing", format=".2f"),
                alt.Tooltip("predicted_direction:N", title="Predicted"),
                alt.Tooltip("actual_direction:N", title="Actual"),
                alt.Tooltip("compound:Q", title="Positivity ratio", format=".3f"),
                alt.Tooltip("CAR:Q", title="30-day CAR", format=".2%"),
                alt.Tooltip("status:N", title="Accuracy"),
            ],
        )
    )

    st.altair_chart((price_line + daily_points + filing_rules + signal_points).properties(height=420), use_container_width=True)
    st.caption("The chart now shows daily Yahoo Finance adjusted close prices. Hover any daily point to see the stock price and that day's return, while the filing markers show whether the model called UP or DOWN.")


def render_event_window_chart(ticker_history: pd.DataFrame, ticker: str):
    if ticker_history.empty or "filing_date" not in ticker_history.columns:
        st.info("Not enough price data is available to show market reaction around filing dates.")
        return

    ticker_history = ticker_history.copy()
    ticker_history["filing_date"] = pd.to_datetime(ticker_history["filing_date"], errors="coerce", utc=True).dt.tz_convert(None)
    ticker_history = ticker_history.dropna(subset=["filing_date"]).sort_values("filing_date")
    if ticker_history.empty:
        st.info("Not enough price data is available to show market reaction around filing dates.")
        return

    start_date = (ticker_history["filing_date"].min() - pd.Timedelta(days=45)).strftime("%Y-%m-%d")
    end_date = (ticker_history["filing_date"].max() + pd.Timedelta(days=45)).strftime("%Y-%m-%d")
    stock_prices = load_price_history(ticker, start_date, end_date)
    spy_prices = load_price_history("SPY", start_date, end_date)

    if stock_prices.empty or spy_prices.empty:
        st.info("Not enough price data is available to show market reaction around filing dates.")
        return

    stock_prices = stock_prices.copy()
    spy_prices = spy_prices.copy()
    stock_prices["date"] = pd.to_datetime(stock_prices["date"], errors="coerce", utc=True).dt.tz_convert(None).dt.normalize()
    spy_prices["date"] = pd.to_datetime(spy_prices["date"], errors="coerce", utc=True).dt.tz_convert(None).dt.normalize()
    stock_prices["close"] = pd.to_numeric(stock_prices["close"], errors="coerce")
    spy_prices["close"] = pd.to_numeric(spy_prices["close"], errors="coerce")

    stock_lookup = stock_prices.dropna(subset=["date", "close"]).groupby("date")["close"].last().sort_index()
    spy_lookup = spy_prices.dropna(subset=["date", "close"]).groupby("date")["close"].last().sort_index()
    common_trading_dates = stock_lookup.index.intersection(spy_lookup.index).sort_values()

    event_rows = []
    for filing_number, (_, filing) in enumerate(ticker_history.iterrows(), start=1):
        filing_date = pd.Timestamp(filing["filing_date"]).normalize()
        event_position = common_trading_dates.searchsorted(filing_date, side="left")

        # If the filing was not on a trading day, the next shared trading day becomes day 0.
        if event_position >= len(common_trading_dates):
            continue

        window_start = event_position - 5
        window_end = event_position + 6
        if window_start < 0 or window_end > len(common_trading_dates):
            continue

        window_dates = common_trading_dates[window_start:window_end]
        stock_day0_price = stock_lookup.loc[common_trading_dates[event_position]]
        spy_day0_price = spy_lookup.loc[common_trading_dates[event_position]]

        if pd.isna(stock_day0_price) or pd.isna(spy_day0_price) or stock_day0_price == 0 or spy_day0_price == 0:
            continue

        stock_window_prices = stock_lookup.loc[window_dates]
        spy_window_prices = spy_lookup.loc[window_dates]
        if len(stock_window_prices) != 11 or len(spy_window_prices) != 11:
            continue
        if stock_window_prices.isna().any() or spy_window_prices.isna().any():
            continue

        for event_day, trading_date in zip(range(-5, 6), window_dates):
            stock_return = (stock_window_prices.loc[trading_date] / stock_day0_price) - 1
            spy_return = (spy_window_prices.loc[trading_date] / spy_day0_price) - 1

            event_rows.append(
                {
                    "ticker": ticker,
                    "filing_date": filing_date,
                    "filing_number": filing_number,
                    "event_day": event_day,
                    "stock_return": stock_return,
                    "spy_return": spy_return,
                    "abnormal_return": stock_return - spy_return,
                    "predicted_direction": filing.get("predicted_direction"),
                    "actual_direction": filing.get("actual_direction"),
                    "correct": filing.get("correct"),
                }
            )

    event_window = pd.DataFrame(event_rows)
    if event_window.empty:
        st.info("Not enough price data is available to show market reaction around filing dates.")
        return

    event_window["filing_date_label"] = event_window["filing_date"].dt.strftime("%Y-%m-%d")
    event_window["correct_label"] = event_window["correct"].map({1: "Yes", 0: "No"}).fillna("N/A")
    for numeric_column in ["event_day", "stock_return", "spy_return", "abnormal_return"]:
        event_window[numeric_column] = pd.to_numeric(event_window[numeric_column], errors="coerce")

    event_window = event_window.dropna(subset=["event_day", "stock_return", "spy_return", "abnormal_return"])
    if event_window.empty:
        st.info("Not enough price data is available to show market reaction around filing dates.")
        return

    chart_data = event_window.melt(
        id_vars=[
            "ticker",
            "filing_date",
            "filing_date_label",
            "filing_number",
            "event_day",
            "abnormal_return",
            "predicted_direction",
            "actual_direction",
            "correct_label",
        ],
        value_vars=["stock_return", "spy_return"],
        var_name="series",
        value_name="return",
    )
    chart_data["series"] = chart_data["series"].map(
        {
            "stock_return": "Stock return",
            "spy_return": "SPY return",
        }
    )
    chart_data["return"] = pd.to_numeric(chart_data["return"], errors="coerce")
    chart_data = chart_data.dropna(subset=["event_day", "return", "series"])

    if chart_data.empty:
        st.info("Not enough price data is available to show market reaction around filing dates.")
        return

    st.write(
        "Each chart shows how the stock moved relative to SPY in the five trading days before and after a 10-K filing."
    )

    for filing_date_label, filing_chart_data in chart_data.groupby("filing_date_label", sort=True):
        filing_chart_data = filing_chart_data.sort_values(["series", "event_day"])
        filing_details = filing_chart_data.iloc[0]
        predicted_direction = filing_details.get("predicted_direction")
        actual_direction = filing_details.get("actual_direction")
        if pd.isna(predicted_direction):
            predicted_direction = "N/A"
        if pd.isna(actual_direction):
            actual_direction = "N/A"
        expander_title = (
            f"Filing date: {filing_date_label} | "
            f"Predicted: {predicted_direction} | Actual: {actual_direction}"
        )

        with st.expander(expander_title, expanded=False):
            return_lines = (
                alt.Chart(filing_chart_data)
                .mark_line(point=True, strokeWidth=2.5)
                .encode(
                    x=alt.X(
                        "event_day:Q",
                        title="Trading Days From 10-K Filing",
                        scale=alt.Scale(domain=[-5, 5]),
                        axis=alt.Axis(values=list(range(-5, 6)), labelAngle=0),
                    ),
                    y=alt.Y("return:Q", title="Return From Event Day 0", axis=alt.Axis(format="%")),
                    color=alt.Color(
                        "series:N",
                        title="Series",
                        scale=alt.Scale(domain=["Stock return", "SPY return"], range=["#2563eb", "#64748b"]),
                    ),
                    tooltip=[
                        alt.Tooltip("filing_date:T", title="Filing date"),
                        alt.Tooltip("event_day:Q", title="Event day"),
                        alt.Tooltip("series:N", title="Series"),
                        alt.Tooltip("return:Q", title="Return", format=".2%"),
                        alt.Tooltip("abnormal_return:Q", title="Abnormal return", format=".2%"),
                        alt.Tooltip("predicted_direction:N", title="Predicted"),
                        alt.Tooltip("actual_direction:N", title="Actual"),
                        alt.Tooltip("correct_label:N", title="Correct"),
                    ],
                )
            )

            event_day_rule = (
                alt.Chart(pd.DataFrame({"event_day": [0]}))
                .mark_rule(color="#0f172a", strokeDash=[5, 5], strokeWidth=1.5)
                .encode(x="event_day:Q")
            )

            st.altair_chart(
                (return_lines + event_day_rule).properties(
                    title="Market Reaction Around 10-K Filing Date",
                    height=340,
                ),
                use_container_width=True,
            )

    st.caption("Returns are normalized to the first shared trading day on or after each 10-K filing date. Filings without a full five-trading-day window are skipped.")

    with st.expander("Debug event-window data", expanded=False):
        st.write(f"Rows generated: {len(event_window)}")
        st.dataframe(event_window.head(), use_container_width=True, hide_index=True)


def render_historical_context(backtest_results: pd.DataFrame, ticker: str):
    ticker_history = prepare_ticker_history(backtest_results, ticker)
    if ticker_history.empty:
        st.info("No historical backtest data for this ticker yet.")
        return

    st.subheader("Price and Prediction Tracking")
    render_price_prediction_chart(ticker_history, ticker)

    st.subheader("Market Reaction Around 10-K Filing Dates")
    render_event_window_chart(ticker_history, ticker)

    st.subheader("Historical Records")
    st.dataframe(
        ticker_history[
            ["filing_date", "compound", "predicted_direction", "actual_direction", "CAR", "correct"]
        ].assign(
            filing_date=lambda df: df["filing_date"].dt.strftime("%Y-%m-%d"),
            compound=lambda df: df["compound"].map(lambda value: f"{value:.3f}"),
            CAR=lambda df: df["CAR"].map(lambda value: f"{value:.2%}"),
            correct=lambda df: df["correct"].map({1: "Yes", 0: "No"}),
        ).rename(columns={"compound": "Positivity ratio"}),
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
    apply_theme()
    backtest_results = load_backtest_results()
    st.title("10-K Sentiment Stock Predictor")
    st.write(
        "Analyze the latest Item 1 section from a company's 10-K and turn its language into a directional stock projection."
    )

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
        st.text_area("Full Item 1 text", value=result["item_1_content"], height=500)

    left_col, right_col = st.columns([1.1, 0.9])

    with left_col:
        render_prediction_banner(result["predicted_direction"], result["confidence"])

        st.subheader("Sentiment Detail")
        detail_cols = st.columns(4)
        directional_mix = get_directional_sentiment_mix(result)
        with detail_cols[0]:
            render_compact_metric("Positivity ratio", f"{result['compound']:.3f}")
        with detail_cols[1]:
            render_compact_metric("Positive", f"{directional_mix['positive'] * 100:.1f}%")
        with detail_cols[2]:
            render_compact_metric("Negative", f"{directional_mix['negative'] * 100:.1f}%")
        with detail_cols[3]:
            render_compact_metric("Matched words", result["matched_word_count"])
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
