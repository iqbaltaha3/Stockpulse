import streamlit as st
st.set_page_config(page_title="StockPulse", page_icon="📊", layout="wide")

import pandas as pd
import requests
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import plotly.graph_objects as go
import plotly.express as px
from bs4 import BeautifulSoup
import numpy as np
import time

# Finnhub API key (replace with your own key)
finnhub_api_key = "cusrje1r01qnihs8c29gcusrje1r01qnihs8c2a0"

# ------------------------------
# Download and Initialize NLTK Data
# ------------------------------
nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize

sia = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))

# ------------------------------
# Custom CSS for a Modern Look
# ------------------------------
st.markdown("""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');
      body {
          background: linear-gradient(135deg, #e0f7fa, #e8f5e9);
          font-family: 'Roboto', sans-serif;
          color: #424242;
      }
      .header {
          text-align: center;
          color: #1b5e20;
          font-size: 3rem;
          font-weight: 700;
          margin-top: 20px;
      }
      .subheader {
          text-align: center;
          color: #2e7d32;
          font-size: 1.75rem;
          margin-top: 0;
      }
      .landing {
          font-size: 1.1rem;
          max-width: 800px;
          margin: 20px auto;
          text-align: justify;
          background-color: rgba(255, 255, 255, 0.8);
          padding: 20px;
          border-radius: 10px;
          box-shadow: 0 4px 8px rgba(0,0,0,0.1);
      }
      .read-more-btn {
          background-color: #2e7d32;
          color: white;
          border: none;
          border-radius: 5px;
          padding: 8px 12px;
          text-decoration: none;
          font-size: 0.9rem;
          transition: background-color 0.3s ease;
      }
      .read-more-btn:hover {
          background-color: #1b5e20;
      }
      table {
          width: 100%;
          border-collapse: collapse;
          margin: 20px 0;
      }
      th, td {
          padding: 12px 16px;
          border: 1px solid #ddd;
          text-align: left;
      }
      th {
          background-color: #c8e6c9;
      }
    </style>
""", unsafe_allow_html=True)

# ------------------------------
# Landing Page – Introduction
# ------------------------------
st.markdown("""
    <div class="header">StockPulse</div>
    <div class="subheader">Live Fundamental, Technical & Sentiment Analysis</div>
    <div class="landing">
      Welcome to <strong>StockPulse</strong> – a tool for retail investors.
      We fetch live fundamental and technical data using Finnhub’s API and display current news sentiment from scraped articles.
      <br><br>
    </div>
""", unsafe_allow_html=True)

# ------------------------------
# News Scraping & Sentiment Functions
# ------------------------------
def _fetch_article_text(url):
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            paragraphs = soup.find_all('p')
            return " ".join(p.get_text() for p in paragraphs).strip()
    except Exception:
        return ""
    return ""

def _scrape_full_articles(stock, max_articles=10):
    sources = [
        f"https://www.moneycontrol.com/news/tags/{stock.lower().replace(' ', '-')}.html",
        f"https://www.businessinsider.in/searchresult.cms?query={stock}",
        f"https://www.reuters.com/search/news?blob={stock.replace(' ', '+')}",
        f"https://economictimes.indiatimes.com/topic/{stock}",
        f"https://www.livemint.com/Search/Link/Keyword/{stock}",
        f"https://www.business-standard.com/search?type=news&q={stock}",
        f"https://www.ndtv.com/search?query={stock}",
        f"https://www.cnbctv18.com/search/?q={stock}",
        f"https://www.bloombergquint.com/search?q={stock}",
        f"https://www.thehindubusinessline.com/search/?q={stock}",
        f"https://www.financialexpress.com/search/?q={stock}",
        f"https://www.zeebiz.com/search?q={stock}",
        f"https://www.outlookindia.com/newsearch?query={stock}",
        f"https://www.businesstoday.in/search?query={stock}",
        f"https://indianexpress.com/search/?s={stock}",
        f"https://www.firstpost.com/search?q={stock}",
        f"https://www.fortuneindia.com/search?query={stock}",
        f"https://www.businessworld.in/search/?q={stock}",
        f"https://www.indiainfoline.com/search?q={stock}",
        f"https://www.equitymaster.com/news/?q={stock}"
    ]
    headers = {'User-Agent': 'Mozilla/5.0'}
    articles = []
    for url in sources:
        try:
            response = requests.get(url, headers=headers, timeout=5)
            soup = BeautifulSoup(response.text, 'html.parser')
            count = 0
            for item in soup.find_all('a', href=True):
                title = item.get_text().strip()
                if not title:
                    continue
                link = item['href'] if item['href'].startswith('http') else url + item['href']
                website = url.split('/')[2]
                if stock.lower() in title.lower():
                    article_text = _fetch_article_text(link)
                    if article_text and len(article_text) > 100:
                        article_date = "N/A"
                        try:
                            page_resp = requests.get(link, headers=headers, timeout=5)
                            if page_resp.status_code == 200:
                                page_soup = BeautifulSoup(page_resp.text, 'html.parser')
                                time_tag = page_soup.find('time')
                                if time_tag:
                                    article_date = time_tag.get('datetime', time_tag.get_text().strip())
                            else:
                                article_date = "N/A"
                        except Exception:
                            article_date = "N/A"
                        articles.append({
                            "Title": title,
                            "Website": website,
                            "Link": link,
                            "Date": article_date,
                            "Article": article_text
                        })
                        count += 1
                    if count >= max_articles:
                        break
        except Exception:
            continue
    return pd.DataFrame(articles)

def _analyze_full_article_sentiment(df):
    pos_list, neg_list, compound_list, overall_sentiments = [], [], [], []
    for _, row in df.iterrows():
        scores = sia.polarity_scores(row["Article"])
        pos_list.append(scores["pos"])
        neg_list.append(scores["neg"])
        compound_list.append(round(scores["compound"], 3))
        if scores["compound"] >= 0.05:
            overall_sentiments.append("Bullish")
        elif scores["compound"] <= -0.05:
            overall_sentiments.append("Bearish")
        else:
            overall_sentiments.append("Neutral")
    df["Positive (%)"] = [x * 100 for x in pos_list]
    df["Negative (%)"] = [x * 100 for x in neg_list]
    df["Compound"] = compound_list
    df["Overall Sentiment"] = overall_sentiments
    return df

def _plot_overall_trend(articles_df):
    trend_df = articles_df[articles_df["Overall Sentiment"].isin(["Bullish", "Bearish"])]
    if trend_df.empty:
        st.info("Not enough data for overall trend indicator.")
        return
    count_df = trend_df.groupby("Overall Sentiment").size().reset_index(name="Count")
    fig = px.pie(count_df, names="Overall Sentiment", values="Count",
                 title="Overall Trend Indicator (Bullish vs. Bearish)",
                 color="Overall Sentiment", color_discrete_map={"Bullish": "green", "Bearish": "red"})
    st.plotly_chart(fig)
    st.markdown("**Overall Trend Explanation:** Counts of bullish vs. bearish articles indicate market sentiment.")

def _plot_source_distribution(articles_df):
    filtered_df = articles_df[articles_df["Overall Sentiment"].isin(["Bullish", "Bearish"])]
    if filtered_df.empty:
        st.info("Not enough data for source distribution analysis.")
        return
    pivot = filtered_df.groupby(["Website", "Overall Sentiment"]).size().unstack(fill_value=0)
    for col in ["Bullish", "Bearish"]:
        if col not in pivot.columns:
            pivot[col] = 0
    pivot = pivot.reset_index()
    fig = px.bar(pivot, x="Website", y=["Bullish", "Bearish"],
                 title="Sentiment Distribution by Source",
                 labels={"value": "Article Count", "Website": "Source"},
                 barmode="stack",
                 color_discrete_map={"Bullish": "green", "Bearish": "red"})
    st.plotly_chart(fig)

def _plot_trend_strength(articles_df):
    if articles_df.empty:
        st.info("No articles available for trend strength analysis.")
        return
    avg_compound = articles_df["Compound"].mean()
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=avg_compound,
        title={"text": "Trend Strength (Avg Compound Score)"},
        gauge={
            "axis": {"range": [-1, 1]},
            "bar": {"color": "darkblue"},
            "steps": [
                {"range": [-1, -0.05], "color": "red"},
                {"range": [-0.05, 0.05], "color": "gray"},
                {"range": [0.05, 1], "color": "green"}
            ]
        }
    ))
    fig.update_layout(title="Overall Trend Strength")
    st.plotly_chart(fig)
    st.markdown("**Trend Strength Explanation:** Values near +1 indicate bullish sentiment; near -1 indicate bearish.")

# ------------------------------
# Revamped Fundamental Analysis Section (Using Finnhub API)
# ------------------------------
def get_fundamental_data_finnhub(stock, api_key):
    url = f"https://finnhub.io/api/v1/stock/metric?symbol={stock.upper()}&metric=all&token={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        metric = data.get("metric", {})
        fundamentals = {
            "Trailing P/E": metric.get("trailingPE"),
            "Trailing EPS": metric.get("trailingEps"),
            "Market Cap": metric.get("marketCapitalization"),
            "Dividend Yield": metric.get("dividendYield"),  # fraction
            "ROE": metric.get("returnOnEquity"),
            "52 Week High": metric.get("52WeekHigh"),
            "52 Week Low": metric.get("52WeekLow"),
            "Beta": metric.get("beta")
        }
        return fundamentals
    return {
        "Trailing P/E": None,
        "Trailing EPS": None,
        "Market Cap": None,
        "Dividend Yield": None,
        "ROE": None,
        "52 Week High": None,
        "52 Week Low": None,
        "Beta": None
    }

# ------------------------------
# Revamped Technical Analysis Section (Using Finnhub Candle API)
# ------------------------------
def get_technical_data_finnhub(stock, api_key):
    end_time = int(time.time())
    start_time = end_time - 31536000  # 1 year
    url = f"https://finnhub.io/api/v1/stock/candle?symbol={stock.upper()}&resolution=D&from={start_time}&to={end_time}&token={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if data.get("s") == "ok":
            df = pd.DataFrame({
                "Timestamp": data["t"],
                "Close": data["c"],
                "Volume": data["v"]
            })
            df["Date"] = pd.to_datetime(df["Timestamp"], unit="s")
            df = df.sort_values("Date")
            # Compute 52-week high and low from the close prices
            week52_high = df["Close"].max()
            week52_low = df["Close"].min()
            # Use the last close as the current price
            last = df.iloc[-1]
            tech_data = {
                "Last Close": last["Close"],
                "Volume": last["Volume"],
                "52 Week High": week52_high,
                "52 Week Low": week52_low
            }
            return tech_data
    return {
        "Last Close": None,
        "Volume": None,
        "52 Week High": None,
        "52 Week Low": None
    }

# ------------------------------
# Main App Execution
# ------------------------------
stock_name = st.text_input("Enter a stock symbol (e.g., AAPL, TSLA) to analyze:", "")

if stock_name:
    # --- News & Sentiment Section ---
    status_placeholder = st.empty()
    status_placeholder.info(f"Fetching full articles for **{stock_name.upper()}**. Please wait...")
    try:
        articles_df = _scrape_full_articles(stock_name, max_articles=10)
        num_articles = articles_df.shape[0]
        status_placeholder.success(f"Fetched {num_articles} articles for {stock_name.upper()}.")
    except Exception as e:
        status_placeholder.error(f"Error during scraping: {e}")
        articles_df = pd.DataFrame()
    
    if not articles_df.empty:
        analysis_status = st.empty()
        analysis_status.info("Performing sentiment analysis on full articles...")
        try:
            articles_df = _analyze_full_article_sentiment(articles_df)
            analysis_status.success("Sentiment analysis complete!")
        except Exception as e:
            analysis_status.error(f"Error during sentiment analysis: {e}")
        
        st.markdown("### Full Article Analysis")
        display_df = articles_df[["Title", "Website", "Date", "Overall Sentiment", "Compound", "Positive (%)", "Negative (%)"]].copy()
        display_df["Read More"] = articles_df["Link"].apply(lambda x: f"<a href='{x}' target='_blank'><button class='read-more-btn'>Read More</button></a>")
        st.write(display_df.to_html(escape=False, index=False), unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            avg_pos = articles_df["Positive (%)"].mean()
            st.metric("Average Positive", f"{avg_pos:.1f}%")
        with col2:
            avg_neg = articles_df["Negative (%)"].mean()
            st.metric("Average Negative", f"{avg_neg:.1f}%")
        
        try:
            fig_bar = go.Figure(data=[
                go.Bar(
                    x=["Positive", "Negative"],
                    y=[avg_pos, avg_neg],
                    marker=dict(color=["green", "red"]),
                    text=[f"{avg_pos:.1f}%", f"{avg_neg:.1f}%"],
                    textposition="auto"
                )
            ])
            fig_bar.update_layout(title="Aggregated Sentiment Distribution", xaxis_title="Sentiment", yaxis_title="Average Percentage")
            st.plotly_chart(fig_bar)
        except Exception as e:
            st.error(f"Error generating sentiment chart: {e}")
        
        st.markdown("### Overall Trend Indicator")
        try:
            _plot_overall_trend(articles_df)
        except Exception as e:
            st.error(f"Error generating overall trend indicator: {e}")
        
        st.markdown("### Sentiment Distribution by Source")
        try:
            _plot_source_distribution(articles_df)
        except Exception as e:
            st.error(f"Error generating sentiment distribution by source: {e}")
        
        st.markdown("### Trend Strength Analysis")
        try:
            _plot_trend_strength(articles_df)
        except Exception as e:
            st.error(f"Error generating trend strength chart: {e}")
        
        # --- Fundamental Analysis Section ---
        st.markdown("### Fundamental Analysis")
        fund_data = get_fundamental_data_finnhub(stock_name, finnhub_api_key)
        st.markdown("#### Key Fundamental Metrics")
        col_f1, col_f2, col_f3 = st.columns(3)
        col_f1.metric("Trailing P/E", f"{fund_data['Trailing P/E']:.2f}" if isinstance(fund_data["Trailing P/E"], (int, float)) else "N/A")
        col_f2.metric("Trailing EPS", f"{fund_data['Trailing EPS']:.2f}" if isinstance(fund_data["Trailing EPS"], (int, float)) else "N/A")
        col_f3.metric("Market Cap", f"{fund_data['Market Cap']:,}" if isinstance(fund_data["Market Cap"], (int, float)) else "N/A")
        col_f4, col_f5 = st.columns(2)
        dyield = fund_data["Dividend Yield"]
        col_f4.metric("Dividend Yield", f"{dyield*100:.2f}%" if isinstance(dyield, (int, float)) else "N/A")
        col_f5.metric("ROE", f"{fund_data['ROE']*100:.2f}%" if isinstance(fund_data["ROE"], (int, float)) else "N/A")
        st.markdown("#### Additional Metrics")
        col_f6, col_f7 = st.columns(2)
        col_f6.metric("52-Week High", f"{fund_data['52 Week High']:.2f}" if isinstance(fund_data["52 Week High"], (int, float)) else "N/A")
        col_f7.metric("52-Week Low", f"{fund_data['52 Week Low']:.2f}" if isinstance(fund_data["52 Week Low"], (int, float)) else "N/A")
        st.metric("Beta", f"{fund_data['Beta']:.2f}" if isinstance(fund_data["Beta"], (int, float)) else "N/A")
        
        # --- Technical Analysis Section ---
        st.markdown("### Technical Analysis")
        tech_data = get_technical_data_finnhub(stock_name, finnhub_api_key)
        st.markdown("#### Key Technical Indicators")
        col_t1, col_t2 = st.columns(2)
        col_t1.metric("Last Close", f"{tech_data['Last Close']:.2f}" if isinstance(tech_data["Last Close"], (int, float)) else "N/A")
        col_t2.metric("Volume", tech_data["Volume"] if tech_data["Volume"] is not None else "N/A")
        col_t3, col_t4 = st.columns(2)
        col_t3.metric("52-Week High", f"{tech_data['52 Week High']:.2f}" if isinstance(tech_data["52 Week High"], (int, float)) else "N/A")
        col_t4.metric("52-Week Low", f"{tech_data['52 Week Low']:.2f}" if isinstance(tech_data["52 Week Low"], (int, float)) else "N/A")
        # Price Chart: Plot the "Close" prices over the last year
        if tech_data["Last Close"] is not None:
            # For simplicity, re-fetch the candle data to plot the price chart.
            end_time = int(time.time())
            start_time = end_time - 31536000
            url = f"https://finnhub.io/api/v1/stock/candle?symbol={stock_name.upper()}&resolution=D&from={start_time}&to={end_time}&token={finnhub_api_key}"
            resp = requests.get(url)
            if resp.status_code == 200:
                candle_data = resp.json()
                if candle_data.get("s") == "ok":
                    df_chart = pd.DataFrame({
                        "Timestamp": candle_data["t"],
                        "Close": candle_data["c"]
                    })
                    df_chart["Date"] = pd.to_datetime(df_chart["Timestamp"], unit="s")
                    df_chart = df_chart.sort_values("Date")
                    fig_chart = go.Figure(go.Scatter(x=df_chart["Date"], y=df_chart["Close"], mode="lines", name="Close Price"))
                    fig_chart.update_layout(title=f"{stock_name.upper()} Price Chart (1Y)", xaxis_title="Date", yaxis_title="Close Price")
                    st.plotly_chart(fig_chart)
        
        # --- Sentiment Analysis (Final Section) ---
        st.markdown("### News Sentiment")
        if not articles_df.empty:
            avg_sentiment = articles_df["Compound"].mean()
            st.metric("Average News Sentiment (Compound Score)", f"{avg_sentiment:.2f}")
        else:
            st.metric("Average News Sentiment (Compound Score)", "N/A")
        
    else:
        st.error("No full articles found for the given stock. Please try a different symbol.")
else:
    st.info("Enter a stock symbol above to begin analysis.")





