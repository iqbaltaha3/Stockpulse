import streamlit as st
st.set_page_config(page_title="StockPulse", page_icon="📊", layout="wide")

import pandas as pd
import requests
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import plotly.graph_objects as go
import plotly.express as px
from bs4 import BeautifulSoup
import joblib
from sklearn.linear_model import LogisticRegression
import numpy as np

# Attempt to import yfinance for live data
try:
    import yfinance as yf
except ImportError:
    yf = None
    st.warning("yfinance not installed. Live data may be incomplete.")

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
    <div class="subheader">Advanced Sentiment Analysis & Live Market Insights</div>
    <div class="landing">
      Welcome to <strong>StockPulse</strong> – a cutting-edge tool for retail investors.
      The app aggregates full-text news articles from 20 reputed Indian financial news websites and performs advanced sentiment analysis 
      to gauge the bullish or bearish outlook of your chosen stock.
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
    # Ensure both 'Bullish' and 'Bearish' columns exist
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

def _load_sample_tweets(stock):
    data = [
         {"Date": "2023-08-01", "Username": "investor1", "Content": f"{stock} is looking strong today! Great buy opportunity."},
         {"Date": "2023-08-02", "Username": "trader2", "Content": f"I am not sure about {stock}, seems overvalued."},
         {"Date": "2023-08-03", "Username": "marketguru", "Content": f"{stock} is crashing, very bearish signal."},
         {"Date": "2023-08-04", "Username": "bullish_bear", "Content": f"Solid performance by {stock} despite market volatility."},
         {"Date": "2023-08-05", "Username": "trader123", "Content": f"{stock} appears to be recovering after a dip."},
    ]
    return pd.DataFrame(data)

def _analyze_tweet_sentiment(df):
    sentiments, compounds = [], []
    for content in df["Content"]:
        score = sia.polarity_scores(content)["compound"]
        compounds.append(score)
        if score >= 0.05:
            sentiments.append("Bullish")
        elif score <= -0.05:
            sentiments.append("Bearish")
        else:
            sentiments.append("Neutral")
    df["Compound"] = compounds
    df["Sentiment"] = sentiments
    return df

# ------------------------------
# Revamped Fundamental Analysis Section
# ------------------------------
def get_fundamental_data(stock):
    fundamentals = {}
    if yf:
        try:
            ticker = yf.Ticker(stock)
            info = ticker.info
        except Exception:
            info = {}
    else:
        info = {}
    # Fetch live fundamental data; missing values remain as None and will be shown as "N/A"
    fundamentals["Stock"] = info.get("symbol", stock.upper())
    fundamentals["P/E Ratio"] = info.get("trailingPE")
    fundamentals["EPS"] = info.get("trailingEps")
    fundamentals["Market Cap"] = info.get("marketCap")
    fundamentals["Dividend Yield (%)"] = info.get("dividendYield")
    fundamentals["ROE (%)"] = info.get("returnOnEquity")
    return fundamentals

# ------------------------------
# Revamped Technical Analysis Section
# ------------------------------
def get_technical_data(stock):
    tech_data = {}
    if yf:
        try:
            # Fetch one year of historical data to improve data availability
            hist = yf.download(stock, period="1y")
        except Exception:
            hist = pd.DataFrame()
    else:
        hist = pd.DataFrame()
    if not hist.empty and "Close" in hist.columns:
        hist["MA50"] = hist["Close"].rolling(window=50).mean()
        hist["MA200"] = hist["Close"].rolling(window=200).mean()
        delta = hist["Close"].diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        ma_up = up.rolling(window=14).mean()
        ma_down = down.rolling(window=14).mean()
        rs = ma_up / ma_down
        hist["RSI"] = 100 - (100 / (1 + rs))
        valid_hist = hist.dropna()
        if not valid_hist.empty:
            last = valid_hist.iloc[-1]
            tech_data["Stock"] = stock.upper()
            tech_data["Last Close"] = last["Close"]
            tech_data["50-Day MA"] = last["MA50"]
            tech_data["200-Day MA"] = last["MA200"]
            tech_data["RSI"] = last["RSI"]
            tech_data["Volume"] = last["Volume"]
        else:
            tech_data = {key: None for key in ["Stock", "Last Close", "50-Day MA", "200-Day MA", "RSI", "Volume"]}
    else:
        tech_data = {key: None for key in ["Stock", "Last Close", "50-Day MA", "200-Day MA", "RSI", "Volume"]}
    return tech_data

# ------------------------------
# Main App Execution
# ------------------------------
stock_name = st.text_input("Enter a stock symbol (e.g., AAPL, TSLA) to analyze:", "")

if stock_name:
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
        
        # Revamped Fundamental Analysis Section
        st.markdown("### Fundamental Analysis")
        fund_data = get_fundamental_data(stock_name)
        st.markdown("#### Key Fundamental Metrics")
        col_f1, col_f2, col_f3 = st.columns(3)
        col_f1.metric("P/E Ratio", fund_data["P/E Ratio"] if fund_data["P/E Ratio"] is not None else "N/A")
        col_f2.metric("EPS", fund_data["EPS"] if fund_data["EPS"] is not None else "N/A")
        col_f3.metric("Market Cap", f"{fund_data['Market Cap']:,}" if isinstance(fund_data["Market Cap"], (int, float)) else "N/A")
        col_f4, col_f5 = st.columns(2)
        dyield = fund_data["Dividend Yield (%)"]
        col_f4.metric("Dividend Yield", f"{dyield*100:.2f}%" if isinstance(dyield, (int, float)) else "N/A")
        col_f5.metric("ROE (%)", f"{fund_data['ROE (%)']*100:.2f}%" if isinstance(fund_data["ROE (%)"], (int, float)) else "N/A")
        # Calculate Fundamental Score = (ROE / P/E Ratio) * 100 (treat missing as 0)
        try:
            pe = float(fund_data["P/E Ratio"]) if isinstance(fund_data["P/E Ratio"], (int, float)) else 0
            roe = float(fund_data["ROE (%)"]) if isinstance(fund_data["ROE (%)"], (int, float)) else 0
            fundamental_score = (roe / pe) * 100 if pe != 0 else 0
            fundamental_score = round(fundamental_score, 2)
        except Exception:
            fundamental_score = 0
        st.markdown("#### Fundamental Value Score")
        st.metric("Fundamental Score", f"{fundamental_score:.2f}")
        st.markdown("""
        **Fundamental Score Explanation:**  
        This score is calculated as (ROE / P/E Ratio) × 100. A higher score suggests a company may be undervalued relative to its profitability.
        """)
        
        # Revamped Technical Analysis Section
        st.markdown("### Technical Analysis")
        tech_data = get_technical_data(stock_name)
        st.markdown("#### Key Technical Indicators")
        col_t1, col_t2, col_t3 = st.columns(3)
        col_t1.metric("Last Close", f"{tech_data['Last Close']:.2f}" if isinstance(tech_data["Last Close"], (int, float)) else "N/A")
        col_t2.metric("50-Day MA", f"{tech_data['50-Day MA']:.2f}" if isinstance(tech_data["50-Day MA"], (int, float)) else "N/A")
        col_t3.metric("200-Day MA", f"{tech_data['200-Day MA']:.2f}" if isinstance(tech_data["200-Day MA"], (int, float)) else "N/A")
        col_t4, col_t5 = st.columns(2)
        col_t4.metric("RSI", f"{tech_data['RSI']:.2f}" if isinstance(tech_data["RSI"], (int, float)) else "N/A")
        col_t5.metric("Volume", tech_data["Volume"] if tech_data["Volume"] is not None else "N/A")
        # Calculate Technical Strength Score = (Last Close / 50-Day MA) * 100
        try:
            last_close = float(tech_data["Last Close"]) if isinstance(tech_data["Last Close"], (int, float)) else 0
            ma50 = float(tech_data["50-Day MA"]) if isinstance(tech_data["50-Day MA"], (int, float)) else 0
            technical_strength = (last_close / ma50) * 100 if ma50 != 0 else 0
            technical_strength = round(technical_strength, 2)
        except Exception:
            technical_strength = 0
        st.markdown("#### Technical Strength Score")
        st.metric("Technical Strength", f"{technical_strength:.2f}")
        st.markdown("""
        **Technical Strength Explanation:**  
        Calculated as (Last Close / 50-Day MA) × 100, a score above 100 suggests bullish momentum.
        """)
        if yf:
            hist = yf.download(stock_name, period="1y")
            if not hist.empty:
                hist["MA50"] = hist["Close"].rolling(window=50).mean()
                hist["MA200"] = hist["Close"].rolling(window=200).mean()
                fig_tech = go.Figure()
                fig_tech.add_trace(go.Scatter(x=hist.index, y=hist["Close"], mode="lines", name="Close Price"))
                fig_tech.add_trace(go.Scatter(x=hist.index, y=hist["MA50"], mode="lines", name="50-Day MA"))
                fig_tech.add_trace(go.Scatter(x=hist.index, y=hist["MA200"], mode="lines", name="200-Day MA"))
                fig_tech.update_layout(title=f"{stock_name.upper()} Price & Moving Averages", xaxis_title="Date", yaxis_title="Price")
                st.plotly_chart(fig_tech)
            fig_rsi = go.Figure(go.Indicator(
                mode="gauge+number",
                value=tech_data["RSI"] if isinstance(tech_data["RSI"], (int, float)) else 0,
                title={"text": "RSI"},
                gauge={"axis": {"range": [0, 100]},
                       "bar": {"color": "blue"},
                       "steps": [
                           {"range": [0, 30], "color": "red"},
                           {"range": [30, 70], "color": "gray"},
                           {"range": [70, 100], "color": "green"}
                       ]}
            ))
            fig_rsi.update_layout(title="Relative Strength Index (RSI)")
            st.plotly_chart(fig_rsi)
            st.markdown("""
            **RSI Explanation:**  
            RSI = 100 - (100 / (1 + RS)); values above 70 may indicate overbought conditions, while values below 30 may indicate oversold.
            """)
        
        # Revamped Final Prediction Section
        st.markdown("### Final Prediction")
        # Build feature vector: average sentiment, fundamental score, technical strength
        avg_sentiment = articles_df["Compound"].mean() if not articles_df.empty else 0
        feature_vector = [avg_sentiment, fundamental_score, technical_strength]
        # For demonstration, we simulate training data with similar features
        train_data = pd.DataFrame({
            'avg_sentiment': [0.30, -0.20, 0.25, -0.30, 0.15, -0.10, 0.35, -0.25],
            'fund_score': [80, 40, 70, 30, 60, 50, 90, 35],
            'tech_strength': [110, 95, 105, 90, 100, 98, 115, 92]
        })
        train_labels = ['Bullish', 'Bearish', 'Bullish', 'Bearish', 'Bullish', 'Bullish', 'Bullish', 'Bearish']
        model = LogisticRegression()
        model.fit(train_data, train_labels)
        
        prediction = model.predict(np.array(feature_vector).reshape(1, -1))[0]
        confidence = model.predict_proba(np.array(feature_vector).reshape(1, -1))[0].max()
        
        st.metric("Predicted Trend", prediction)
        st.markdown(f"**Model Confidence:** {confidence*100:.1f}%")
        fig_pred = go.Figure(go.Indicator(
            mode="gauge+number",
            value=confidence*100,
            title={"text": "Trend Confidence (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "orange"},
                "steps": [
                    {"range": [0, 50], "color": "red"},
                    {"range": [50, 75], "color": "yellow"},
                    {"range": [75, 100], "color": "green"}
                ]
            }
        ))
        st.plotly_chart(fig_pred)
        st.markdown("""
        **Final Prediction Explanation:**  
        The model uses three features:
        - Average News Sentiment (compound score)
        - Fundamental Score (ROE / P/E Ratio × 100)
        - Technical Strength (Last Close / 50-Day MA × 100)
        to predict the stock's trend. A prediction of **Bullish** suggests a positive outlook, while **Bearish** indicates potential downside.
        """)
    else:
        st.error("No full articles found for the given stock. Please try a different symbol.")
else:
    st.info("Enter a stock symbol above to begin analysis.")


