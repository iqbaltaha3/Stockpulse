# Stockpulse
https://stockpulse01.streamlit.app/

StockPulse is an AI-powered tool designed for retail investors to gain actionable insights on stock performance. Leveraging Python, Streamlit, and cutting-edge libraries such as NLTK (with VADER sentiment analysis) and Plotly, StockPulse scrapes full-text news articles from top financial websites to assess market sentiment—classifying news as bullish, bearish, or neutral.

## Key technical highlights include:

Live Data Integration: Uses yfinance (when available) to fetch real-time fundamental and technical metrics (e.g., P/E Ratio, EPS, moving averages, RSI) or falls back to simulated data.

Robust Sentiment Analysis: Processes and scores news articles with sophisticated natural language tools, transforming raw text into quantifiable sentiment indicators.

Interactive Visualizations: Presents findings through dynamic pie charts, bar graphs, and gauge indicators to illustrate overall sentiment, trend strength, and source distribution.

Predictive Modeling: Combines sentiment, fundamental value, and technical strength scores using a logistic regression model to predict stock trends and quantify confidence in its forecast.
