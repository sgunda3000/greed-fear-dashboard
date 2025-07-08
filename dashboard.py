# greed_fear_dashboard/app/dashboard.py

import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
from transformers import pipeline

# Connect to SQLite database
conn = sqlite3.connect("sentiment.db")
cursor = conn.cursor()

# Create table if it doesn't exist
cursor.execute('''
CREATE TABLE IF NOT EXISTS sentiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT,
    sentiment TEXT,
    score REAL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);
''')
conn.commit()

# Load or simulate data
def load_data():
    try:
        data = pd.read_sql("SELECT * FROM sentiments ORDER BY timestamp DESC", conn)
    except pd.io.sql.DatabaseError:
        data = pd.DataFrame()

    if data.empty:
        # Simulate some data if DB is empty
        sample_data = pd.DataFrame({
            "text": ["Market is booming!", "Crypto crash incoming", "I'm neutral about stocks today"],
            "sentiment": ["positive", "negative", "neutral"],
            "score": [0.9, -0.8, 0.1]
        })
        sample_data.to_sql("sentiments", conn, if_exists="append", index=False)
        data = pd.read_sql("SELECT * FROM sentiments ORDER BY timestamp DESC", conn)
    return data

# Sentiment classifier (pretrained model)
sentiment_pipeline = pipeline("sentiment-analysis")

def analyze_and_store(text):
    result = sentiment_pipeline(text)[0]
    label = result["label"].lower()
    score = result["score"] if "positive" in label else -result["score"] if "negative" in label else 0
    cursor.execute("INSERT INTO sentiments (text, sentiment, score) VALUES (?, ?, ?)",
                   (text, label, score))
    conn.commit()

# Streamlit UI
st.title("📈 Greed & Fear Sentiment Dashboard")

st.write("This dashboard analyzes recent financial sentiment to compute a Greed & Fear index.")

# User input for testing new entries
with st.form("sentiment_form"):
    user_text = st.text_area("Enter a financial headline or tweet:", "Bitcoin hits all-time high!")
    submitted = st.form_submit_button("Analyze & Add")
    if submitted and user_text.strip():
        analyze_and_store(user_text)
        st.success("Sentiment analyzed and added to database!")

# Load data
data = load_data()

# Calculate Greed & Fear Index
greed_score = data[data['score'] > 0]['score'].sum()
fear_score = abs(data[data['score'] < 0]['score'].sum())
total_score = greed_score + fear_score

greed_percent = (greed_score / total_score * 100) if total_score != 0 else 50
fear_percent = 100 - greed_percent

# Display Gauge
st.metric("Greed Index", f"{greed_percent:.1f}%", delta=f"{greed_percent-50:+.1f}%")

# Pie Chart
fig_pie = px.pie(
    names=["Greed", "Fear"],
    values=[greed_percent, fear_percent],
    color_discrete_map={"Greed": "green", "Fear": "red"}
)
st.plotly_chart(fig_pie)

# Sentiment Trend
fig_line = px.line(
    data.sort_values("timestamp"),
    x="timestamp",
    y="score",
    title="Sentiment Scores Over Time"
)
st.plotly_chart(fig_line)

# Display Recent Data
st.subheader("Recent Sentiments")
st.dataframe(data[["timestamp", "text", "sentiment", "score"]].head(10))

conn.close()


# greed_fear_dashboard/requirements.txt
streamlit
pandas
transformers
torch
plotly


# greed_fear_dashboard/.streamlit/config.toml
[server]
headless = true
enableCORS = false
port = $PORT


# greed_fear_dashboard/README.md
# Greed & Fear Sentiment Dashboard

This project analyzes financial sentiment from text data (tweets/headlines) and visualizes a Greed & Fear index on a Streamlit dashboard.

## Features
- Sentiment analysis with Hugging Face Transformers.
- SQLite backend to store data.
- Interactive dashboard with charts and metrics.
- Add your own headlines/tweets to update sentiment.

## Setup
```bash
# Clone the repo
git clone https://github.com/sgunda3000/greed-fear-dashboard.git
cd greed-fear-dashboard

# Create virtual environment and install dependencies
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Run Streamlit app
streamlit run app/dashboard.py


