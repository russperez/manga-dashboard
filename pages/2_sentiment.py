import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.loader import load_data

st.set_page_config(page_title="Sentiment Analysis", page_icon="None", layout="wide")
st.title("Sentiment Analysis")
st.markdown("VADER sentiment scores computed from manga descriptions.")
st.markdown("---")

df, _ = load_data()

# ── Metrics ────────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)
col1.metric("Average Sentiment",  f"{df['sentiment_score'].mean():.4f}")
col2.metric("Most Positive Score", f"{df['sentiment_score'].max():.4f}")
col3.metric("Most Negative Score", f"{df['sentiment_score'].min():.4f}")

st.markdown("---")

# ── Row 1 ──────────────────────────────────────────────────────────────────
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Sentiment Score Distribution")
    fig = px.histogram(df, x='sentiment_score', nbins=60,
                       color_discrete_sequence=['#7B68EE'],
                       labels={'sentiment_score': 'Sentiment Score'})
    fig.add_vline(x=0, line_dash="dash", line_color="red",
                  annotation_text="Neutral", annotation_position="top right")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Scores near +1 = positive, near -1 = negative, near 0 = neutral.")

with col_b:
    st.subheader("Average Sentiment by Demographic")
    sentiment_demo = df.dropna(subset=['publicationDemographic', 'sentiment_score'])
    grouped = sentiment_demo.groupby('publicationDemographic')['sentiment_score'].mean().reset_index()
    grouped.columns = ['Demographic', 'Avg Sentiment']
    fig = px.bar(grouped, x='Demographic', y='Avg Sentiment',
                 color='Avg Sentiment', color_continuous_scale='RdYlGn',
                 range_color=[-0.2, 0.2])
    st.plotly_chart(fig, use_container_width=True)

# ── Row 2 ──────────────────────────────────────────────────────────────────
col_c, col_d = st.columns(2)

with col_c:
    st.subheader("Average Sentiment by Status")
    status_group = df.dropna(subset=['status', 'sentiment_score'])
    status_avg = status_group.groupby('status')['sentiment_score'].mean().reset_index()
    status_avg.columns = ['Status', 'Avg Sentiment']
    fig = px.bar(status_avg, x='Status', y='Avg Sentiment',
                 color='Avg Sentiment', color_continuous_scale='RdBu')
    st.plotly_chart(fig, use_container_width=True)

with col_d:
    st.subheader("Sentiment Category Breakdown")
    df['sentiment_label'] = df['sentiment_score'].apply(
        lambda x: 'Positive' if x > 0.05 else ('Negative' if x < -0.05 else 'Neutral')
    )
    label_counts = df['sentiment_label'].value_counts().reset_index()
    label_counts.columns = ['Sentiment', 'Count']
    fig = px.pie(label_counts, names='Sentiment', values='Count',
                 color='Sentiment',
                 color_discrete_map={
                     'Positive': '#2ecc71',
                     'Neutral':  '#f39c12',
                     'Negative': '#e74c3c'
                 })
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ── Top/Bottom Tables ──────────────────────────────────────────────────────
col_e, col_f = st.columns(2)

with col_e:
    st.subheader("Top 10 Most Positive Manga")
    top_pos = df.nlargest(10, 'sentiment_score')[['title', 'publicationDemographic', 'sentiment_score']]
    top_pos.columns = ['Title', 'Demographic', 'Score']
    st.dataframe(top_pos.reset_index(drop=True), use_container_width=True)

with col_f:
    st.subheader("Top 10 Most Negative Manga")
    top_neg = df.nsmallest(10, 'sentiment_score')[['title', 'publicationDemographic', 'sentiment_score']]
    top_neg.columns = ['Title', 'Demographic', 'Score']
    st.dataframe(top_neg.reset_index(drop=True), use_container_width=True)