import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.loader import load_data

st.set_page_config(page_title="Trend Detection", page_icon="None", layout="wide")
st.title("Trend Detection")
st.markdown("Publication trends over time: by year, status, demographic, and keyword.")
st.markdown("---")

df, _ = load_data()
df_yearly = df.dropna(subset=['year'])
df_yearly = df_yearly[df_yearly['year'] >= 1990]

# ── Manga per year ────────────────────────────────────────────────────────
st.subheader("Number of Manga Published per Year")
manga_per_year = df_yearly['year'].value_counts().sort_index().reset_index()
manga_per_year.columns = ['Year', 'Count']
fig = px.line(manga_per_year, x='Year', y='Count', markers=True,
              color_discrete_sequence=['#3498db'])
fig.update_layout(xaxis_title="Year", yaxis_title="Number of Manga")
st.plotly_chart(fig, use_container_width=True)
st.caption("Shows peak publication periods and industry growth trends.")

st.markdown("---")

# ── Row 2 ─────────────────────────────────────────────────────────────────
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Status Trends Over Time")
    status_trend = df_yearly.groupby(['year', 'status']).size().reset_index(name='Count')
    fig = px.line(status_trend, x='year', y='Count', color='status',
                  markers=True,
                  color_discrete_sequence=px.colors.qualitative.Set1)
    fig.update_layout(xaxis_title="Year", yaxis_title="Number of Manga",
                      legend_title="Status")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Shows whether more manga are being completed or remain ongoing over time.")

with col_b:
    st.subheader("Demographic Trends Over Time")
    demo_trend = df_yearly.dropna(subset=['publicationDemographic'])
    demo_trend = demo_trend.groupby(['year', 'publicationDemographic']).size().reset_index(name='Count')
    fig = px.line(demo_trend, x='year', y='Count',
                  color='publicationDemographic', markers=True,
                  color_discrete_sequence=px.colors.qualitative.Pastel1)
    fig.update_layout(xaxis_title="Year", yaxis_title="Number of Manga",
                      legend_title="Demographic")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Shounen=young boys · Shoujo=young girls · Seinen=adult men · Josei=adult women")

st.markdown("---")

# ── Keyword Trend ─────────────────────────────────────────────────────────
st.subheader("Keyword Trend Search")
st.markdown("See how often a keyword appears in manga descriptions over time.")

keyword = st.text_input("Enter a keyword:", value="fantasy")

if keyword:
    df_yearly_copy = df_yearly.copy()
    df_yearly_copy['has_keyword'] = df_yearly_copy['text_clean'].str.contains(
        keyword.lower(), case=False, na=False
    )
    keyword_trend = df_yearly_copy.groupby('year')['has_keyword'].mean() * 100
    keyword_df = keyword_trend.reset_index()
    keyword_df.columns = ['Year', 'Percentage']

    fig = px.line(keyword_df, x='Year', y='Percentage', markers=True,
                  color_discrete_sequence=['#9b59b6'])
    fig.update_layout(
        title=f"Usage of '{keyword.capitalize()}' in Manga Descriptions (%)",
        xaxis_title="Year",
        yaxis_title="% of Manga Containing Keyword"
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Percentage of manga descriptions containing the word '{keyword}' per year.")