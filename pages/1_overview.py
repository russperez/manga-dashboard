import streamlit as st
import plotly.express as px
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.loader import load_data

st.set_page_config(page_title="Overview", page_icon="None", layout="wide")
st.title("Overview")
st.markdown("General statistics and distributions across the MangaDex dataset.")
st.markdown("---")

df, _ = load_data()

# ── Metrics ────────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Manga",       f"{len(df):,}")
col2.metric("Unique Languages",  df['originalLanguage'].nunique())
col3.metric("Demographics",      df['publicationDemographic'].nunique())
col4.metric("Status Types",      df['status'].nunique())

st.markdown("---")

# ── Row 1 ──────────────────────────────────────────────────────────────────
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Publication Demographic")
    demo_counts = df['publicationDemographic'].value_counts().reset_index()
    demo_counts.columns = ['Demographic', 'Count']
    fig = px.pie(demo_counts, names='Demographic', values='Count',
                 color_discrete_sequence=px.colors.qualitative.Pastel)
    st.plotly_chart(fig, use_container_width=True)

with col_b:
    st.subheader("Manga Status Breakdown")
    status_counts = df['status'].value_counts().reset_index()
    status_counts.columns = ['Status', 'Count']
    fig = px.bar(status_counts, x='Status', y='Count',
                 color='Status',
                 color_discrete_sequence=px.colors.qualitative.Set2)
    st.plotly_chart(fig, use_container_width=True)

# ── Row 2 ──────────────────────────────────────────────────────────────────
col_c, col_d = st.columns(2)

with col_c:
    st.subheader("Top 15 Original Languages")
    lang_counts = df['originalLanguage'].value_counts().head(15).reset_index()
    lang_counts.columns = ['Language', 'Count']
    fig = px.bar(lang_counts, x='Count', y='Language', orientation='h',
                 color='Count', color_continuous_scale='Blues')
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, use_container_width=True)

with col_d:
    st.subheader("Status by Demographic")
    cross = df.groupby(['publicationDemographic', 'status']).size().reset_index(name='Count')
    fig = px.bar(cross, x='publicationDemographic', y='Count',
                 color='status', barmode='group',
                 color_discrete_sequence=px.colors.qualitative.Vivid)
    st.plotly_chart(fig, use_container_width=True)

st.caption("Data source: MangaDex API")