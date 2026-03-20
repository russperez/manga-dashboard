import streamlit as st
import plotly.express as px
import pandas as pd
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.loader import load_data, get_clusters

st.set_page_config(page_title="Clustering", page_icon="None", layout="wide")
st.title("K-Means Clustering")
st.markdown("Manga grouped by description similarity using TF-IDF + KMeans (5 clusters) + PCA visualization.")
st.markdown("---")

df, _ = load_data()

with st.spinner("Running KMeans clustering..."):
    labels, coords = get_clusters(df)

df['cluster'] = labels
df['pca_x']   = coords[:, 0]
df['pca_y']   = coords[:, 1]
df['cluster_label'] = df['cluster'].apply(lambda x: f"Cluster {x}")

# ── PCA Scatter ────────────────────────────────────────────────────────────
st.subheader("PCA Scatter Plot: Manga Clusters")
fig = px.scatter(
    df, x='pca_x', y='pca_y',
    color='cluster_label',
    hover_data=['title', 'publicationDemographic', 'status'],
    color_discrete_sequence=px.colors.qualitative.Plotly,
    opacity=0.7
)
fig.update_traces(marker=dict(size=4))
fig.update_layout(
    xaxis_title="PCA Component 1",
    yaxis_title="PCA Component 2",
    legend_title="Cluster"
)
st.plotly_chart(fig, use_container_width=True)
st.caption("Manga positioned closer together share similar themes and keywords in their descriptions. PCA reduces TF-IDF features to 2D for visualization.")

st.markdown("---")

# ── Cluster size bar chart ─────────────────────────────────────────────────
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Cluster Sizes")
    cluster_counts = df['cluster_label'].value_counts().reset_index()
    cluster_counts.columns = ['Cluster', 'Count']
    fig = px.bar(cluster_counts, x='Cluster', y='Count',
                 color='Cluster',
                 color_discrete_sequence=px.colors.qualitative.Plotly)
    st.plotly_chart(fig, use_container_width=True)

with col_b:
    st.subheader("Demographic Distribution per Cluster")
    cross = df.dropna(subset=['publicationDemographic'])
    cross = cross.groupby(['cluster_label', 'publicationDemographic']).size().reset_index(name='Count')
    fig = px.bar(cross, x='cluster_label', y='Count',
                 color='publicationDemographic', barmode='stack',
                 color_discrete_sequence=px.colors.qualitative.Pastel)
    fig.update_layout(xaxis_title="Cluster", legend_title="Demographic")
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ── Browse by cluster ──────────────────────────────────────────────────────
st.subheader("Browse Manga by Cluster")
selected = st.selectbox("Select a cluster:", sorted(df['cluster_label'].unique()))
cluster_df = df[df['cluster_label'] == selected][['title', 'publicationDemographic', 'status', 'sentiment_score']].reset_index(drop=True)
cluster_df.columns = ['Title', 'Demographic', 'Status', 'Sentiment Score']
st.dataframe(cluster_df, use_container_width=True)
st.caption(f"Showing {len(cluster_df)} manga in {selected}")