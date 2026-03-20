import streamlit as st

st.set_page_config(
    page_title="Manga Analytics Dashboard",
    page_icon=None,
    layout="wide"
)

st.title("Manga Analytics Dashboard")
st.markdown(
    "Exploratory analysis of MangaDex metadata: "
    "sentiment analysis, trend detection, clustering, recommendations, and decision tree."
)

st.markdown("---")

st.page_link("pages/1_overview.py",         label="Overview")
st.page_link("pages/2_sentiment.py",        label="Sentiment")
st.page_link("pages/3_trends.py",           label="Trends")
st.page_link("pages/4_clusters.py",         label="Clusters")
st.page_link("pages/5_recommendations.py",  label="Recommendations")
st.page_link("pages/6_decision_tree.py",    label="Decision Tree")

st.markdown("---")
st.caption("Data source: MangaDex API · Course: ITS132L · Author: Dharl Russell C. Perez")