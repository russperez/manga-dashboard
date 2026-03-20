import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.loader import load_data, get_tfidf_matrix

st.set_page_config(page_title="Recommendations", page_icon="None", layout="wide")
st.title("Manga Recommendation System")
st.markdown("Content-based recommendations using TF-IDF cosine similarity on manga descriptions.")
st.markdown("---")

df, _ = load_data()
tfidf, tfidf_matrix = get_tfidf_matrix(df)

# ── Search ─────────────────────────────────────────────────────────────────
col_a, col_b, col_c = st.columns([3, 1, 1])

with col_a:
    # Autocomplete via selectbox
    all_titles = sorted(df['title'].dropna().unique().tolist())
    selected_title = st.selectbox("Search for a manga title:", all_titles)

with col_b:
    top_n = st.number_input("Number of recommendations:", min_value=1, max_value=20, value=5)

with col_c:
    filter_demo = st.checkbox("Same demographic only", value=True)

st.markdown("")

if st.button("Get Recommendations", type="primary"):
    if selected_title not in df['title'].values:
        st.error(f"'{selected_title}' not found in the dataset.")
    else:
        idx = df[df['title'] == selected_title].index[0]

        # Show selected manga info
        st.markdown("### Selected Manga")
        sel = df.loc[idx]
        c1, c2, c3 = st.columns(3)
        c1.metric("Title", sel['title'])
        c2.metric("Demographic", str(sel['publicationDemographic']))
        c3.metric("Status", str(sel['status']))
        st.info(str(sel['description'])[:300] + "..." if len(str(sel['description'])) > 300 else str(sel['description']))

        st.markdown("---")
        st.markdown("### Recommendations")

        # Cosine similarity
        cosine_sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
        sim_indices = cosine_sim.argsort()[::-1][1:top_n + 20]

        # Filter by demographic
        if filter_demo:
            original_demo = df.loc[idx, 'publicationDemographic']
            sim_indices = [i for i in sim_indices if df.loc[i, 'publicationDemographic'] == original_demo]

        sim_indices = sim_indices[:top_n]

        if len(sim_indices) == 0:
            st.warning("No recommendations found. Try unchecking 'Same demographic only'.")
        else:
            for rank, i in enumerate(sim_indices, 1):
                row = df.loc[i]
                with st.expander(f"#{rank} — {row['title']}"):
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Demographic", str(row['publicationDemographic']))
                    col2.metric("Status", str(row['status']))
                    col3.metric("Language", str(row['originalLanguage']))
                    col4.metric("Sentiment", f"{row['sentiment_score']:.3f}")
                    st.write(str(row['description'])[:300] + "..." if len(str(row['description'])) > 300 else str(row['description']))
                    sim_score = cosine_sim[i]
                    st.progress(float(sim_score), text=f"Similarity score: {sim_score:.4f}")