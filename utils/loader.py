import pandas as pd
import streamlit as st
import re
import string
import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

nltk.download('vader_lexicon', quiet=True)

@st.cache_data
def load_data():
    df = pd.read_csv('data/mangadex_final.csv')

    # text cleaning (same as notebook)
    df['text'] = (df['title'].fillna('') + ' ' + df['description'].fillna('')).str.lower()
    df['text_clean'] = df['text'].apply(
        lambda x: re.sub(r'[{}0-9]'.format(re.escape(string.punctuation)), ' ', x)
    )
    stop_words = ENGLISH_STOP_WORDS
    df['text_clean'] = df['text_clean'].apply(
        lambda x: ' '.join([w for w in x.split() if w not in stop_words])
    )

    # sentiment
    if 'sentiment_score' not in df.columns:
        sia = SentimentIntensityAnalyzer()
        df['sentiment_score'] = df['description'].astype(str).apply(
            lambda x: sia.polarity_scores(x)['compound']
        )

    # encoders
    le_status = LabelEncoder()
    le_lang   = LabelEncoder()
    le_demo   = LabelEncoder()
    df['status_encoded']   = le_status.fit_transform(df['status'].fillna('unknown'))
    df['language_encoded'] = le_lang.fit_transform(df['originalLanguage'].fillna('unknown'))
    df['demo_encoded']     = le_demo.fit_transform(df['publicationDemographic'].fillna('unknown'))

    # year
    df['year'] = pd.to_numeric(df['year'], errors='coerce')

    return df, le_demo


@st.cache_resource
def get_tfidf_matrix(df):
    tfidf = TfidfVectorizer(max_features=500, stop_words='english')
    matrix = tfidf.fit_transform(df['text_clean'])
    return tfidf, matrix


@st.cache_data
def get_clusters(_df):
    tfidf = TfidfVectorizer(max_features=500, stop_words='english')
    X = tfidf.fit_transform(_df['text_clean'])
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    pca = PCA(n_components=2)
    coords = pca.fit_transform(X.toarray())
    return labels, coords