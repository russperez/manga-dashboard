# manga-dashboard

Interactive Streamlit dashboard for MangaDex metadata: Sentiment analysis, trend detection, K-Means clustering, content-based recommendations, and decision tree classification.

![Python](https://img.shields.io/badge/Python-3.8+-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red) ![License](https://img.shields.io/badge/License-MIT-green)

---

## Overview

This dashboard visualizes and explores manga metadata scraped from the MangaDex API. It is built on top of the [manga-data-warehouse](https://github.com/russperez/manga-data-warehouse) pipeline and presents the mining results in an interactive multi-page Streamlit app.

| | |
|---|---|
| **Course** | ITS132L – Data Warehousing and Data Mining |
| **Author** | Dharl Russell C. Perez |
| **Date** | August 2025 |

---

## Pages

| Page | Description |
|------|-------------|
| Overview | Total manga count, demographic distribution, status breakdown, language chart |
| Sentiment | VADER sentiment scores — histogram, by demographic, by status, top positive/negative |
| Trends | Manga per year, status over time, demographic over time, keyword trend search |
| Clusters | TF-IDF + KMeans (5 clusters) with PCA scatter plot and cluster browser |
| Recommendations | Content-based cosine similarity search with similarity score progress bars |
| Decision Tree | Demographic classifier with confusion matrix, classification report, feature importance |

---

## Installation
```bash
pip install -r requirements.txt
```

---

## Usage
```bash
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

---

## Project Structure
```
manga-dashboard/
├── app.py                   # Main entry point
├── requirements.txt
├── .streamlit/
│   └── config.toml          # Dark theme config
├── data/
│   └── mangadex_final.csv   # Cleaned dataset from manga-data-warehouse
├── pages/
│   ├── 1_overview.py
│   ├── 2_sentiment.py
│   ├── 3_trends.py
│   ├── 4_clusters.py
│   ├── 5_recommendations.py
│   └── 6_decision_tree.py
└── utils/
    └── loader.py            # Shared data loading and caching
```

---

## Data Source

Manga metadata scraped from the [MangaDex API](https://api.mangadex.org) — public metadata only, no user data collected.

---

## Related Repository

[manga-data-warehouse](https://github.com/russperez/manga-data-warehouse) — the full pipeline: scraping, cleaning, Oracle star schema, and data mining notebooks.

---

## Contact

**Dharl Russell C. Perez**  
dharlrussell@gmail.com
