import streamlit as st
import plotly.express as px
import plotly.figure_factory as ff
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import hstack
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.loader import load_data

st.set_page_config(page_title="Decision Tree", page_icon="None", layout="wide")
st.title("Decision Tree Classifier")
st.markdown("Predicts manga **publication demographic** using status, language, and description text (TF-IDF).")
st.markdown("---")

df, le_demo = load_data()

with st.spinner("Training Decision Tree..."):
    X_struct = df[['status_encoded', 'language_encoded']]
    tfidf = TfidfVectorizer(max_features=300)
    X_tfidf = tfidf.fit_transform(df['text_clean'])
    X_final = hstack([X_tfidf, X_struct])
    y = df['demo_encoded']

    X_train, X_test, y_train, y_test = train_test_split(
        X_final, y, test_size=0.2, random_state=42
    )
    clf = DecisionTreeClassifier(max_depth=10, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)

# ── Metrics ────────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)
col1.metric("Accuracy",      f"{acc:.2%}")
col2.metric("Train Samples", f"{X_train.shape[0]:,}")
col3.metric("Test Samples",  f"{X_test.shape[0]:,}")

st.markdown("---")

# ── Confusion Matrix ───────────────────────────────────────────────────────
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Confusion Matrix")
    class_names = le_demo.classes_.tolist()
    cm = confusion_matrix(y_test, y_pred)
    fig = px.imshow(
        cm,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=class_names, y=class_names,
        color_continuous_scale="Blues",
        text_auto=True
    )
    fig.update_layout(xaxis_title="Predicted Label", yaxis_title="True Label")
    st.plotly_chart(fig, use_container_width=True)

with col_b:
    st.subheader("Classification Report")
    report = classification_report(y_test, y_pred,
                                   target_names=class_names,
                                   output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df = report_df.drop(['accuracy', 'macro avg', 'weighted avg'], errors='ignore')
    report_df = report_df[['precision', 'recall', 'f1-score', 'support']].round(3)
    report_df.index.name = 'Class'
    st.dataframe(report_df, use_container_width=True)

    st.markdown("---")
    st.markdown("""
    **Interpretation:**
    - **Precision** — of all predicted as X, how many were correct
    - **Recall** — of all actual X, how many were caught
    - **F1-score** — harmonic mean of precision and recall
    - **Accuracy: {:.1%}** — overall correct predictions
    """.format(acc))

st.markdown("---")

# ── Feature importance ─────────────────────────────────────────────────────
st.subheader("Top 20 Most Important Features")
feature_names = tfidf.get_feature_names_out().tolist() + ['status_encoded', 'language_encoded']
importances = clf.feature_importances_

top_idx = np.argsort(importances)[::-1][:20]
top_features = pd.DataFrame({
    'Feature':    [feature_names[i] for i in top_idx],
    'Importance': [importances[i] for i in top_idx]
})

fig = px.bar(top_features, x='Importance', y='Feature', orientation='h',
             color='Importance', color_continuous_scale='Teal')
fig.update_layout(yaxis={'categoryorder': 'total ascending'})
st.plotly_chart(fig, use_container_width=True)
st.caption("Words and features the Decision Tree found most useful for predicting demographic.")