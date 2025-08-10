# pages/2_Model_Training.py
# --- make project root importable for "utils" ---
from __future__ import annotations
import os, sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# -----------------------------------------------


import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn import __version__ as sklearn_version
import joblib

from utils.shared import get_dataset, get_columns, ensure_dataset_loaded
from utils.text_cleaner import TextCleaner

st.set_page_config(page_title="Model Training", layout="wide")
st.title("🧪 Model Training — Bag-of-Words + Logistic Regression")

ensure_dataset_loaded()
df = get_dataset()
text_col, label_col = get_columns()

if df is None or text_col is None or label_col is None:
    st.error("Dataset or columns not set. Go to the Home page to select them.")
    st.stop()

st.write("Preview:")
st.dataframe(df[[text_col, label_col]].head(10), use_container_width=True)

# Preprocess controls
st.sidebar.header("🧹 Preprocessing")
lowercase = st.sidebar.checkbox("Lowercase", True)
remove_punct_num = st.sidebar.checkbox("Remove punctuation & numbers", True)
remove_stopwords = st.sidebar.checkbox("Remove stopwords", True)
use_stemming = st.sidebar.checkbox("Apply stemming", False)
min_word_len = st.sidebar.slider("Min word length", 1, 10, 2, 1)

st.sidebar.header("📦 Bag-of-Words")
max_features = st.sidebar.slider("Max features", 500, 30000, 5000, 500)
min_df = st.sidebar.slider("Min document frequency", 1, 10, 1, 1)
ngram = st.sidebar.selectbox("N-gram range", [(1,1), (1,2), (1,3)], index=0)

st.sidebar.header("🔀 Train/Test Split")
test_size = st.sidebar.slider("Test size", 0.1, 0.4, 0.2, 0.05)
random_state = st.sidebar.number_input("Random seed", 0, 9999, 42, 1)
stratify_split = st.sidebar.checkbox("Stratify by label", True)

st.sidebar.header("⚙️ Logistic Regression")
C_val = st.sidebar.slider("C (inverse regularization)", 0.01, 5.0, 1.0, 0.01)
max_iter = st.sidebar.slider("Max iterations", 100, 3000, 800, 100)
solver = st.sidebar.selectbox("Solver", ["liblinear", "lbfgs", "saga"], index=1)

train_btn = st.sidebar.button("🚀 Train model")

# Label mapping (to 0/1) if needed
st.subheader("Label Mapping")
label_series = df[label_col]
is_numeric = pd.api.types.is_numeric_dtype(label_series)

if is_numeric and set(pd.Series(label_series.unique()).dropna().unique()).issubset({0, 1}):
    st.success("Detected binary numeric labels (0/1).")
    y_raw = label_series.astype(int)
else:
    uniques = sorted(pd.Series(label_series.unique()).dropna().tolist())
    col_a, col_b = st.columns(2)
    with col_a:
        positive_choice = st.selectbox("Which value is **Positive (1)**?", uniques, index=0)
    with col_b:
        negative_choice = st.selectbox("Which value is **Negative (0)**?",
                                       [u for u in uniques if u != positive_choice], index=0)

    def map_label(v):
        if v == positive_choice:
            return 1
        elif v == negative_choice:
            return 0
        else:
            return np.nan

    y_raw = label_series.map(map_label).astype("float")
    n_missing = int(y_raw.isna().sum())
    if n_missing > 0:
        st.warning(f"{n_missing} rows have labels other than your chosen two values and will be dropped.")
    y_raw = y_raw.dropna().astype(int)

# Align X with y
X_raw = df.loc[y_raw.index, text_col].astype(str).fillna("")

# Build pipeline
cleaner = TextCleaner(
    lowercase=lowercase,
    remove_punct_num=remove_punct_num,
    remove_stopwords=remove_stopwords,
    use_stemming=use_stemming,
    min_word_len=min_word_len,
)
vectorizer = CountVectorizer(
    max_features=max_features,
    ngram_range=ngram,
    min_df=min_df,
    lowercase=False
)
clf = LogisticRegression(C=C_val, max_iter=max_iter, solver=solver)

pipeline = Pipeline([
    ("clean", cleaner),
    ("vectorizer", vectorizer),
    ("clf", clf),
])

# Train / Evaluate
report_placeholder = st.empty()
cm_placeholder = st.empty()
acc_placeholder = st.empty()

if train_btn:
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_raw, y_raw,
            test_size=float(test_size),
            random_state=int(random_state),
            stratify=y_raw if stratify_split else None
        )
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        probs = pipeline.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, preds)
        acc_placeholder.success(f"✅ Test Accuracy: **{acc:.3f}**")

        report_text = classification_report(
            y_test, preds, digits=3,
            target_names=["Negative (0)", "Positive (1)"]
        )
        report_placeholder.code(report_text, language="text")

        cm = confusion_matrix(y_test, preds, labels=[0, 1])
        fig, ax = plt.subplots()
        ax.imshow(cm, aspect="auto")
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["0", "1"]); ax.set_yticklabels(["0", "1"])
        for (i, j), z in np.ndenumerate(cm):
            ax.text(j, i, str(z), ha='center', va='center')
        cm_placeholder.pyplot(fig, use_container_width=True)

        st.session_state._pipeline = pipeline
        st.session_state._pipeline_meta = {
            "text_col": text_col,
            "label_col": label_col,
            "sklearn_version": sklearn_version,
            "params": {
                "C": C_val, "max_iter": max_iter, "solver": solver,
                "max_features": max_features, "min_df": min_df, "ngram": ngram
            }
        }
        st.success("Model trained and stored in session.")

        buffer = io.BytesIO()
        joblib.dump(pipeline, buffer)
        buffer.seek(0)
        st.download_button(
            label="💾 Download trained pipeline (.pkl)",
            data=buffer,
            file_name="sentiment_bow_logreg_pipeline.pkl",
            mime="application/octet-stream"
        )
        st.caption("The downloaded pipeline includes: TextCleaner → CountVectorizer → LogisticRegression.")

    except Exception as e:
        st.error(f"Training failed: {e}")
