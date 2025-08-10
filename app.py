# app.py
import streamlit as st
import pandas as pd
from utils.shared import (
    ensure_dataset_loaded, get_dataset, set_dataset,
    get_columns, set_columns
)

st.set_page_config(page_title="Sentiment App (BoW + Logistic Regression)", layout="wide")
st.title("🧱 Sentiment App — Multi-page")

st.markdown("""
Use the sidebar to load your dataset and choose the **text** and **label** columns.  
Then open the pages on the left:
1) **Visualization** → Word Cloud & Top-K frequencies  
2) **Model Training** → Bag-of-Words + Logistic Regression + .pkl download  
3) **Prediction** → Enter feedback → get **P(positive)**
""")

# ---------- Sidebar: dataset selection ----------
st.sidebar.header("📂 Dataset")
ensure_dataset_loaded()
df = get_dataset()

uploaded = st.sidebar.file_uploader("Upload CSV/TSV (optional)", type=["csv", "tsv", "txt"])
if uploaded is not None:
    try:
        tmp = pd.read_csv(uploaded, sep=None, engine="python")
    except Exception:
        uploaded.seek(0)
        try:
            tmp = pd.read_csv(uploaded, sep="\t")
        except Exception:
            uploaded.seek(0)
            tmp = pd.read_csv(uploaded)
    set_dataset(tmp)
    df = tmp

if df is None:
    st.warning("No dataset found. I’ll look for `/mnt/data/Restaurant_Reviews.tsv`. Please ensure it exists or upload a file from the sidebar.")
    st.stop()

st.success("Dataset loaded.")

st.subheader("Preview")
st.dataframe(df.head(10), use_container_width=True)

# Column selection
st.sidebar.header("🔧 Columns")
text_guess = "Review" if "Review" in df.columns else df.columns[0]
label_guess = "Liked" if "Liked" in df.columns else (df.columns[1] if len(df.columns) > 1 else df.columns[0])

text_col = st.sidebar.selectbox("Text column", options=list(df.columns),
                                index=list(df.columns).index(text_guess))
label_col = st.sidebar.selectbox("Label column (0/1 or mappable)", options=list(df.columns),
                                 index=list(df.columns).index(label_guess))

set_columns(text_col, label_col)

st.write(f"**Text column:** `{text_col}`  |  **Label column:** `{label_col}`")
if pd.api.types.is_numeric_dtype(df[label_col]):
    st.write("Label value counts:")
    st.write(df[label_col].value_counts(dropna=False))
else:
    st.info("Label column is not numeric; we’ll map it during training.")

st.markdown("---")
st.markdown("➡️ Now open the **pages** (left sidebar): **Visualization**, **Model Training**, **Prediction**.")
