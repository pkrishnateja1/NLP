# pages/1_📊_Visualization.py
# --- make project root importable for "utils" ---
import os, sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# -----------------------------------------------

import re
from collections import Counter

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Try to import wordcloud; fallback gracefully if missing
try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except Exception:
    WORDCLOUD_AVAILABLE = False

import nltk
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from utils.shared import get_dataset, get_columns, ensure_dataset_loaded

st.set_page_config(page_title="Visualization", layout="wide")
st.title("📊 Data Visualization")

ensure_dataset_loaded()
df = get_dataset()
text_col, label_col = get_columns()

if df is None or text_col is None or label_col is None:
    st.error("Dataset or columns not set. Go to the Home page to select them.")
    st.stop()

# Sidebar
st.sidebar.header("🧹 Preprocessing (for visuals only)")
lowercase = st.sidebar.checkbox("Lowercase", value=True)
remove_punct_num = st.sidebar.checkbox("Remove punctuation & numbers", value=True)
use_stopwords = st.sidebar.checkbox("Remove stopwords", value=True)
use_stemming = st.sidebar.checkbox("Apply stemming", value=False)
min_word_len = st.sidebar.slider("Min word length", 1, 10, 2, 1)

st.sidebar.header("📦 Slice")
slice_mode = st.sidebar.selectbox(
    "Show words for:",
    ["Overall", "Only label = 1 (positive)", "Only label = 0 (negative)"],
    index=0
)

st.sidebar.header("📊 Bar Chart")
top_k = st.sidebar.slider("Top-K words", 10, 50, 20, 5)

# Helpers
BASIC_PUNCT_RE = re.compile(r"[^a-zA-Z\s]")
MULTI_SPACE_RE = re.compile(r"\s+")
STEMMER = None

def get_stopwords():
    sw = set(ENGLISH_STOP_WORDS)
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        try:
            nltk.download("stopwords", quiet=True)
        except Exception:
            pass
    try:
        from nltk.corpus import stopwords as nltk_sw
        sw.update(nltk_sw.words("english"))
    except Exception:
        pass
    return sw

def stem_tokens(tokens):
    global STEMMER
    if STEMMER is None:
        try:
            from nltk.stem.porter import PorterStemmer
            STEMMER = PorterStemmer()
        except Exception:
            return tokens
    return [STEMMER.stem(t) for t in tokens]

def clean_text(t: str) -> str:
    if not isinstance(t, str):
        t = str(t)
    if lowercase:
        t = t.lower()
    if remove_punct_num:
        t = BASIC_PUNCT_RE.sub(" ", t)
    t = MULTI_SPACE_RE.sub(" ", t).strip()
    tokens = [tok for tok in t.split() if len(tok) >= min_word_len]
    if use_stopwords:
        sw = get_stopwords()
        tokens = [tok for tok in tokens if tok not in sw]
    if use_stemming:
        tokens = stem_tokens(tokens)
    return " ".join(tokens)

def corpus_from_df(_df: pd.DataFrame) -> list[str]:
    return [clean_text(x) for x in _df[text_col].astype(str).fillna("")]

def top_k_freq(cleaned_docs: list[str], k: int) -> pd.DataFrame:
    counter = Counter()
    for doc in cleaned_docs:
        counter.update(doc.split())
    common = counter.most_common(k)
    return pd.DataFrame(common, columns=["word", "count"])

# Slice
if slice_mode.startswith("Only label = 1"):
    try:
        df_view = df[df[label_col].astype(int) == 1]
    except Exception:
        st.warning("Could not cast label to int; showing overall instead.")
        df_view = df
elif slice_mode.startswith("Only label = 0"):
    try:
        df_view = df[df[label_col].astype(int) == 0]
    except Exception:
        st.warning("Could not cast label to int; showing overall instead.")
        df_view = df
else:
    df_view = df

st.write("Preview (current slice):")
st.dataframe(df_view[[text_col, label_col]].head(10), use_container_width=True)

with st.spinner("Cleaning text for visuals..."):
    cleaned = corpus_from_df(df_view)

col1, col2 = st.columns(2)

with col1:
    st.subheader("☁️ Word Cloud")
    if not WORDCLOUD_AVAILABLE:
        st.info("`wordcloud` package not installed. Add `wordcloud>=1.9` to requirements.txt to enable this chart.")
    elif len(cleaned) == 0 or all(len(doc.strip()) == 0 for doc in cleaned):
        st.info("No text after preprocessing to build a word cloud.")
    else:
        freqs_df = top_k_freq(cleaned, k=500)
        freq_dict = dict(zip(freqs_df["word"], freqs_df["count"]))
        if len(freq_dict) == 0:
            st.info("No words found to render the word cloud.")
        else:
            from wordcloud import WordCloud
            wc = WordCloud(width=900, height=500, background_color="white")
            wc = wc.generate_from_frequencies(freq_dict)
            fig_wc, ax_wc = plt.subplots(figsize=(7, 4))
            ax_wc.imshow(wc, interpolation="bilinear")
            ax_wc.axis("off")
            st.pyplot(fig_wc, use_container_width=True)

with col2:
    st.subheader("📈 Top-K Word Frequency")
    if len(cleaned) == 0:
        st.info("No text after preprocessing to compute frequencies.")
    else:
        freq_df = top_k_freq(cleaned, k=top_k)
        if freq_df.empty:
            st.info("No words found.")
        else:
            st.dataframe(freq_df, use_container_width=True)
            fig_bar, ax_bar = plt.subplots(figsize=(7, 4))
            ax_bar.bar(freq_df["word"], freq_df["count"])
            ax_bar.set_ylabel("Count")
            ax_bar.set_xlabel("Word")
            ax_bar.set_title(f"Top {top_k} Words")
            plt.setp(ax_bar.get_xticklabels(), rotation=45, ha="right")
            st.pyplot(fig_bar, use_container_width=True)

st.caption("These visuals use only your loaded dataset and chosen columns. No sample data is ever used.")
