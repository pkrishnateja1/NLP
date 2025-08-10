# pages/3_Prediction.py
# --- make project root importable for "utils" ---
import os, sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# -----------------------------------------------

import io
import joblib
import streamlit as st
from utils.shared import ensure_dataset_loaded, get_columns

st.set_page_config(page_title="Prediction", layout="wide")
st.title("🔮 Prediction")

ensure_dataset_loaded()
text_col, label_col = get_columns()

# Option 1: use in-session pipeline (from Model Training page)
pipeline = st.session_state.get("_pipeline", None)

st.subheader("Load Trained Pipeline")
uploaded_pkl = st.file_uploader("Upload a trained pipeline (.pkl) (optional)", type=["pkl"])
if uploaded_pkl is not None:
    try:
        # Read bytes and joblib.load
        byts = uploaded_pkl.read()
        pipeline = joblib.load(io.BytesIO(byts))
        st.success("Uploaded pipeline loaded.")
    except Exception as e:
        st.error(f"Failed to load pipeline: {e}")

if pipeline is None:
    st.info("No pipeline in session and no .pkl uploaded. Train a model first or upload a pipeline file.")
    st.stop()

st.subheader("Try a Prediction")
user_text = st.text_area("Type a feedback/review:", "The service was quick and the food was delicious.")
if st.button("Predict sentiment"):
    try:
        prob_pos = float(pipeline.predict_proba([user_text])[0][1])
        st.metric("Probability (Positive)", f"{prob_pos:.2f}")
        if prob_pos >= 0.5:
            st.success(f"Likely Positive ✅ (P = {prob_pos:.2f})")
        else:
            st.error(f"Likely Negative ❌ (Positive probability is low: {prob_pos:.2f})")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
