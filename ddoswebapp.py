# -*- coding: utf-8 -*-
"""Victor Yusuf DDoS Web App"""

import streamlit as st
import pandas as pd
import joblib
import requests
from pathlib import Path
import plotly.express as px

# -----------------------
# Model download config
# -----------------------
MODEL_PATH = Path("bestmodel.pkl")
MODEL_URL = "https://drive.google.com/uc?export=download&id=1KKiMyCLpeXfYYQcj5Nsu_g79RZp9KKs4"


@st.cache_resource
def download_and_load_model():
    if not MODEL_PATH.exists():
        with st.spinner("Downloading ML model..."):
            r = requests.get(MODEL_URL, stream=True)
            r.raise_for_status()
            with open(MODEL_PATH, "wb") as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)
    bundle = joblib.load(MODEL_PATH)

    # Check type and handle dict or raw model
    if isinstance(bundle, dict):
        model = bundle.get("model")
        scaler = bundle.get("scaler", None)
        selected_features = bundle.get("selected_features", None)
    else:
        model = bundle
        scaler = None
        selected_features = None
    return model, scaler, selected_features


model, scaler, selected_features = download_and_load_model()
st.title("Victor Yusuf DDoS Web App")
st.write("Upload one or more CSV files for batch prediction.")

# -----------------------
# File uploader
# -----------------------
uploaded_files = st.file_uploader(
    "Upload CSV file(s)",
    type="csv",
    accept_multiple_files=True
)


# -----------------------
# Prediction function
# -----------------------
def predict_csv(df):
    # Detect features if not provided
    cols = selected_features or df.select_dtypes(include="number").columns.tolist()
    X = df[cols].fillna(0)
    preds = model.predict(X)
    df["prediction"] = preds

    if hasattr(model, "predict_proba"):
        df["probability"] = model.predict_proba(X)[:, 1]

    return df


# -----------------------
# Process uploaded files
# -----------------------
if uploaded_files:
    results = []
    progress = st.progress(0)

    for i, file in enumerate(uploaded_files, 1):
        try:
            df = pd.read_csv(file)
            df = predict_csv(df)
            df["source_file"] = file.name
            results.append(df)
            progress.progress(i / len(uploaded_files))
        except Exception as e:
            st.error(f"Error processing {file.name}: {e}")

    if results:
        final_df = pd.concat(results, ignore_index=True)
        st.success("Prediction completed!")
        st.dataframe(final_df.style.apply(
            lambda x: ['background-color: red' if v > 0.8 else '' for v in x]
            if x.name == "probability" else [''] * len(x), axis=0
        ))

        # Plot interactive bar chart with Plotly
        fig = px.bar(
            final_df,
            x="prediction",
            y="probability" if "probability" in final_df.columns else None,
            color="probability" if "probability" in final_df.columns else None,
            color_continuous_scale="Reds",
            title="Predicted Class Counts with Probability",
            hover_data=["source_file"]
        )
        st.plotly_chart(fig, use_container_width=True)

        # Download button
        st.download_button(
            "Download results",
            final_df.to_csv(index=False).encode(),
            "ddos_predictions.csv",
            "text/csv"
        )
