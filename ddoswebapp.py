# ddoswebapp.py
import streamlit as st
import pandas as pd
import joblib
import requests
from pathlib import Path
import plotly.express as px

# -----------------------
# Configuration
# -----------------------
MODEL_URL = "https://nileuniversityedung-my.sharepoint.com/:u:/g/personal/242220003_nileuniversity_edu_ng/IQAMHZ9Q6qxkT6okCdyPblloARhbisvYocVQY4fE5L21Ibo?download=1"
MODEL_PATH = Path("bestmodel.pkl")

# -----------------------
# Download & load model
# -----------------------
@st.cache_resource(show_spinner=True)
def load_model():
    if not MODEL_PATH.exists():
        with st.spinner("Downloading model... (one-time)"):
            r = requests.get(MODEL_URL, stream=True)
            r.raise_for_status()
            with open(MODEL_PATH, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)

    bundle = joblib.load(MODEL_PATH)

    model = bundle["model"]
    scaler = bundle.get("scaler")
    selected_features = bundle["selected_features"]

    return model, scaler, selected_features


model, scaler, selected_features = load_model()

# -----------------------
# Streamlit UI
# -----------------------
st.title("Victor Yusuf – DDoS Detection App")
st.write("Upload one or more CSV files for batch prediction.")

uploaded_files = st.file_uploader(
    "Upload CSV file(s)",
    type="csv",
    accept_multiple_files=True
)

# -----------------------
# Prediction logic
# -----------------------
def predict_csv(df):
    cols = selected_features or df.select_dtypes(include="number").columns.tolist()
    X = df[cols].fillna(0)

    if scaler is not None:
        X = scaler.transform(X)

    preds = model.predict(X)
    df["prediction"] = preds

    if hasattr(model, "predict_proba"):
        df["probability"] = model.predict_proba(X)[:, 1]

    return df

# -----------------------
# Processing
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

        st.dataframe(
            final_df.style.apply(
                lambda x: [
                    "background-color: red" if v > 0.8 else ""
                    for v in x
                ] if x.name == "probability" else ["" for _ in x],
                axis=0
            )
        )

        fig = px.bar(
            final_df,
            x="prediction",
            y="probability" if "probability" in final_df.columns else None,
            color="probability" if "probability" in final_df.columns else None,
            color_continuous_scale="Reds",
            title="Prediction Probability Distribution",
            hover_data=["source_file"]
        )
        st.plotly_chart(fig, use_container_width=True)

        st.download_button(
            "Download results",
            final_df.to_csv(index=False).encode(),
            "ddos_predictions.csv",
            "text/csv"
        )
