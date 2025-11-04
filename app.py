# app.py
import os
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
# removed Jinja2Templates; serving static HTML via FileResponse
import joblib
import numpy as np
import pandas as pd

from data_utils import prepare_dataset, preprocess_text  # completed import

import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

# Initialize FastAPI app
app = FastAPI(title="Sentiment Analysis for Product Reviews")

# Directories
TEMPLATE_DIR = "templates"
STATIC_DIR = "static"
MODEL_DIR = "models"

# Mount static directory for JS/CSS
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
# no templates engine needed

# Load models if available
NB_MODEL_PATH = os.path.join(MODEL_DIR, "tfidf_nb.joblib")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer.joblib")
LSTM_MODEL_PATH = os.path.join(MODEL_DIR, "lstm_model.keras")

NB_PIPE = joblib.load(NB_MODEL_PATH) if os.path.exists(NB_MODEL_PATH) else None
TOKENIZER = joblib.load(TOKENIZER_PATH) if os.path.exists(TOKENIZER_PATH) else None
LSTM_MODEL = load_model(LSTM_MODEL_PATH) if os.path.exists(LSTM_MODEL_PATH) else None

# ================================
# ROUTES
# ================================

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    """Render the dashboard UI."""
    return FileResponse(os.path.join(TEMPLATE_DIR, "dashboard.html"))


@app.post("/api/predict")
async def predict(payload: dict):
    """Predict sentiment using selected model.
    payload: { text: str, model?: "nb"|"lstm" }
    """
    text = payload.get("text", "")
    model_choice = (payload.get("model") or "nb").lower()
    if not text:
        return JSONResponse({"error": "No text provided."}, status_code=400)

    cleaned = preprocess_text(pd.Series([text]))[0]

    if model_choice == "lstm":
        if TOKENIZER is None or LSTM_MODEL is None:
            return JSONResponse({"error": "LSTM model not loaded. Train it first."}, status_code=500)
        seq = TOKENIZER.texts_to_sequences([cleaned])
        seq_padded = pad_sequences(seq, maxlen=LSTM_MODEL.input_shape[1], padding="post")
        probs = LSTM_MODEL.predict(seq_padded)[0].tolist()
        label_map = {0: "negative", 1: "neutral", 2: "positive"}
        label_idx = int(np.argmax(probs))
        return {
            "model": "LSTM",
            "input": text,
            "prediction": label_map[label_idx],
            "probabilities": {
                "negative": probs[0],
                "neutral": probs[1],
                "positive": probs[2]
            }
        }

    # default: Naive Bayes
    if NB_PIPE is None:
        return JSONResponse({"error": "Naive Bayes model not loaded. Train it first."}, status_code=500)
    pred = NB_PIPE.predict([cleaned])[0]
    probs = NB_PIPE.predict_proba([cleaned])[0].tolist()
    labels = NB_PIPE.classes_.tolist()
    return {
        "model": "Naive Bayes",
        "input": text,
        "prediction": pred,
        "probabilities": dict(zip(labels, probs))
    }


@app.post("/api/predict_lstm")
async def predict_lstm(payload: dict):
    """Predict sentiment using LSTM model."""
    text = payload.get("text", "")
    if not text:
        return JSONResponse({"error": "No text provided."}, status_code=400)

    if TOKENIZER is None or LSTM_MODEL is None:
        return JSONResponse({"error": "LSTM model not loaded. Train it first."}, status_code=500)

    cleaned = preprocess_text(pd.Series([text]))[0]
    seq = TOKENIZER.texts_to_sequences([cleaned])
    seq_padded = pad_sequences(seq, maxlen=LSTM_MODEL.input_shape[1], padding="post")
    probs = LSTM_MODEL.predict(seq_padded)[0].tolist()
    label_map = {0: "negative", 1: "neutral", 2: "positive"}
    label_idx = int(np.argmax(probs))

    return {
        "model": "LSTM",
        "input": text,
        "prediction": label_map[label_idx],
        "probabilities": {
            "negative": probs[0],
            "neutral": probs[1],
            "positive": probs[2]
        }
    }


@app.get("/api/stats")
def stats():
    """Show sentiment distribution statistics from the datasets."""
    paths = [
        "1429_1.csv",
        "Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv",
        "Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv"
    ]
    df = prepare_dataset(paths)
    counts = df["sentiment"].value_counts().to_dict()
    return {"counts": counts, "total_reviews": len(df)}


@app.get("/api/trends")
def trends():
    """Show sentiment trends over time based on review dates."""
    paths = [
        "1429_1.csv",
        "Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv",
        "Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv"
    ]

    dfs = [pd.read_csv(p, low_memory=False) for p in paths if os.path.exists(p)]
    raw = pd.concat(dfs, ignore_index=True)

    if "reviews.date" not in raw.columns or "reviews.rating" not in raw.columns:
        return {"trends": []}

    raw = raw[["reviews.date", "reviews.rating"]].dropna()
    raw["reviews.date"] = pd.to_datetime(raw["reviews.date"], errors="coerce")
    raw = raw.dropna(subset=["reviews.date"])
    raw["sentiment"] = raw["reviews.rating"].apply(lambda r: "positive" if r >= 4 else ("neutral" if r == 3 else "negative"))
    raw["date"] = raw["reviews.date"].dt.date
    counts = raw.groupby(["date", "sentiment"]).size().unstack(fill_value=0).reset_index()
    records = counts.to_dict(orient="records")

    return {"trends": records}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    """Upload a CSV file and analyze sentiment distribution."""
    contents = await file.read()
    temp_path = "temp_upload.csv"
    with open(temp_path, "wb") as f:
        f.write(contents)

    df = prepare_dataset([temp_path])
    counts = df["sentiment"].value_counts().to_dict()

    return {"uploaded_file": file.filename, "counts": counts, "total_reviews": len(df)}


# ================================
# RUN APP
# ================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
