## Amazon Product Review Sentiment Dashboard

An end-to-end sentiment analysis project for Amazon product reviews with:
- A FastAPI backend serving a modern dashboard UI
- Two models: TF‑IDF + Naive Bayes and an LSTM (Keras)
- Interactive charts (overall distribution and trends) and a model selector to choose which model to use for prediction


### Project Structure
- `app.py` — FastAPI app, API routes, and model loading
- `templates/dashboard.html` — UI (served as a static HTML page)
- `static/styles.css` — Styles for the dashboard
- `data_utils.py` — Data loading and preprocessing utilities (NLTK/SpaCy)
- `train_models.py` — Scripts to train Naive Bayes and LSTM models
- `models/` — Saved models and tokenizer


### Prerequisites
- Python 3.11 (recommended; works with 3.9–3.11)
- Windows, macOS, or Linux


### Installation (Windows PowerShell)
```bash
python -m venv .venv
.\.venv\Scripts\activate

# Core deps
pip install --upgrade pip
pip install "numpy<2" pandas scikit-learn joblib fastapi "uvicorn[standard]" nltk spacy tensorflow keras

# Optional: if you hit SciPy wheels issues on some environments, install pre-built wheels before sklearn
# pip install --only-binary :all: numpy==1.26.4 scipy==1.11.4

# SpaCy small English model (optional but recommended for faster/better preprocessing)
python -m spacy download en_core_web_sm
```


### Dataset
Place the CSVs in the project root (same folder as `app.py`):
- `1429_1.csv`
- `Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv`
- `Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv`

If you use different file names or locations, update the lists in `app.py` (`/api/stats`, `/api/trends`) and/or in `train_models.py`.


### First Run: Download NLTK Resources
`data_utils.py` will download what it needs on first import, including:
- `punkt`, `punkt_tab`, and `stopwords`

If you run into NLTK lookup errors, you can manually run:
```python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
```


### Train Models
This trains both TF‑IDF + Naive Bayes and the LSTM, saving artifacts into `models/`.
```bash
python train_models.py
```
Artifacts produced:
- `models/tfidf_nb.joblib` — Naive Bayes pipeline
- `models/tokenizer.joblib` — Keras tokenizer for LSTM
- `models/lstm_model.keras` — LSTM model (Keras native format)


### Start the App
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```
Open `http://localhost:8000`.

In the top input bar:
- Enter any review text
- Choose the model (Naive Bayes or LSTM)
- Click Analyze

You will see the predicted sentiment and the charts for distribution and trends.


### API Endpoints
- `GET /` — Dashboard UI
- `POST /api/predict` — Unified prediction endpoint
  - Body: `{ "text": string, "model": "nb" | "lstm" }` (model defaults to `nb`)
  - Response: `{ model, input, prediction, probabilities }`
- `GET /api/stats` — Distribution across sentiments from the datasets
- `GET /api/trends` — Time series of sentiments
- `POST /api/upload` — Upload a CSV and return sentiment distribution


### Common Issues & Fixes
- TensorFlow fails with NumPy 2.x: install NumPy 1.x
  - Error example: `AttributeError: _ARRAY_API not found` or "A module compiled using NumPy 1.x cannot be run in NumPy 2.x"
  - Fix: `pip install "numpy<2"`

- Missing NLTK resources (e.g., `punkt_tab`):
  - Fix: see NLTK section above or ensure `data_utils.py` runs once to download.

- LSTM save format error (Keras 3):
  - Use `.keras` extension with `model.save("models/lstm_model.keras")`


### Development Notes
- The dashboard is plain HTML/JS + Chart.js via CDN; no template engine is required
- `/` serves `templates/dashboard.html` directly using `FileResponse` for fast reloads
- Model selector is built into the UI; the backend routes the request accordingly in `/api/predict`


### License
This project is for educational purposes. Use at your own discretion.


