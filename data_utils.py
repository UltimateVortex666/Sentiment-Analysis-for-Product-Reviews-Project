import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy

# Download necessary NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

STOPWORDS = set(stopwords.words('english'))

# Try loading SpaCy model (small English)
try:
    nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser'])
except:
    nlp = None


def normalize_text(text: str) -> str:
    """Basic normalization: remove URLs, special chars, lowercase, trim"""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # remove URLs
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    # remove non-alphanumeric (keep spaces)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def lemmatize_spacy(text: str) -> str:
    """Lemmatize text using SpaCy (if available)"""
    if not nlp:
        return text
    doc = nlp(text)
    lemmas = [tok.lemma_ for tok in doc if not tok.is_stop and not tok.is_punct]
    return ' '.join(lemmas)


def preprocess_text(series: pd.Series) -> pd.Series:
    """Efficient text preprocessing: normalization + lemmatization/token filtering"""
    series = series.fillna('').astype(str)
    series = series.map(normalize_text)
    if nlp:
        # apply in chunks for speed
        CHUNK = 2000
        out = []
        for i in range(0, len(series), CHUNK):
            chunk = series.iloc[i:i+CHUNK].tolist()
            docs = nlp.pipe(chunk, batch_size=1000)
            out.extend([' '.join([t.lemma_ for t in d if not t.is_stop and not t.is_punct]) for d in docs])
        return pd.Series(out, index=series.index)
    else:
        # fallback: simple tokenization + stopword removal
        return series.map(lambda s: ' '.join([w for w in word_tokenize(s) if w not in STOPWORDS]))


def label_sentiment(rating: float) -> str:
    """Map numeric ratings to sentiment labels"""
    if rating >= 4:
        return 'positive'
    elif rating == 3:
        return 'neutral'
    else:
        return 'negative'


def load_and_concat(paths):
    """Load and combine multiple CSVs"""
    dfs = []
    for p in paths:
        try:
            df = pd.read_csv(p)
            dfs.append(df)
            print(f"Loaded {p} with {len(df)} rows.")
        except Exception as e:
            print(f"❌ Error loading {p}: {e}")
    return pd.concat(dfs, ignore_index=True)


def extract_reviews(df):
    """Extract text and rating columns, handling various dataset formats"""
    possible_text_cols = ['reviews.text', 'reviewText', 'text']
    possible_rating_cols = ['reviews.rating', 'rating', 'stars']

    text_col = next((c for c in possible_text_cols if c in df.columns), None)
    rating_col = next((c for c in possible_rating_cols if c in df.columns), None)

    if not text_col or not rating_col:
        raise ValueError("Text or rating column not found in dataset.")

    df = df[[text_col, rating_col]].rename(columns={text_col: 'text', rating_col: 'rating'})
    return df


def prepare_dataset(paths):
    """Main pipeline: load, preprocess, label sentiments"""
    df_all = load_and_concat(paths)
    df = extract_reviews(df_all)
    df['text_clean'] = preprocess_text(df['text'])
    df['sentiment'] = df['rating'].apply(label_sentiment)
    # Remove extremely short or empty after cleaning
    df = df[df['text_clean'].str.len() > 2].reset_index(drop=True)
    return df


if __name__ == '__main__':
    paths = [
        '1429_1.csv',
        'Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv',
        'Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv'
    ]
    df = prepare_dataset(paths)
    print('✅ Prepared', len(df), 'rows')
    print(df.head())
