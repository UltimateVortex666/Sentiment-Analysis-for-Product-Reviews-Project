import os
import joblib
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.utils import to_categorical
from data_utils import prepare_dataset

# ================= CONFIG =================
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

PATHS = [
    "1429_1.csv",
    "Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv",
    "Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv"
]
# ==========================================


def train_tfidf_nb(df):
    print("\n🧠 Training TF-IDF + Naive Bayes model...")
    X = df['text_clean']
    y = df['sentiment']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    vect = TfidfVectorizer(max_features=30000, ngram_range=(1, 2))
    selector = SelectKBest(chi2, k=8000)
    clf = MultinomialNB(alpha=0.5)

    pipe = Pipeline([
        ('tfidf', vect),
        ('sel', selector),
        ('nb', clf)
    ])

    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    print('✅ Naive Bayes Accuracy:', accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))

    joblib.dump(pipe, os.path.join(MODEL_DIR, 'tfidf_nb.joblib'))
    print('💾 Saved tfidf_nb.joblib')


def train_lstm(df, max_words=30000, maxlen=200, embed_dim=128, epochs=3, batch_size=256):
    print("\n🧠 Training LSTM model...")
    texts = df['text_clean'].tolist()
    labels = df['sentiment'].map({'negative': 0, 'neutral': 1, 'positive': 2}).values

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.15, random_state=42, stratify=labels
    )

    tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
    tokenizer.fit_on_texts(X_train)

    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    X_train_pad = pad_sequences(X_train_seq, maxlen=maxlen, padding='post', truncating='post')
    X_test_pad = pad_sequences(X_test_seq, maxlen=maxlen, padding='post', truncating='post')

    y_train_cat = to_categorical(y_train, num_classes=3)
    y_test_cat = to_categorical(y_test, num_classes=3)

    model = Sequential()
    model.add(Embedding(max_words, embed_dim, input_length=maxlen))
    model.add(Bidirectional(LSTM(128, return_sequences=False)))
    model.add(Dropout(0.4))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(3, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    model.fit(
        X_train_pad, y_train_cat,
        validation_data=(X_test_pad, y_test_cat),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    loss, acc = model.evaluate(X_test_pad, y_test_cat, verbose=0)
    print(f'✅ LSTM Test Accuracy: {acc:.4f}')

    # Save tokenizer and model
    joblib.dump(tokenizer, os.path.join(MODEL_DIR, 'tokenizer.joblib'))
    model.save(os.path.join(MODEL_DIR, 'lstm_model.keras'))
    print('💾 Saved tokenizer.joblib and lstm_model.keras')


if __name__ == '__main__':
    print('📦 Preparing dataset...')
    df = prepare_dataset(PATHS)
    print('📊 Class distribution:\n', df['sentiment'].value_counts())

    # Train models
    train_tfidf_nb(df)
    train_lstm(df)
