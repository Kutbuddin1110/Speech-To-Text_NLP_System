import os
import pandas as pd
import joblib
import nltk
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Download NLTK resources
nltk.download("punkt")
nltk.download("stopwords")

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# ------------------------------
# TEXT CLEANING FUNCTION
# ------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stopwords.words('english')]
    return " ".join(tokens)


# ------------------------------
# GENERIC TRAINING FUNCTION
# ------------------------------
def train_model(dataset_path, model_name):

    print(f"\n🔹 Training {model_name} model...")

    df = pd.read_csv(dataset_path)

    # Clean text
    df["text"] = df["text"].astype(str).apply(clean_text)

    X = df["text"]
    y = df["label"]

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    # Evaluation
    y_pred = model.predict(X_test_vec)

    print(f"\n📊 {model_name.upper()} MODEL RESULTS")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    # Save models
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, f"models/{model_name}.pkl")
    joblib.dump(vectorizer, f"models/{model_name}_vectorizer.pkl")

    print(f"✅ {model_name} model saved successfully!")


# ------------------------------
# MAIN EXECUTION
# ------------------------------
if __name__ == "__main__":

    train_model("data/sentiment_dataset.csv", "sentiment")
    train_model("data/emotion_dataset.csv", "emotion")
    train_model("data/intent_dataset.csv", "intent")

    print("\n🎉 All models trained successfully!")