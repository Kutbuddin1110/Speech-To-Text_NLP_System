import os
import pandas as pd
import joblib
import nltk
import re
import time

from tqdm import tqdm
from rich.console import Console
from rich.progress import track

from visualize import plot_confusion_matrix, plot_confidence_hist, plot_confidence_box, plot_confidence_scatter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

# nltk.download("punkt")
# nltk.download("stopwords")
# nltk.download("wordnet")

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

console = Console()
lemmatizer = WordNetLemmatizer()

# STOPWORDS FIX
stop_words = set(stopwords.words('english'))
stop_words.discard("not")
stop_words.discard("no")

# NEGATION WORDS
NEGATION_WORDS = {"not", "no", "never", "n't"}

# TEXT CLEANING FUNCTION 

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)

    tokens = word_tokenize(text)

    processed_tokens = []
    negation = False
    window = 0

    for w in tokens:
        if w in NEGATION_WORDS:
            negation = True
            window = 3
            processed_tokens.append(w)
            continue

        if negation and window > 0:
            processed_tokens.append("NOT_" + w)
            window -= 1
        else:
            processed_tokens.append(w)

        if window == 0:
            negation = False

    tokens = [
        lemmatizer.lemmatize(w)
        for w in processed_tokens
        if w not in stop_words and len(w) > 2
    ]

    return " ".join(tokens)

# GENERIC TRAINING FUNCTION

def train_model(dataset_path, model_name):

    console.print(f"\n[bold yellow]🔹 Training {model_name} model...[/bold yellow]")

    start_total = time.time()

    df = pd.read_csv(dataset_path)
    df.columns = df.columns.str.strip()

    # Clean text
    console.print("[cyan]🔹 Cleaning text data...[/cyan]")
    tqdm.pandas()
    df["text"] = df["text"].astype(str).progress_apply(clean_text)

    # IMPROVED BALANCING (USE MAX AVAILABLE DATA)
    # if model_name in ["emotion", "intent"]:
    #     label_counts = df["label"].value_counts()
    #     min_samples = label_counts.min()

    #     df = df.groupby('label', group_keys=False).sample(
    #         n=min_samples,
    #         replace=False
    #     ).reset_index(drop=True)

    steps = ["Splitting Data", "Vectorizing", "Training Model", "Evaluating"]

    for step in track(steps, description=f"{model_name} pipeline..."):

        if step == "Splitting Data":
            X = df["text"]
            y = df["label"]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

        elif step == "Vectorizing":

            # WORD N-GRAMS
            word_vectorizer = TfidfVectorizer(
                max_features=10000,
                ngram_range=(1,3),
                min_df=2,
                max_df=0.9,
                sublinear_tf=True
            )

            # CHAR N-GRAMS
            char_vectorizer = TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3,5),
                max_features=5000
            )

            # COMBINE BOTH
            vectorizer = FeatureUnion([
                ("word", word_vectorizer),
                ("char", char_vectorizer)
            ])

            X_train_vec = vectorizer.fit_transform(X_train)
            X_test_vec = vectorizer.transform(X_test)

        elif step == "Training Model":
            start = time.time()

            if model_name in ["emotion", "sentiment"]:
                base_model = LinearSVC(
                    class_weight="balanced",
                    C=1.5
                )

                model = CalibratedClassifierCV(base_model, cv=3)
            else:
                model = LogisticRegression(max_iter=1000)

            model.fit(X_train_vec, y_train)

            end = time.time()
            console.print(f"[green]⏱ Training time: {round(end-start,2)} sec[/green]")

        elif step == "Evaluating":
            y_pred = model.predict(X_test_vec)

    console.print(f"\n[bold green]{model_name.upper()} MODEL RESULTS[/bold green]")
    console.print(f"[blue]Accuracy:[/blue] {accuracy_score(y_test, y_pred)}")
    console.print("\n[magenta]Classification Report:[/magenta]\n")
    console.print(classification_report(y_test, y_pred, digits=4))

    # Graphs
    plot_confusion_matrix(model, X_test_vec, y_test, model_name)
    plot_confidence_hist(model, X_test_vec, model_name)
    plot_confidence_box(model, X_test_vec, model_name)
    plot_confidence_scatter(model, X_test_vec, model_name)

    # Save
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, f"models/{model_name}.pkl")
    joblib.dump(vectorizer, f"models/{model_name}_vectorizer.pkl")

    console.print(f"[bold green]{model_name} model saved successfully![/bold green]")

    end_total = time.time()
    console.print(f"[bold cyan]Total time: {round(end_total-start_total,2)} sec[/bold cyan]")

# MAIN

if __name__ == "__main__":

    train_model("data/sentiment_dataset.csv", "sentiment")
    train_model("data/emotion_dataset.csv", "emotion")
    train_model("data/intent_dataset.csv", "intent")

    console.print("\n[bold green]All models trained successfully![/bold green]")