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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

# Download NLTK resources
# nltk.download("punkt")
# nltk.download("punkt_tab")   
# nltk.download("stopwords")
# nltk.download("wordnet")

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Initialize tools
console = Console()
lemmatizer = WordNetLemmatizer()

# TEXT CLEANING FUNCTION 

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)

    tokens = word_tokenize(text)

    tokens = [
        lemmatizer.lemmatize(w)
        for w in tokens
        if w not in stopwords.words('english') and len(w) > 2
    ]

    return " ".join(tokens)

# GENERIC TRAINING FUNCTION

def train_model(dataset_path, model_name):

    console.print(f"\n[bold yellow]🔹 Training {model_name} model...[/bold yellow]")

    start_total = time.time()

    df = pd.read_csv(dataset_path)
    df.columns = df.columns.str.strip()
    print("COLUMNS:", list(df.columns))
    print(df.head())
    
    # Clean text with progress bar
    console.print("[cyan]🔹 Cleaning text data...[/cyan]")
    tqdm.pandas()
    df["text"] = df["text"].astype(str).progress_apply(clean_text)

    # Balance dataset
    df = df.groupby('label', group_keys=False).apply(
        lambda x: x.sample(min(len(x), 2000)),
        # include_groups=False
    ).reset_index(drop=True)

    steps = [
        "Splitting Data",
        "Vectorizing",
        "Training Model",
        "Evaluating"
    ]

    for step in track(steps, description=f"{model_name} pipeline..."):

        if step == "Splitting Data":
            X = df["text"]
            y = df["label"]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

        elif step == "Vectorizing":
            vectorizer = TfidfVectorizer(
                max_features=10000,
                ngram_range=(1,2),
                min_df=2,
                max_df=0.9,
                sublinear_tf=True
            )

            X_train_vec = vectorizer.fit_transform(X_train)
            X_test_vec = vectorizer.transform(X_test)

        elif step == "Training Model":
            start = time.time()

            # HYBRID MODEL SELECTION
            if model_name in ["emotion", "sentiment"]:
                base_model = LinearSVC(
                    class_weight="balanced",
                    C=1.5
                )

                model = CalibratedClassifierCV(
                    base_model,
                    cv=3
                )
            else:
                model = LogisticRegression(max_iter=1000)

            model.fit(X_train_vec, y_train)

            end = time.time()
            console.print(f"[green]⏱ Training time: {round(end-start,2)} sec[/green]")

        elif step == "Evaluating":
            y_pred = model.predict(X_test_vec)

    console.print(f"\n[bold green] {model_name.upper()} MODEL RESULTS[/bold green]")
    console.print(f"[blue]Accuracy:[/blue] {accuracy_score(y_test, y_pred)}")
    console.print("\n[magenta]Classification Report:[/magenta]\n")
    console.print(classification_report(y_test, y_pred, digits=4))

    plot_confusion_matrix(model, X_test_vec, y_test, model_name)
    plot_confidence_hist(model, X_test_vec, model_name)
    plot_confidence_box(model, X_test_vec, model_name)
    plot_confidence_scatter(model, X_test_vec, model_name)
    
    # Save models
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, f"models/{model_name}.pkl")
    joblib.dump(vectorizer, f"models/{model_name}_vectorizer.pkl")

    console.print(f"[bold green] {model_name} model saved successfully![/bold green]")

    end_total = time.time()
    console.print(f"[bold cyan] Total time: {round(end_total-start_total,2)} sec[/bold cyan]")

# MAIN EXECUTION

if __name__ == "__main__":

    train_model("data/sentiment_dataset.csv", "sentiment")
    train_model("data/emotion_dataset.csv", "emotion")
    train_model("data/intent_dataset.csv", "intent")

    console.print("\n [bold green]All models trained successfully![/bold green]")