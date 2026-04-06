import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from sklearn.metrics import confusion_matrix

# 1. CONFUSION MATRIX 

def plot_confusion_matrix(model, X_test, y_test, name):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.savefig(f"images/{name}_cm.png")
    plt.close()

# 2. CLASS DISTRIBUTION 

def plot_class_distribution(csv_path, name):
    df = pd.read_csv(csv_path)
    counts = df["label"].value_counts()

    plt.figure()
    plt.pie(counts, labels=counts.index, autopct='%1.1f%%')
    plt.title(f"{name} Class Distribution")

    plt.savefig(f"images/{name}_distribution.png")
    plt.close()

# 3. CONFIDENCE HISTOGRAM

def plot_confidence_hist(model, X_test, name):
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test)
        confidence = np.max(probs, axis=1)

        plt.figure()
        plt.hist(confidence, bins=20)

        plt.title(f"{name} Confidence Distribution")
        plt.xlabel("Confidence")
        plt.ylabel("Frequency")

        plt.savefig(f"images/{name}_confidence_hist.png")
        plt.close()

# 4. CONFIDENCE BOX PLOT

def plot_confidence_box(model, X_test, name):
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test)
        confidence = np.max(probs, axis=1)

        plt.figure()
        plt.boxplot(confidence)

        plt.title(f"{name} Confidence Spread")

        plt.savefig(f"images/{name}_confidence_box.png")
        plt.close()

# 5. CONFIDENCE SCATTER

def plot_confidence_scatter(model, X_test, name):
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test)
        confidence = np.max(probs, axis=1)

        plt.figure()
        plt.scatter(range(len(confidence)), confidence, alpha=0.5)

        plt.title(f"{name} Confidence Scatter")
        plt.xlabel("Sample Index")
        plt.ylabel("Confidence")

        plt.savefig(f"images/{name}_confidence_scatter.png")
        plt.close()

# 6. TOP WORDS PER CLASS

def plot_top_words(vectorizer, model, name, n=10):
    if hasattr(model, "coef_"):
        feature_names = vectorizer.get_feature_names_out()

        for i, label in enumerate(model.classes_):
            top_indices = model.coef_[i].argsort()[-n:]

            words = [feature_names[j] for j in top_indices]

            plt.figure()
            plt.barh(words, range(n))

            plt.title(f"{name} - Top Words ({label})")

            plt.savefig(f"images/{name}_top_words_{label}.png")
            plt.close()

# 7. ACCURACY COMPARISON

def plot_accuracy():
    models = ["Sentiment", "Emotion", "Intent"]
    accuracy = [0.86, 0.63, 0.91]  # update if needed

    plt.figure()
    plt.bar(models, accuracy)

    plt.title("Model Accuracy Comparison")
    plt.ylabel("Accuracy")

    for i, v in enumerate(accuracy):
        plt.text(i, v + 0.01, str(round(v, 2)), ha='center')

    plt.savefig("images/accuracy_comparison.png")
    plt.close()