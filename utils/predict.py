import joblib

def predict(text, model_name):
    model = joblib.load(f"models/{model_name}.pkl")
    vectorizer = joblib.load(f"models/{model_name}_vectorizer.pkl")

    text_vectorized = vectorizer.transform([text])

    # Prediction
    prediction = model.predict(text_vectorized)[0]

    # Confidence (IMPORTANT)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(text_vectorized)
        confidence = max(probs[0])
    else:
        # fallback (for models like SVM without probability)
        confidence = 1.0

    return prediction, round(confidence, 2)