import joblib

def predict(text, model_name):
    model = joblib.load(f"models/{model_name}.pkl")
    vectorizer = joblib.load(f"models/{model_name}_vectorizer.pkl")

    text_vectorized = vectorizer.transform([text])
    prediction = model.predict(text_vectorized)

    return prediction[0]