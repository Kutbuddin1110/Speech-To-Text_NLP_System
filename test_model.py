from utils.predict import predict

text = "I am very unhappy with your service"

print("Sentiment:", predict(text, "sentiment"))
print("Emotion:", predict(text, "emotion"))
print("Intent:", predict(text, "intent"))