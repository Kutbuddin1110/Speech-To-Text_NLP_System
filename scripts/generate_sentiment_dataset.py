import random
import pandas as pd

positive_phrases = [
    "I am very happy with this service",
    "This is absolutely amazing and satisfying",
    "I really enjoyed the experience",
    "Everything works perfectly",
    "I am extremely pleased with the results"
]

negative_phrases = [
    "I am very unhappy with this service",
    "This is extremely disappointing and frustrating",
    "I hate how this works",
    "This is a terrible experience",
    "I am not satisfied at all"
]

neutral_phrases = [
    "The service is okay overall",
    "It works as expected",
    "Nothing special about this",
    "Average experience",
    "It is fine"
]

data = []

for _ in range(700):
    data.append([random.choice(positive_phrases), "positive"])

for _ in range(700):
    data.append([random.choice(negative_phrases), "negative"])

for _ in range(700):
    data.append([random.choice(neutral_phrases), "neutral"])

random.shuffle(data)

df = pd.DataFrame(data, columns=["text", "label"])
df.to_csv("data/sentiment_dataset.csv", index=False)

print("Sentiment dataset created with", len(df), "samples")