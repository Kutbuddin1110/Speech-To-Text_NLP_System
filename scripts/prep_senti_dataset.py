import pandas as pd

# load dataset
df = pd.read_csv("data/sentiment140.csv", encoding="latin-1", header=None)

# rename columns
df.columns = [
    "sentiment",
    "id",
    "date",
    "query",
    "user",
    "text"
]

# keep only text + sentiment
df = df[["text", "sentiment"]]

# convert labels
df["label"] = df["sentiment"].map({
    0: "negative",
    4: "positive"
})

df = df[["text", "label"]]

print("Dataset size:", len(df))

# optional sampling
df = df.sample(30000)

df.to_csv("data/sentiment_dataset.csv", index=False)

print("sentiment_dataset.csv created")