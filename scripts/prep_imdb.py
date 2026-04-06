import pandas as pd

# Load original IMDb dataset
df = pd.read_csv("raw_data/IMDB Dataset.csv")

# Rename columns
df = df.rename(columns={
    "review": "text",
    "sentiment": "label"
})

# Convert labels (kept for consistency)
df["label"] = df["label"].map({
    "positive": "positive",
    "negative": "negative"
})

# remove duplicates
df = df.drop_duplicates(subset=["text"])

print("Total samples after cleaning:", len(df))

# Save cleaned dataset
df.to_csv("data/sentiment_dataset.csv", index=False)

print("IMDb dataset prepared using FULL dataset!")