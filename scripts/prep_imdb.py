import pandas as pd

# Load original IMDb dataset
df = pd.read_csv("raw_data/IMDB Dataset.csv")

# Rename columns
df = df.rename(columns={
    "review": "text",
    "sentiment": "label"
})

# Convert labels
df["label"] = df["label"].map({
    "positive": "positive",
    "negative": "negative"
})

# Optional: reduce size (faster training)
df = df.sample(20000, random_state=42)

# Save cleaned dataset
df.to_csv("data/sentiment_dataset.csv", index=False)

print("✅ IMDb dataset prepared!")