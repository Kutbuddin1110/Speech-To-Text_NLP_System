import pandas as pd

# Load the three GoEmotions files
df1 = pd.read_csv("data/goemotions_1.csv")
df2 = pd.read_csv("data/goemotions_2.csv")
df3 = pd.read_csv("data/goemotions_3.csv")

# Merge datasets
df = pd.concat([df1, df2, df3], ignore_index=True)

print("Total rows before filtering:", len(df))


# 12 target emotions
target_emotions = [
    "admiration",
    "amusement",
    "anger",
    "approval",
    "confusion",
    "curiosity",
    "disappointment",
    "fear",
    "gratitude",
    "joy",
    "sadness",
    "neutral"
]

# Keep only relevant columns
columns = ["text"] + target_emotions
df = df[columns]

rows = []

# Convert multi-label dataset → single label
for _, row in df.iterrows():
    
    labels = []
    
    for emotion in target_emotions:
        if row[emotion] == 1:
            labels.append(emotion)

    # Keep only rows with single emotion
    if len(labels) == 1:
        rows.append({
            "text": row["text"],
            "label": labels[0]
        })


clean_df = pd.DataFrame(rows)

print("Rows after filtering:", len(clean_df))

# Save cleaned dataset
clean_df.to_csv("data/emotion_dataset.csv", index=False)

print("emotion_dataset.csv created successfully")