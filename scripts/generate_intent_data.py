import random
import pandas as pd
import os

# CONFIG

OUTPUT_PATH = "data/intent_dataset.csv"
SAMPLES_PER_INTENT = 2000

# SEED PHRASES

INTENT_SEEDS = {
    "greeting": [
        "hey", "hello", "hi", "hey bro", "yo", "good morning", "what's up"
    ],
    "goodbye": [
        "bye", "see you", "goodbye", "catch you later", "see ya"
    ],
    "question": [
        "what is this", "why is this happening", "how does this work",
        "where are you", "when will it happen"
    ],
    "statement": [
        "this is nice", "i am working", "it is raining",
        "today was tiring", "i feel okay"
    ],
    "command": [
        "do this", "open the app", "play music",
        "start recording", "stop that"
    ],
    "request": [
        "can you help me", "please assist me",
        "i need help", "can you do this", "help me out"
    ],
    "emotion_expression": [
        "i am happy", "i feel sad", "i am angry",
        "this is frustrating", "i am excited"
    ],
    "opinion": [
        "this is amazing", "this is bad", "i love this",
        "i hate this", "this is boring"
    ],
    "gratitude": [
        "thank you", "thanks a lot", "appreciate it",
        "thanks bro", "many thanks"
    ],
    "apology": [
        "sorry", "my bad", "i apologize",
        "sorry about that", "forgive me"
    ],
    "confusion": [
        "i don't understand", "this is confusing",
        "what is going on", "i am lost", "this makes no sense"
    ]
}

# VARIATION HELPERS

prefixes = ["", "hey", "bro", "dude", "man", "please", "uh"]
suffixes = ["", "please", "now", "quickly", "right now", "lol", ""]

def generate_variation(text):
    return f"{random.choice(prefixes)} {text} {random.choice(suffixes)}".strip()

# DATA GENERATION

data = []

for intent, phrases in INTENT_SEEDS.items():
    for _ in range(SAMPLES_PER_INTENT):
        base = random.choice(phrases)
        variation = generate_variation(base)
        data.append((variation, intent))

# CREATE DATAFRAME

df = pd.DataFrame(data, columns=["text", "label"])

# Shuffle dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# SAVE

os.makedirs("data", exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)

print("Dataset generated successfully!")
print("Total samples:", len(df))
print("\nDistribution:")
print(df["label"].value_counts())