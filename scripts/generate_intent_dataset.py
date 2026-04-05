import random
import pandas as pd

intents = {
    "complaint": [
        "I am not happy with your service",
        "This is very frustrating",
        "I want to file a complaint",
        "Your service is terrible",
        "I am disappointed"
    ],
    "inquiry": [
        "Can you tell me my balance?",
        "What are your working hours?",
        "I want to know more about this",
        "How does this work?",
        "Can you explain this?"
    ],
    "request": [
        "I want to cancel my order",
        "Please process my refund",
        "I need to update my details",
        "Can you change my plan?",
        "I want to upgrade my account"
    ],
    "feedback": [
        "Great service, thank you",
        "I really liked this",
        "Good experience overall",
        "Very satisfied",
        "Keep up the good work"
    ],
    "greeting": [
        "Hello",
        "Hi there",
        "Good morning",
        "Hey",
        "Greetings"
    ],
    "support": [
        "I need help with my account",
        "Can you assist me?",
        "I am facing an issue",
        "Help me fix this problem",
        "I need support"
    ]
}

data = []

for label, phrases in intents.items():
    for _ in range(350):  # ~2000 total
        sentence = random.choice(phrases)
        data.append([sentence, label])

random.shuffle(data)

df = pd.DataFrame(data, columns=["text", "label"])
df.to_csv("data/intent_dataset.csv", index=False)

print("Intent dataset created with", len(df), "samples")