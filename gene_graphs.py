from visualize import (
    plot_class_distribution,
    plot_accuracy
)

plot_class_distribution("data/sentiment_dataset.csv", "sentiment")
plot_class_distribution("data/emotion_dataset.csv", "emotion")
plot_class_distribution("data/intent_dataset.csv", "intent")

plot_accuracy()