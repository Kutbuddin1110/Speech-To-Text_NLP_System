# 🎙️ Speech-to-Text NLP Analysis System

An end-to-end AI system that converts speech into text and performs **Sentiment Analysis, Emotion Detection, and Intent Classification** using classical machine learning techniques.

---

## 🚀 Overview

This project combines **speech recognition and natural language processing (NLP)** to analyze user input from audio files.

The pipeline:

1. Converts speech → text
2. Processes and cleans text
3. Applies machine learning models
4. Outputs sentiment, emotion, and intent predictions with confidence

---

## ✨ Features

* 🎧 Audio input support (multiple formats)
* 🧠 Speech-to-text using Whisper
* 📊 Multi-level NLP analysis:

  * Sentiment (Positive / Negative)
  * Emotion (6 core emotions)
  * Intent detection
* ⚡ Real-time predictions
* 📂 File history system (UI-based)
* 📈 Model evaluation with metrics and visualizations
* 🧩 Modular architecture (training, inference, visualization separated)

---

## 🧠 Models Used

### 🔹 Sentiment Analysis

* Classes: Positive, Negative
* Algorithm: Support Vector Machine (SVM) with calibration
* Accuracy: ~86–90%

---

### 🔹 Emotion Detection

* Classes:

  * Joy
  * Anger
  * Sadness
  * Fear
  * Love
  * Surprise
* Algorithm: SVM with calibration
* Accuracy: ~60–65%

---

### 🔹 Intent Classification

* Dataset: Custom Dataset (subset)
* Algorithm: Logistic Regression
* Accuracy: ~90%+

---

## 🏗️ System Architecture

```
Audio Input
   ↓
Speech-to-Text (Whisper)
   ↓
Text Preprocessing
   ↓
TF-IDF Vectorization
   ↓
ML Models
   ├── Sentiment Model
   ├── Emotion Model
   └── Intent Model
   ↓
Predictions + Confidence
   ↓
Streamlit UI
```

---

## 📂 Project Structure

```
project/
│
├── app.py                      # Streamlit UI
├── train_model.py              # Model training pipeline
├── visualize.py                # Visualization utilities
├── generate_graphs.py          # Graph generation script
│
├── scripts/
│   ├── prep_emo_dataset.py             # Emotion dataset preparation
│   ├── prep_imdb_dataset.py            # Seniment dataset preparation
│   └── generate_intent_dataset.py      # Intent dataset preparation
│
├── utils/
│   ├── preprocess.py           # Text cleaning functions
│   ├── predict.py              # Prediction logic
│   └── speech_to_text.py       # Whisper integration
│
├── data/
│   ├── sentiment_dataset.csv
│   ├── emotion_dataset.csv
│   └── intent_dataset.csv
│
├── raw_data/
│   ├── IMDB dataset.csv
│   ├── goemotiom_1.csv
│   ├── goemotiom_2.csv
│   └── goemotiom_3.csv
│
├── models/
│   ├── sentiment.pkl
│   ├── emotion.pkl
│   └── intent.pkl
│  
├── images/ 
│   └── *.png (generated graphs)
│
└── requirements.txt
```

---

## ⚙️ Installation

```bash
git clone https://github.com/Kutbuddin1110/Speech-To-Text_NLP_System.git
cd Speech-To-Text_NLP_System

python -m venv venv
venv\Scripts\activate   # Windows

pip install -r requirements.txt
```

---

## ▶️ Usage

### 1. Prepare Emotion Dataset

```bash
python prep_emo_dataset.py
```

### 2. Train Models

```bash
python train_model.py
```

### 3. Generate Graphs

```bash
python generate_graphs.py
```

### 4. Run Application

```bash
streamlit run app.py
```

---

## 📊 Model Evaluation

Each model is evaluated using:

* Accuracy
* Precision
* Recall
* F1-score

Additionally, the project supports:

* Confusion Matrix
* Class Distribution
* Confidence Analysis
* Feature Importance (Top Words)

---

## 📈 Visualization Support

The system includes a dedicated visualization module that can generate:

* Confusion matrices (heatmaps)
* Class distribution charts (donut charts)
* Confidence distribution plots (histogram, box, scatter)
* Model comparison charts
* Top contributing words per class

---

## 🧪 Dataset Details

### Sentiment Dataset

* Source: IMDb movie reviews
* Format: text, label
* Balanced dataset

---

### Emotion Dataset

* Source: GoEmotions dataset (Google Research)
* Reduced to 6 core emotions
* Multi-label data converted to single-label
* Cleaned and balanced

---

### Intent Dataset

* Source: CLINC150
* Reduced to relevant intents
* Balanced for training

---

## 🧹 Preprocessing Steps

* Lowercasing
* Removing special characters
* Tokenization
* Stopword removal
* Lemmatization

---

## 🔧 Key Improvements Implemented

* Reduced emotion classes for better performance
* Switched to SVM for improved text classification
* Added model calibration for confidence scores
* Optimized TF-IDF vectorization
* Implemented dataset balancing
* Added visualization module for explainability

---

## ⚡ Performance Summary

| Model     | Accuracy |
| --------- | -------- |
| Sentiment | ~86%     |
| Emotion   | ~63%     |
| Intent    | ~91%     |

---

## 🔮 Future Enhancements

* Upgrade to transformer-based models (BERT)
* Real-time streaming audio processing
* Deployment on cloud platforms
* Interactive dashboards with Plotly
* Multi-language support

---

## 💯 Key Learnings

* Importance of dataset preprocessing and label design
* Trade-offs between model complexity and performance
* Benefits of SVM for text classification
* Need for model explainability through visualization
* Modular system design for scalability

---

## 👨‍💻 Author

Kutbuddin Attarwala

---

## ⭐ Support

If you found this project useful, consider giving it a star ⭐
