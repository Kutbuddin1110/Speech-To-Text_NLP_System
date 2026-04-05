import streamlit as st
import whisper
import librosa
import numpy as np
import tempfile
import matplotlib.pyplot as plt
import json

from utils.preprocess import clean_text
from utils.predict import predict

# ------------------------------
# PAGE CONFIG
# ------------------------------
st.set_page_config(layout="wide")

# ------------------------------
# 🎨 PREMIUM CSS (MATCH IMAGE)
# ------------------------------
st.markdown("""
<style>

body {
    background: linear-gradient(135deg, #0f172a, #1e3a5f);
    color: white;
}

.main {
    padding: 20px;
}

/* Card style */
.card {
    background: rgba(255,255,255,0.05);
    padding: 18px;
    border-radius: 18px;
    backdrop-filter: blur(14px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
}

/* Upload box */
.upload-box {
    border: 2px dashed rgba(255,255,255,0.2);
    padding: 40px;
    border-radius: 15px;
    text-align: center;
}

/* Sidebar items */
.file-item {
    padding: 10px;
    border-radius: 10px;
    background: rgba(255,255,255,0.05);
    margin-bottom: 8px;
}

/* Result cards */
.result-card {
    text-align: center;
    padding: 15px;
    border-radius: 15px;
    background: rgba(255,255,255,0.05);
}

/* Text boxes */
.text-box {
    padding: 12px;
    border-radius: 10px;
    background: rgba(255,255,255,0.05);
}

</style>
""", unsafe_allow_html=True)

# ------------------------------
# SESSION STATE
# ------------------------------
if "files" not in st.session_state:
    st.session_state.files = []

if "active" not in st.session_state:
    st.session_state.active = None

# ------------------------------
# MODEL
# ------------------------------
@st.cache_resource
def load_model():
    return whisper.load_model("base")

model = load_model()

# ------------------------------
# HEADER
# ------------------------------
st.title("💡 Speech-to-Text NLP Analysis System")
st.caption("AI-Powered Audio Intelligence")

# ------------------------------
# LAYOUT
# ------------------------------
left, center, right = st.columns([2,5,3])

# ==============================
# LEFT PANEL (Audio Hub)
# ==============================
with left:
    st.markdown("### 🎧 Audio Hub")

    uploaded = st.file_uploader("Upload", type=["mp3","wav","m4a","ogg","flac"])

    if uploaded:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(uploaded.read())
            path = tmp.name

        with st.spinner("Processing..."):
            audio, sr = librosa.load(path, sr=16000)
            audio = audio.astype(np.float32)

            result = model.transcribe(audio)
            text = result["text"]

            cleaned = clean_text(text)

            sentiment, s_conf = predict(cleaned, "sentiment")
            emotion, e_conf = predict(cleaned, "emotion")
            intent, i_conf = predict(cleaned, "intent")

            st.session_state.files.append({
                "name": uploaded.name,
                "audio": audio,
                "text": text,
                "cleaned": cleaned,
                "sentiment": sentiment,
                "emotion": emotion,
                "intent": intent,
                "conf": [s_conf, e_conf, i_conf]
            })

            st.session_state.active = len(st.session_state.files) - 1

    st.markdown("### Uploaded Files")

    for i, f in enumerate(st.session_state.files):
        if st.button(f"📄 {f['name']}", key=i):
            st.session_state.active = i

# ==============================
# CENTER PANEL
# ==============================
with center:

    if st.session_state.active is None:
        st.info("Upload a file to start")
    else:
        data = st.session_state.files[st.session_state.active]

        st.markdown(f"### 📄 {data['name']}")

        st.audio(data["audio"], sample_rate=16000)

        # 🔥 IMPROVED WAVEFORM (MATCH IMAGE)
        fig, ax = plt.subplots(figsize=(8,3))
        ax.plot(data["audio"], color="#38bdf8", linewidth=1.5)
        ax.fill_between(range(len(data["audio"])), data["audio"], color="#38bdf8", alpha=0.3)

        ax.set_facecolor("#0f172a")
        ax.axis("off")

        st.pyplot(fig)

        # ------------------------------
        # RESULT CARDS
        # ------------------------------
        c1, c2, c3 = st.columns(3)

        labels = ["Sentiment", "Emotion", "Intent"]
        values = [data["sentiment"], data["emotion"], data["intent"]]

        for i, col in enumerate([c1,c2,c3]):
            with col:
                st.markdown(f"""
                <div class="result-card">
                    <h4>{labels[i]}</h4>
                    <h2>{values[i].upper()}</h2>
                    <p>{data['conf'][i]*100:.1f}% confidence</p>
                </div>
                """, unsafe_allow_html=True)

                st.progress(float(data["conf"][i]))

# ==============================
# RIGHT PANEL
# ==============================
with right:

    if st.session_state.active is not None:
        data = st.session_state.files[st.session_state.active]

        st.markdown("### 📝 Transcribed Text")
        st.markdown(f"<div class='text-box'>{data['text']}</div>", unsafe_allow_html=True)

        st.markdown("### 🧠 Cleaned Text")
        st.markdown(f"<div class='text-box'>{data['cleaned']}</div>", unsafe_allow_html=True)

        st.markdown("### 📊 Multi-file Comparison")

        table = [
            {
                "File": f["name"],
                "Sentiment": f["sentiment"],
                "Emotion": f["emotion"],
                "Intent": f["intent"]
            }
            for f in st.session_state.files
        ]

        st.dataframe(table)

        st.download_button(
            "📥 Download Report",
            data=json.dumps(table, indent=2),
            file_name="report.json"
        )