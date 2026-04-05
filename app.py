import streamlit as st
import tempfile
import whisper
import librosa
import numpy as np
import matplotlib.pyplot as plt

from utils.preprocess import clean_text
from utils.predict import predict

# ------------------------------
# PAGE CONFIG
# ------------------------------
st.set_page_config(
    page_title="Speech NLP Analyzer",
    page_icon="🎙",
    layout="wide"
)

# ------------------------------
# DARK UI STYLING (FORCE DARK)
# ------------------------------
st.markdown("""
<style>
body {
    background-color: #0f172a;
}
.block-container {
    padding-top: 1rem;
}
.card {
    background-color: #1f2937;
    padding: 20px;
    border-radius: 15px;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------
# TITLE
# ------------------------------
st.title("🎙 Speech-to-Text NLP Analysis System")
st.markdown("### 🤖 AI-Powered Audio Intelligence")

# ------------------------------
# LOAD MODEL
# ------------------------------
@st.cache_resource
def load_model():
    return whisper.load_model("base")

model = load_model()

# ------------------------------
# SESSION STORAGE (HISTORY)
# ------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# ------------------------------
# LAYOUT
# ------------------------------
left, center, right = st.columns([1,2,1])

# ------------------------------
# LEFT PANEL (UPLOAD)
# ------------------------------
with left:
    st.markdown("### 🎧 Audio Hub")

    uploaded_file = st.file_uploader(
        "Upload Audio",
        type=["mp3","wav","m4a","flac","ogg"]
    )

# ------------------------------
# MAIN PROCESSING
# ------------------------------
if uploaded_file:

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    with st.spinner("Processing..."):

        audio, sr = librosa.load(temp_path, sr=16000)
        audio = audio.astype(np.float32)

        result = model.transcribe(audio)
        text = result["text"]

        cleaned = clean_text(text)

        sentiment, s_conf = predict(cleaned, "sentiment")
        emotion, e_conf = predict(cleaned, "emotion")
        intent, i_conf = predict(cleaned, "intent")

        # Save to history
        st.session_state.history.insert(0, {
            "file": uploaded_file.name,
            "text": text,
            "cleaned": cleaned,
            "sentiment": sentiment,
            "emotion": emotion,
            "intent": intent,
            "conf": (s_conf, e_conf, i_conf)
        })

# ------------------------------
# CENTER PANEL (LATEST RESULT)
# ------------------------------
with center:

    if st.session_state.history:

        latest = st.session_state.history[0]

        st.markdown(f"### 📄 {latest['file']}")

        # Waveform
        fig, ax = plt.subplots()
        ax.plot(audio)
        st.pyplot(fig)

        # Results
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f"""
            <div class="card">
                <h4>Sentiment</h4>
                <h2 style="color:green;">{latest['sentiment'].upper()}</h2>
                <p>{latest['conf'][0]*100:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="card">
                <h4>Emotion</h4>
                <h2>😊 {latest['emotion'].upper()}</h2>
                <p>{latest['conf'][1]*100:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="card">
                <h4>Intent</h4>
                <h2>🎯 {latest['intent'].upper()}</h2>
                <p>{latest['conf'][2]*100:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)

        # Text panels
        colA, colB = st.columns(2)

        with colA:
            st.markdown("### 📝 Transcribed")
            st.info(latest["text"])

        with colB:
            st.markdown("### 🧠 Cleaned")
            st.success(latest["cleaned"])

# ------------------------------
# RIGHT PANEL (HISTORY)
# ------------------------------
with right:

    st.markdown("### 📚 History")

    if len(st.session_state.history) > 1:
        for item in st.session_state.history[1:]:

            with st.expander(item["file"]):
                st.write("Sentiment:", item["sentiment"])
                st.write("Emotion:", item["emotion"])
                st.write("Intent:", item["intent"])
    else:
        st.info("No previous files yet")