import whisper

def transcribe_audio(wav_path):
    model = whisper.load_model("base")
    result = model.transcribe(wav_path)
    return result["text"]