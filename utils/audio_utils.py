from pydub import AudioSegment

# 🔥 FORCE PATH (IMPORTANT)
AudioSegment.converter = "C:\\ffmpeg\\bin\\ffmpeg.exe"
AudioSegment.ffprobe = "C:\\ffmpeg\\bin\\ffprobe.exe"


def convert_to_wav(input_path, output_path="converted.wav"):
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_frame_rate(16000).set_channels(1)
    audio.export(output_path, format="wav")
    return output_path