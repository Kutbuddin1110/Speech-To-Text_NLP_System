import os
from pydub import AudioSegment

def convert_to_wav(input_path, output_path="converted.wav"):
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_frame_rate(16000).set_channels(1)
    audio.export(output_path, format="wav")
    return output_path