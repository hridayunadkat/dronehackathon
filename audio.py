import sounddevice as sd
import numpy as np
import whisper
import tempfile
import scipy.io.wavfile as wav

DURATION = 5  # seconds
SAMPLE_RATE = 16000  # Whisper expects 16kHz

def record_audio(duration, sample_rate):
    print("🎙️ Recording...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()
    return audio

def save_audio_to_file(audio, sample_rate):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
        wav.write(tmpfile.name, sample_rate, audio)
        return tmpfile.name

def load_audio_from_file(filepath):
    return wav.read(filepath)

def transcribe_audio(filepath):
    model = whisper.load_model("base")  # or tiny / small / medium / large
    result = model.transcribe(filepath)
    return result["text"]

if __name__ == "__main__":
    audio = record_audio(DURATION, SAMPLE_RATE)
    filepath = save_audio_to_file(audio, SAMPLE_RATE)
    transcript = transcribe_audio(filepath)
    print("📝 Transcription:", transcript)