import os
import librosa
import numpy as np

# WAV dosyalarının bulunduğu klasör
source_dir = "C:/Users/FURKAN/pythonvoiceproject/LibriSpeech/dev-other"


target_sr = 16000

def preprocess_audio(file_path):
    try:
        signal, sr = librosa.load(file_path, sr=target_sr)
        signal = signal[:target_sr*3] 
        return signal
    except Exception as e:
        print(f"Hata: {file_path} işlenemedi. {e}")
        return None


all_signals = []

for root, _, files in os.walk(source_dir):
    for file in files:
        if file.endswith(".wav"):
            file_path = os.path.join(root, file)
            signal = preprocess_audio(file_path)
            if signal is not None:
                all_signals.append(signal)

print(f"Toplam işlenen dosya sayısı: {len(all_signals)}")
