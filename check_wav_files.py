import os

source_dir = "C:/Users/FURKAN/pythonvoiceproject/LibriSpeech/dev-other"
wav_count = 0

for root, _, files in os.walk(source_dir):
    for file in files:
        if file.endswith(".wav"):
            wav_count += 1
            print("Bulunan dosya:", os.path.join(root, file))

print(f"Toplam WAV dosyası: {wav_count}")
