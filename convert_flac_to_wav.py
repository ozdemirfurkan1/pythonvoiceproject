from pydub import AudioSegment
import os

source_dir = "C:/Users/FURKAN/pythonvoiceproject/LibriSpeech/dev-other"

print("Dönüştürme başlatildi ...")


for root, dirs, files in os.walk(source_dir):
    for file in files:
        if file.endswith(".flac"):
            flac_path = os.path.join(root, file)
            wav_path = os.path.join(root, file.replace(".flac", ".wav"))
            
            try:
                sound = AudioSegment.from_file(flac_path, format="flac")
                sound.export(wav_path, format="wav")
                print(f"[✔] {file} dönüştürüldü.")
            except Exception as e:
                print(f"[✖] {file} dönüştürülemedi: {e}")

print("Tüm dönüşümler tamamlandi.")
