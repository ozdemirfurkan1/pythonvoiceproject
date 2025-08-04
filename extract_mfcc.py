import os
import librosa
import numpy as np

source_dir = "C:/Users/FURKAN/pythonvoiceproject/LibriSpeech/dev-other"
mfcc_featuresnew = []
file_pathsnew = []

sample_rate = 16000
duration = 3 #saniye
n_mfcc = 40 #mfcc değeri

for root, _, files in os.walk(source_dir):
    for file in files:
        if file.endswith(".wav"):
            path = os.path.join(root, file)
            try:
                print(f"İşleniyor: {path}")
                signal, sr = librosa.load(path, sr=sample_rate)

                if len(signal) < sample_rate * duration:
                    print(f"UYARI: {file} 3 saniyeden kisa olduğu için atlandi.")
                    continue

                signal = signal[:sample_rate * duration]
                mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)
                mfcc = mfcc.T  

                mfcc_featuresnew.append(mfcc)
                file_pathsnew.append(path)

            except Exception as e:
                print(f"HATA: {path} işlenemedi. Sebep: {e}")

# dosya yolları
np.save("mfcc_features_bign.npy", mfcc_featuresnew)
np.save("mfcc_file_paths_bign.npy", file_pathsnew)


print(f"Toplam çikarilan MFCC sayisi: {len(mfcc_featuresnew)}")
