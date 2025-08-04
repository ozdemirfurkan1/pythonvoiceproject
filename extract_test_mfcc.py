import librosa
import numpy as np


file_path = "C:/Users/FURKAN/pythonvoiceproject/LibriSpeech1/dev-clean/174/50561/174-50561-0001.wav"
signal, sr = librosa.load(file_path, sr=16000)

# MFCC çıkarımı
mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=40)

mfcc = mfcc.T


if mfcc.shape[0] > 94:
    mfcc = mfcc[:94]
elif mfcc.shape[0] < 94:
    
    pad_width = 94 - mfcc.shape[0]
    mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode='constant')


mfcc = mfcc[np.newaxis, ...]


np.save("testnew_mfcc.npy", mfcc)
