import os
import numpy as np
import librosa
from tensorflow.keras.models import load_model

# === AYARLAR ===
model_path = "voice_auth_model_cnn_bign.h5"
test_file = "C:/Users/FURKAN/pythonvoiceproject/LibriSpeech1/dev-clean/84/121123/84-121123-0000.wav"
n_mfcc = 40
frame_count = 94


def extract_mfcc(file_path, n_mfcc=40, frame_count=94):
    signal, sr = librosa.load(file_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc).T

    if mfcc.shape[0] > frame_count:
        mfcc = mfcc[:frame_count]
    elif mfcc.shape[0] < frame_count:
        pad_width = frame_count - mfcc.shape[0]
        mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode='constant')

    return mfcc[np.newaxis, ..., np.newaxis]  

model = load_model(model_path)

mfcc = extract_mfcc(test_file)
prediction = model.predict(mfcc)
predicted_index = np.argmax(prediction)

person_ids = np.load("person_ids1.npy") 


predicted_id = person_ids[predicted_index]


filename = os.path.basename(test_file)  
real_id = filename.split("-")[0]        


print(f"Tahmin edilen kişi ID'si: {predicted_id}")
print(f"Gerçek kişi ID'si: {real_id}")

if predicted_id == real_id:
    print("✅ Doğru kişi tahmin edildi!")
else:
    print(f"❌ Yanlış tahmin. Bu ses {real_id} kişisine aitti.")
