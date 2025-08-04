import numpy as np
from tensorflow.keras.models import load_model

model=load_model("voice_auth_model_big.h5")

X_test = np.load("testn_mfcc.npy")

prediction=model.predict(X_test)
predicted_class = np.argmax(prediction)

print(f"Tahmin edilen kişinin ID'si : {predicted_class}")