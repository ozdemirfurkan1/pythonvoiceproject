import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# 1. MFCC Özelliklerini Yükle
X = np.load("mfcc_features_bign.npy")
y = np.load("mfcc_file_paths_bign.npy")


y = [path.split("\\")[2] for path in y]  


encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)  
np.save("person_ids1.npy", encoder.classes_)  


y_categorical = to_categorical(y_encoded)

# 5. MFCC Normalizasyonu
num_samples, time_steps, features = X.shape
X = X.reshape(num_samples, -1)
scaler = StandardScaler()
X = scaler.fit_transform(X)
X = X.reshape(num_samples, time_steps, features)


X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)


X_train = X_train[..., np.newaxis]  
X_test = X_test[..., np.newaxis]

num_classes = y_categorical.shape[1]

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(94, 40, 1)),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

# 9. Derleme
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 10. Eğitme
model.fit(X_train, y_train, epochs=70, batch_size=64, validation_data=(X_test, y_test))

# 11. Kaydet
model.save("voice_auth_model_cnn_bign.h5")
print("✅ Model başarıyla eğitildi ve voice_auth_model_cnn_bign.h5 olarak kaydedildi.")
