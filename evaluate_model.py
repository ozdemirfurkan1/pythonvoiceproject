import numpy as np
from sklearn.metrics import class_likelihood_ratios,accuracy_score
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

model = load_model("voice_auth_model.h5")
X=np.load("mfcc_features.npy")
Y=np.load("mfcc_file_paths.npy")

y=[path.split("\\")[2]for path in Y]
encoder =LabelEncoder()
y_encoded=encoder.fit_transform(y)

#test verisi bölündü
from sklearn.model_selection import train_test_split
X_train , X_test,y_train, y_test =train_test_split(X,y_encoded,test_size=0.2,random_state=42)

y_pred =model.predict(X_test)
y_pred_labels = np.argmax(y_pred,axis=1)


print("\n METRİKLER")
print("ACCURACY:",accuracy_score(y_test,y_pred_labels))
