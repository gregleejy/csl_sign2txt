import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical
import joblib
import os

DATA_PATH = 'phrase_data'
X = np.load(os.path.join(DATA_PATH, 'X.npy'), allow_pickle=True)
y = np.load(os.path.join(DATA_PATH, 'y.npy'), allow_pickle=True)

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_cat = to_categorical(y_encoded)

# Build model
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(30, 63)),
    LSTM(64),
    Dense(32, activation='relu'),
    Dense(len(le.classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train
model.fit(X, y_cat, epochs=30, batch_size=8)

# Save model and encoder
model.save("lstm_phrase_model.h5")
joblib.dump(le, "label_encoder_phrase.joblib")
print("Model and label encoder saved.")
