import joblib
import numpy as np

model = joblib.load('letter_classifier.pkl')
scaler = joblib.load('scaler.joblib')
encoder = joblib.load('label_encoder.pkl')

input_features = np.random.rand(1, 16)  # Example random features
scaled_input = scaler.transform(input_features)
predicted_number = model.predict(scaled_input)
predicted_letter = encoder.inverse_transform(predicted_number)

print(predicted_number, predicted_letter)
