import joblib
import pandas as pd

model = joblib.load("models/best_model.pkl")

sample = {}

for feature in model.features:
    if feature in model.categorical:
        sample[feature] = model.categorical_defaults[feature]
    else:
        sample[feature] = model.defaults[feature]

input_data = pd.DataFrame([sample])

prediction = model.predict(input_data)[0]
probability = model.predict_proba(input_data)[0][1]

print("\n========== MODEL TEST ==========")
print("Input Data:")
print(input_data)

print("\nPrediction:", prediction)
print("Probability:", round(probability * 100, 2), "%")

print("\nTest Passed:", prediction in [0, 1])