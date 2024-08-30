# Machine Learning Model Workflow

## 1. Preprocess Data

- Clean, handle missing values, and scale features.

## 2. Train the Model

- Initialize and train your model with the preprocessed data.

## 3. Save the Model

- Use the `joblib` package to save the trained model.

```python
import joblib

# Save the trained model
joblib.dump(model, 'model.pkl')
```

## 4. Update the Model with New Data (if required)

- Load the saved model using `joblib`.
- Retrain the model with new data or use incremental learning if supported.
- Save the updated model using `joblib`.

```python
import joblib

# Load the existing model
model = joblib.load('model.pkl')

# Retrain or update the model with new data

# (Add your retraining code here)

# Save the updated model
joblib.dump(model, 'model_updated.pkl')
```


## 5. Create an API for the Model

- Use the `joblib` package to load the model and deploy it with Flask.
- Define an endpoint to make predictions.

**Example:**

```python
from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the model and scaler
model = joblib.load('model_updated.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = np.array(data['features']).reshape(1, -1)
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)

```
 - RUN AND SEND API REQUESTS
