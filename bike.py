from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

# ---------------------------
# Load Saved Objects
# ---------------------------
model = joblib.load("model/final_bike_model.pkl")          # stacked model
top_features = joblib.load("model/top_features_avg.joblib") # top 10 features
scaler = joblib.load("model/scaler.pkl")                    # scaler

app = Flask(__name__)

# ---------------------------
# Home Route
# ---------------------------
@app.route('/')
def home():
    return {"message": "Bike Rental Prediction API is running"}

# ---------------------------
# Predict Route
# ---------------------------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Convert to DataFrame
        df = pd.DataFrame([data])

        # Select only the top features (you saved earlier)
        df = df[top_features]

        # Scale numerical input
        df_scaled = scaler.transform(df)

        # Predict using the final stacked model
        prediction = model.predict(df_scaled)[0]

        return jsonify({
            "input": data,
            "prediction_cnt": float(prediction)
        })

    except Exception as e:
        return jsonify({"error": str(e)})

# ---------------------------
# Run API locally
# ---------------------------
if __name__ == '__main__':
    app.run(debug=True)
