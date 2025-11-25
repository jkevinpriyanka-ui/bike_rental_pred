from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load("model/final_bike_model.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    df = pd.DataFrame([data])
    prediction = model.predict(df)[0]
    return jsonify({"prediction": float(prediction)})

@app.route("/", methods=["GET"])
def home():
    return "Bike Rental Prediction API is running"

if __name__ == "__main__":
    app.run()
