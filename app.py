# app.py - Flask API for Pricing Optimization
# Author: Maruf Ajimati
# BAN6800 Business Analytics Capstone Project

from flask import Flask, request, jsonify
import joblib
import pandas as pd
import traceback
import sys
print(sys.executable)

# --------------------------
# Load Model & Feature Columns
# --------------------------
MODEL_PATH = "best_pricing_model.pkl"
FEATURES_PATH = "model_feature_columns.txt"

# Load the trained model
model = joblib.load(MODEL_PATH)

# Load feature column names
with open(FEATURES_PATH, "r") as f:
    feature_columns = [line.strip() for line in f.readlines()]

# Initialize Flask app
app = Flask(__name__)

# --------------------------
# Home Route
# --------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Welcome to the Pricing Optimization API.",
        "endpoints": ["/predict"]
    })

# --------------------------
# Predict Route
# --------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON input
        data = request.get_json()

        # Validate input
        if not data:
            return jsonify({"error": "No input data provided."}), 400

        # Convert input to DataFrame
        input_df = pd.DataFrame([data], columns=feature_columns)

        # Predict quantity
        predicted_qty = model.predict(input_df)[0]

        # Compute revenue if price is given
        unit_price = data.get("unit_price", 0)
        revenue = unit_price * max(predicted_qty, 0)

        return jsonify({
            "predicted_quantity": round(float(predicted_qty), 2),
            "unit_price": unit_price,
            "expected_revenue": round(float(revenue), 2)
        })

    except Exception as e:
        return jsonify({
            "error": str(e),
            "trace": traceback.format_exc()
        }), 500

# --------------------------
# Main Entry
# --------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
