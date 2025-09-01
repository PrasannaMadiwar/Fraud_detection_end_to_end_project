from flask import Flask, render_template, request, url_for
import pickle
import numpy as np
import shap
import matplotlib.pyplot as plt
import os
from datetime import datetime
app = Flask(__name__)

# Ensure static folder exists
os.makedirs("static", exist_ok=True)

# Load model and scaler with error handling
try:
    model = pickle.load(open("model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    print("Model and scaler loaded successfully!")
except FileNotFoundError as e:
    print(f"Error loading files: {e}")
    print("Make sure xgb_model.pkl and scaler.pkl are in the same directory as this script")

# Feature names
features = [f"V{i}" for i in range(1, 29)] + ["Amount"]

@app.route('/')
def index():
    return render_template("index.html")


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    API endpoint for fraud detection
    Expected JSON format:
    {
        "V1": -1.359807, "V2": -0.072781, ..., "V28": -0.021053, "Amount": 149.62
    }
    Returns JSON response with prediction results
    """
    try:
        # Check if request has JSON data
        if not request.is_json:
            return {"error": "Request must be JSON"}, 400
        
        data = request.get_json()
        
        # Validate input data
        input_data = []
        missing_fields = []
        
        for f in features:
            if f not in data:
                missing_fields.append(f)
            else:
                try:
                    input_data.append(float(data[f]))
                except (ValueError, TypeError):
                    return {"error": f"Invalid numeric value for field {f}"}, 400
        
        if missing_fields:
            return {"error": f"Missing fields: {missing_fields}"}, 400
        
        # Transform and predict
        X_scaled = scaler.transform([input_data])
        pred = model.predict(X_scaled)[0]
        prob = model.predict_proba(X_scaled)[0][1] * 100
        
        # SHAP explanation
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_scaled)
        
        # Prepare feature importance
        feature_importance = []
        for i, feature_name in enumerate(features):
            feature_importance.append({
                "feature": feature_name,
                "value": float(data[feature_name]),
                "shap_value": float(shap_values[0][i]),
                "contribution": "positive" if shap_values[0][i] > 0 else "negative"
            })
        
        # Sort by absolute SHAP value (most important first)
        feature_importance.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
        
        # Response
        response = {
            "status": "success",
            "prediction": {
                "result": "fraudulent" if pred == 1 else "legitimate",
                "fraud_probability": round(prob, 2),
                "risk_level": "high" if prob > 70 else "medium" if prob > 30 else "low"
            },
            "model_info": {
                "model_type": "XGBoost",
                "features_count": len(features),
                "explainability": "SHAP"
            },
            "feature_analysis": {
                "top_5_contributors": feature_importance[:5],
                "all_features": feature_importance
            },
            "timestamp": str(datetime.now())
        }
        
        return response, 200
        
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}, 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Read input values with better error handling
        input_data = []
        missing_fields = []
        
        for f in features:
            value = request.form.get(f)
            if value is None or value == '':
                missing_fields.append(f)
            else:
                try:
                    input_data.append(float(value))
                except ValueError:
                    return f"Error: Invalid numeric value for field {f}"
        
        if missing_fields:
            return f"Error: Missing values for fields: {', '.join(missing_fields)}"
        
        # Transform input data
        X_scaled = scaler.transform([input_data])

        # Make prediction
        pred = model.predict(X_scaled)[0]
        prob = model.predict_proba(X_scaled)[0][1] * 100

        # SHAP explanation
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_scaled)

        # Create SHAP plot with better formatting
        shap_img_path = os.path.join("static", "shap_plot.png")
        
        plt.figure(figsize=(14, 4))
        shap.force_plot(
            explainer.expected_value, 
            shap_values[0], 
            X_scaled[0],
            feature_names=features, 
            matplotlib=True, 
            show=False
        )
        plt.tight_layout()
        plt.savefig(shap_img_path, bbox_inches="tight", dpi=150, facecolor='white')
        plt.close()

        # Prepare feature importance values for chart (all features)
        chart_feature_names = features
        chart_feature_values = [float(val) for val in np.abs(shap_values[0])]

        return render_template("dashboard.html",
                               prediction="Fraudulent" if pred == 1 else "Legitimate",
                               probability=round(prob, 2),
                               shap_img="shap_plot.png",
                               feature_names=chart_feature_names,
                               feature_values=chart_feature_values)
                               
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return f"Error processing prediction: {str(e)}"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Heroku/Dokku gives a PORT
    app.run(host="0.0.0.0", port=port)
