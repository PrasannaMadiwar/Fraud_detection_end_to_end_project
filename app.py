from flask import Flask, render_template, request, url_for
import pickle
import numpy as np
import shap
import matplotlib.pyplot as plt
import os

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
    app.run(debug=True, host='0.0.0.0', port=5000)