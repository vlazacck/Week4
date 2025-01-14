from flask import Flask, request, render_template, jsonify
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# Initialize Flask app
app = Flask(__name__)

# Load pre-trained LSTM model
model_path = "model/lstm_sales_model.keras"
model = tf.keras.models.load_model(model_path)

# Set up a scaler for feature scaling (match it to your training setup)
scaler = MinMaxScaler(feature_range=(-1, 1))

# Sample dummy fit for scaler; replace with appropriate scaling setup for real data
scaler.fit(np.array([[0], [1000]]))  # Adjust based on actual sales range

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Extract features from form input
        features = [
            float(request.form['customers']),
            float(request.form['store_type']),
            float(request.form['competition_distance']),
            float(request.form['promo'])
        ]
        
        # Process input for prediction
        features_scaled = scaler.transform(np.array(features).reshape(-1, 1)).reshape(1, -1, 1)
        
        # Get prediction
        prediction_scaled = model.predict(features_scaled)
        prediction = scaler.inverse_transform(prediction_scaled).flatten()[0]
        
        return render_template("index.html", prediction_text=f"Predicted Sales: {prediction:.2f}")
    
    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
