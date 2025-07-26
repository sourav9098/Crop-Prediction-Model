from flask import Flask, render_template, request
import numpy as np
import pickle

# Initialize Flask app
app = Flask(__name__)

# Load the trained model, scaler, and label encoder
model = pickle.load(open("best_crop_model.pkl", "rb"))
scaler = pickle.load(open("crop_scaler.pkl", "rb"))
le = pickle.load(open("label_encoder.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input features from the form
        features = [
            float(request.form['N']),
            float(request.form['P']),
            float(request.form['K']),
            float(request.form['temperature']),
            float(request.form['humidity']),
            float(request.form['ph']),
            float(request.form['rainfall'])
        ]

        # Scale the features using the same scaler used during training
        final_input = np.array(features).reshape(1, -1)
        scaled_input = scaler.transform(final_input)

        # Predict and decode the crop label
        prediction = model.predict(scaled_input)[0]
        predicted_crop = le.inverse_transform([prediction])[0]

        return render_template('index.html', prediction_text=f"🌾 Recommended Crop: {predicted_crop.capitalize()}")

    except Exception as e:
        return render_template('index.html', prediction_text=f"❌ Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
