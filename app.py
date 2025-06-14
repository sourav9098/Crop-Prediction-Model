from flask import Flask, render_template, request
import numpy as np
import pickle
import os
# Create flask app
app = Flask(__name__)
model = pickle.load(open("model_best_model.pkl", "rb"))
# Labels mapping
label_mapping = [
    'apple', 'banana', 'blackgram', 'chickpea', 'coconut', 'coffee', 'cotton',
    'grapes', 'jute', 'kidneybeans', 'lentil', 'maize', 'mango', 'mothbeans',
    'mungbean', 'muskmelon', 'orange', 'papaya', 'pigeonpeas', 'pomegranate',
    'rice', 'watermelon'
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [
            float(request.form['N']),
            float(request.form['P']),
            float(request.form['K']),
            float(request.form['temperature']),
            float(request.form['humidity']),
            float(request.form['ph']),
            float(request.form['rainfall'])
        ]
        final_features = np.array(features).reshape(1, -1)
        prediction = model.predict(final_features)[0]
        predicted_crop = label_mapping[prediction]
        return render_template('index.html', prediction_text=f"Recommended Crop: {predicted_crop}")
    except Exception as e:
        return f"Error: {e}"

if __name__ == '__main__':
    app.run(debug=True)
