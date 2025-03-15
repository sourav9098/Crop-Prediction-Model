import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import sklearn

# Create flask app
app = Flask(__name__)
model = pickle.load(open("model9.pkl", "rb"))

@app.route("/")
def Home():
    return render_template("index.html")

@app.route("/predict", methods = ["POST"])
def predict():
    N=int(request.form.get('N'))
    P=int(request.form.get('P'))
    K=int(request.form.get('K'))
    temperature=float(request.form.get('temperature'))
    humidity=float(request.form.get('humidity'))
    pH=float(request.form.get('pH'))
    rainfall=float(request.form.get('rainfall')) 
    input_features = np.array([[N, P, K, temperature, humidity, pH, rainfall]])
    prediction = model.predict(input_features)[0]
    
    return render_template("index.html", prediction_text=f"The Predicted Crop is: {prediction}")

if __name__ == "__main__":
    app.run(debug=True)