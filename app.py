from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the model
model = joblib.load('model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get AQI form data
    pm25 = float(request.form['PM2.5'])
    pm10 = float(request.form['PM10'])
    no2 = float(request.form['NO2'])
    so2 = float(request.form['SO2'])
    co = float(request.form['CO'])
    o3 = float(request.form['O3'])

    features = np.array([[pm25, pm10, no2, so2, co, o3]])
    prediction = model.predict(features)[0]
    result = f"Predicted AQI: {prediction:.2f}"
    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)