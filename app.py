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
    # Get form data
    preg = int(request.form['preg'])
    plas = int(request.form['plas'])
    pres = int(request.form['pres'])
    skin = int(request.form['skin'])
    test = int(request.form['test'])
    mass = float(request.form['mass'])
    pedi = float(request.form['pedi'])
    age = int(request.form['age'])
    
    # Prepare input
    features = np.array([[preg, plas, pres, skin, test, mass, pedi, age]])
    
    # Predict
    prediction = model.predict(features)[0]
    result = 'Diabetic' if prediction == 1 else 'Not Diabetic'
    
    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)