# Diabetes Prediction Web App

This is a web application that predicts diabetes risk based on health metrics using a machine learning model.

## Features

- Input health metrics (pregnancies, glucose, etc.)
- Predict diabetes status using a trained Random Forest model
- Clean and responsive web interface

## Installation

1. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

2. Train the model (requires internet for data download):
   ```
   python model.py
   ```

3. Run the application:
   ```
   python app.py
   ```

4. Open your browser and go to `http://127.0.0.1:5000/`

## Usage

Enter your health details and click "Predict" to get the diabetes prediction.