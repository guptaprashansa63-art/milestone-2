


import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
import joblib

# Generate synthetic AQI dataset using sklearn
X, y = make_regression(n_samples=1000, n_features=6, noise=0.1, random_state=42)

# Create DataFrame with AQI feature names
features = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']
df = pd.DataFrame(X, columns=features)
df['AQI'] = y

# Select features and target
X = df[features]
y = df['AQI']

# Train AQI regression model
model = RandomForestRegressor(random_state=42)
model.fit(X, y)

# Save model
joblib.dump(model, 'model.pkl')