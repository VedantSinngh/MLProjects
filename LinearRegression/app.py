# Add these at the VERY TOP before other imports
import matplotlib
matplotlib.use('Agg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Load dataset
    df = pd.read_csv('height-weight.csv')

    # Prepare data
    X = df['Weight'].values.reshape(-1, 1)
    y = df['Height'].values

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    # Standardize data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train model
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    # Predict
    weight = float(request.form['weight'])
    weight_scaled = scaler.transform(np.array([[weight]]))
    prediction = regressor.predict(weight_scaled)

    # Plot
    plt.scatter(X_train, y_train)
    plt.plot(X_train, regressor.predict(X_train), color='red')
    plt.xlabel('Weight')
    plt.ylabel('Height')
    plt.title('Linear Regression')

    # Save plot to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_url = base64.b64encode(buf.getvalue()).decode('utf8')
    plt.close()

    return render_template('index.html', prediction=prediction[0], plot_url=plot_url)

if __name__ == '__main__':
    app.run(debug=True)