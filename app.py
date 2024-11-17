from flask import Flask, render_template, request
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the trained model
model = pickle.load(open('model2.pkl', 'rb'))

# Initialize the Flask app
app = Flask(__name__)

# Route to render the homepage/input form
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle form submission and display prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get form data
        watch_time = float(request.form['watch_time'])
        stream_time = float(request.form['stream_time'])
        peak_viewers = float(request.form['peak_viewers'])
        average_viewers = float(request.form['average_viewers'])
        language = int(request.form['language'])  # Assuming language is encoded as 1 for English, etc.

        # Preprocess data (scaling the numerical features)
        features = np.array([[watch_time, stream_time, peak_viewers, average_viewers, language]])
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)  # Scale features (if needed)

        # Predict using the trained model
        prediction = model.predict(features_scaled)

        # Return the result page with the prediction
        result = "High Growth" if prediction[0] == 1 else "Low Growth"
        return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
