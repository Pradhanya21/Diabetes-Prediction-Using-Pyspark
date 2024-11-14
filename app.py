from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model from the pickle file
model_path = r'C:\Users\pradh\OneDrive\Desktop\random-forest-flask\models\random_forest_model.pkl'

with open(model_path, 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from the POST request
    data = request.get_json(force=True)
    input_data = np.array([list(data.values())]).reshape(1, -1)

    # Make a prediction
    prediction = model.predict(input_data)

    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
