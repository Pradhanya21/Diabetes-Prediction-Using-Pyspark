from flask import Flask, request, render_template
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

# Load the trained diabetes prediction model from model.pkl
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)

# Initialize the label encoder for the gender field
label_encoder = LabelEncoder()

# Function to categorize BMI
def categorize_bmi(bmi):
    if bmi < 18.5:
        return "Underweight"
    elif 18.5 <= bmi < 24.9:
        return "Normal weight"
    elif 25 <= bmi < 29.9:
        return "Overweight"
    else:
        return "Obese"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Updated mapping for smoking history
    smoking_history_map = {
        'never': 0,        # Never smoked
        'No Info': 1,      # No information
        'current': 2,      # Currently smoking
        'ever': 3          # Ever smoked
    }

    # Get input values from the form
    input_data = {
        'gender': int(request.form['gender']),  # Male = 1, Female = 0
        'age': float(request.form['age']),
        'hypertension': float(request.form['hypertension']),
        'heart_disease': float(request.form['heart_disease']),
        'smoking_history': smoking_history_map[request.form['smoking_history']],  # Map the string to numeric value
        'bmi': float(request.form['bmi']),
        'HbA1c_level': float(request.form['HbA1c_level']),
        'blood_glucose_level': float(request.form['blood_glucose_level'])
    }

    # Categorize BMI
    bmi_category = categorize_bmi(input_data['bmi'])

    # Convert input data to DataFrame with only the selected features
    input_df = pd.DataFrame([input_data])

    # Predict Diabetes Status
    prediction = model.predict(input_df)
    prediction_text = "Diabetes Detected" if prediction[0] == 1 else "No Diabetes Detected"
    
    # Display result on the page
    return render_template('index.html', prediction_text=f'Diabetes Prediction: {prediction_text}', bmi_category=f'BMI Category: {bmi_category}')

if __name__ == "__main__":
    app.run(debug=False,host='0.0.0.0')
