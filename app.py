import pickle
import numpy as np
from flask import Flask, request, render_template_string

app = Flask(__name__)

# Load the trained SVM model
with open('svm.pkl', 'rb') as f:
    model = pickle.load(f)

# HTML Template with a responsive grid layout and embedded CSS
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Health Risk Prediction - SVM Model</title>
    <style>
        :root {
            --primary: #4f46e5;
            --primary-hover: #4338ca;
            --bg-color: #f3f4f6;
            --card-bg: #ffffff;
            --text-color: #1f2937;
            --border-color: #d1d5db;
        }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-color);
            margin: 0;
            padding: 20px;
        }
        .container {
            max-width: 850px;
            margin: 40px auto;
            background: var(--card-bg);
            padding: 40px;
            border-radius: 12px;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.05);
        }
        h1 {
            text-align: center;
            color: var(--primary);
            margin-bottom: 5px;
        }
        p.subtitle {
            text-align: center;
            color: #6b7280;
            margin-bottom: 30px;
        }
        .form-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 20px;
        }
        .form-group {
            display: flex;
            flex-direction: column;
        }
        label {
            font-weight: 600;
            margin-bottom: 8px;
            font-size: 0.85rem;
            color: #374151;
            text-transform: uppercase;
            letter-spacing: 0.025em;
        }
        input, select {
            padding: 10px 12px;
            border: 1px solid var(--border-color);
            border-radius: 6px;
            font-size: 1rem;
            transition: border-color 0.2s, box-shadow 0.2s;
        }
        input:focus {
            outline: none;
            border-color: var(--primary);
            box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.1);
        }
        .btn-submit {
            grid-column: 1 / -1;
            background-color: var(--primary);
            color: white;
            padding: 14px;
            border: none;
            border-radius: 6px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: background-color 0.2s;
            margin-top: 10px;
        }
        .btn-submit:hover {
            background-color: var(--primary-hover);
        }
        .result-card {
            margin-top: 30px;
            padding: 20px;
            background-color: #e0e7ff;
            border-left: 5px solid var(--primary);
            border-radius: 6px;
            text-align: center;
            font-size: 1.25rem;
            font-weight: bold;
            color: #312e81;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Health Risk Prediction</h1>
        <p class="subtitle">Enter patient information below to process through the SVM model.</p>
        
        <form method="POST">
            <div class="form-grid">
                <div class="form-group">
                    <label for="age">Age</label>
                    <input type="number" step="any" id="age" name="age" required value="{{ request.form.get('age', '') }}">
                </div>
                <div class="form-group">
                    <label for="gender">Gender (Numeric/Encoded)</label>
                    <input type="number" step="any" id="gender" name="gender" required value="{{ request.form.get('gender', '') }}">
                </div>
                <div class="form-group">
                    <label for="city">City (Numeric/Encoded)</label>
                    <input type="number" step="any" id="city" name="city" required value="{{ request.form.get('city', '') }}">
                </div>
                <div class="form-group">
                    <label for="bmi">BMI</label>
                    <input type="number" step="any" id="bmi" name="bmi" required value="{{ request.form.get('bmi', '') }}">
                </div>
                <div class="form-group">
                    <label for="family_history_diabetes">Family History Diabetes (0/1)</label>
                    <input type="number" step="any" id="family_history_diabetes" name="family_history_diabetes" required value="{{ request.form.get('family_history_diabetes', '') }}">
                </div>
                <div class="form-group">
                    <label for="physical_activity_level">Physical Activity Level</label>
                    <input type="number" step="any" id="physical_activity_level" name="physical_activity_level" required value="{{ request.form.get('physical_activity_level', '') }}">
                </div>
                <div class="form-group">
                    <label for="diet_type">Diet Type (Numeric)</label>
                    <input type="number" step="any" id="diet_type" name="diet_type" required value="{{ request.form.get('diet_type', '') }}">
                </div>
                <div class="form-group">
                    <label for="smoking_status">Smoking Status (Numeric)</label>
                    <input type="number" step="any" id="smoking_status" name="smoking_status" required value="{{ request.form.get('smoking_status', '') }}">
                </div>
                <div class="form-group">
                    <label for="alcohol_consumption">Alcohol Consumption (Numeric)</label>
                    <input type="number" step="any" id="alcohol_consumption" name="alcohol_consumption" required value="{{ request.form.get('alcohol_consumption', '') }}">
                </div>
                <div class="form-group">
                    <label for="hours_sleep_per_night">Hours Sleep Per Night</label>
                    <input type="number" step="any" id="hours_sleep_per_night" name="hours_sleep_per_night" required value="{{ request.form.get('hours_sleep_per_night', '') }}">
                </div>
                <div class="form-group">
                    <label for="stress_level">Stress Level</label>
                    <input type="number" step="any" id="stress_level" name="stress_level" required value="{{ request.form.get('stress_level', '') }}">
                </div>
                <div class="form-group">
                    <label for="fasting_blood_sugar">Fasting Blood Sugar</label>
                    <input type="number" step="any" id="fasting_blood_sugar" name="fasting_blood_sugar" required value="{{ request.form.get('fasting_blood_sugar', '') }}">
                </div>
                <div class="form-group">
                    <label for="hba1c_level">HbA1c Level</label>
                    <input type="number" step="any" id="hba1c_level" name="hba1c_level" required value="{{ request.form.get('hba1c_level', '') }}">
                </div>
                <div class="form-group">
                    <label for="blood_pressure_systolic">BP Systolic</label>
                    <input type="number" step="any" id="blood_pressure_systolic" name="blood_pressure_systolic" required value="{{ request.form.get('blood_pressure_systolic', '') }}">
                </div>
                <div class="form-group">
                    <label for="blood_pressure_diastolic">BP Diastolic</label>
                    <input type="number" step="any" id="blood_pressure_diastolic" name="blood_pressure_diastolic" required value="{{ request.form.get('blood_pressure_diastolic', '') }}">
                </div>
                <div class="form-group">
                    <label for="waist_circumference_cm">Waist Circumference (cm)</label>
                    <input type="number" step="any" id="waist_circumference_cm" name="waist_circumference_cm" required value="{{ request.form.get('waist_circumference_cm', '') }}">
                </div>
                <div class="form-group">
                    <label for="income_bracket">Income Bracket (Numeric)</label>
                    <input type="number" step="any" id="income_bracket" name="income_bracket" required value="{{ request.form.get('income_bracket', '') }}">
                </div>
                
                <button type="submit" class="btn-submit">Predict Result</button>
            </div>
        </form>

        {% if prediction is not none %}
        <div class="result-card">
            Prediction: {{ prediction }}
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        try:
            # Collect and convert form values in the exact order expected by the model
            features = [
                float(request.form['age']),
                float(request.form['gender']),
                float(request.form['city']),
                float(request.form['bmi']),
                float(request.form['family_history_diabetes']),
                float(request.form['physical_activity_level']),
                float(request.form['diet_type']),
                float(request.form['smoking_status']),
                float(request.form['alcohol_consumption']),
                float(request.form['hours_sleep_per_night']),
                float(request.form['stress_level']),
                float(request.form['fasting_blood_sugar']),
                float(request.form['hba1c_level']),
                float(request.form['blood_pressure_systolic']),
                float(request.form['blood_pressure_diastolic']),
                float(request.form['waist_circumference_cm']),
                float(request.form['income_bracket'])
            ]
            
            final_features = np.array([features])
            pred = model.predict(final_features)[0]
            prediction = f"Positive (Class {pred})" if pred == 1 else f"Negative (Class {pred})"
        except Exception as e:
            prediction = f"Error: {str(e)}"
            
    return render_template_string(HTML_TEMPLATE, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
