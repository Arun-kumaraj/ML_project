from flask import Flask, request, render_template
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Get absolute path to salary_model.pkl (same folder as app.py)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "salary_model.pkl")

# Load model
model = joblib.load(model_path)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    exp_str = request.form.get('experience')
    age_str = request.form.get('age')
    gender = request.form.get('gender')
    education = request.form.get('education')

    # Validate presence
    if not all([exp_str, age_str, gender, education]):
        return render_template(
            'index.html',
            error_text='Please fill in all fields before predicting.',
        )

    # Validate types
    try:
        years_exp = float(exp_str)
        age = int(age_str)
    except ValueError:
        return render_template(
            'index.html',
            error_text='Please enter valid numeric values for experience and age.',
        )

    # Build input according to model's expected features
    expected = None
    try:
        if hasattr(model, 'feature_names_in_') and model.feature_names_in_ is not None:
            expected = list(model.feature_names_in_)
        elif hasattr(model, 'named_steps') and 'preprocessor' in model.named_steps:
            pre = model.named_steps['preprocessor']
            if hasattr(pre, 'feature_names_in_'):
                expected = list(pre.feature_names_in_)
    except Exception:
        expected = None

    input_df = None
    if expected:
        data_map = {
            'Years of Experience': years_exp,
            'Age': age,
            'Gender': gender,
            'Education Level': education,
        }
        row = [data_map[col] for col in expected if col in data_map]
        input_df = pd.DataFrame([row], columns=[col for col in expected if col in data_map])
    else:
        try:
            input_df = pd.DataFrame(
                [[years_exp, age, gender, education]],
                columns=['Years of Experience', 'Age', 'Gender', 'Education Level']
            )
        except Exception:
            input_df = pd.DataFrame([[years_exp]], columns=['Years of Experience'])

    try:
        prediction = model.predict(input_df)[0]
    except Exception:
        input_df = pd.DataFrame([[years_exp]], columns=['Years of Experience'])
        prediction = model.predict(input_df)[0]

    return render_template('index.html', prediction_text=f"Predicted Salary: ₹{prediction:.2f}")

if __name__ == "__main__":
    app.run(debug=True)
