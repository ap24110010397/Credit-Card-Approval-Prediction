from flask import Flask, render_template, request, jsonify, redirect
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__)

# Load model, preprocessor, and feature order at startup
models_dir = "models"
preprocessor_path = os.path.join(models_dir, "preprocessor.joblib")
model_path = os.path.join(models_dir, "best_model.joblib")

# Check if model files exist
if os.path.exists(preprocessor_path) and os.path.exists(model_path):
    preprocessor = joblib.load(preprocessor_path)
    model = joblib.load(model_path)
    
    # Try to load best model name for UI display
    model_name_path = os.path.join(models_dir, "best_model_name.txt")
    if os.path.exists(model_name_path):
        with open(model_name_path, "r") as f:
            MODEL_NAME = f.read().strip()
    else:
        MODEL_NAME = "Random Forest"
else:
    preprocessor = None
    model = None
    MODEL_NAME = "Not Loaded"
    print("Warning: Model or Preprocessor not found! Run preprocessing and training first.")

# Column order expected by the preprocessor pipeline
EXPECTED_COLUMNS = [
    'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'CNT_CHILDREN', 
    'AMT_INCOME_TOTAL', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 
    'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'FLAG_WORK_PHONE', 
    'FLAG_PHONE', 'FLAG_EMAIL', 'OCCUPATION_TYPE', 'CNT_FAM_MEMBERS', 
    'AGE', 'YEARS_EMPLOYED', 'IS_UNEMPLOYED'
]

@app.route('/')
def home():
    """Renders the dashboard with data insights and the application form."""
    # Check if we have data analysis comparison
    comparison_path = "data/processed/model_comparison.csv"
    model_metrics = []
    if os.path.exists(comparison_path):
        comp_df = pd.read_csv(comparison_path)
        # Convert to dictionary list for rendering in Jinja
        model_metrics = comp_df.rename(columns={'Unnamed: 0': 'Model'}).to_dict(orient='records')
        
    return render_template('index.html', model_name=MODEL_NAME, metrics=model_metrics)

@app.route('/predict', methods=['POST'])
def predict():
    """Handles applicant form submission, runs inference, and displays results."""
    if preprocessor is None or model is None:
        return render_template('result.html', error="Model not loaded on server. Please train models first.")
        
    try:
        # Extract inputs from form
        gender = request.form.get('gender', 'F')
        own_car = request.form.get('own_car', 'N')
        own_realty = request.form.get('own_realty', 'N')
        children = int(request.form.get('children', 0))
        income = float(request.form.get('income', 0.0))
        income_type = request.form.get('income_type', 'Working')
        education = request.form.get('education', 'Secondary / secondary special')
        family_status = request.form.get('family_status', 'Single / not married')
        housing = request.form.get('housing', 'House / apartment')
        
        age = float(request.form.get('age', 30.0))
        
        employment_status = request.form.get('employment_status', 'employed')
        if employment_status == 'unemployed':
            years_employed = 0.0
            is_unemployed = 1
        else:
            years_employed = float(request.form.get('years_employed', 0.0))
            is_unemployed = 0
            
        work_phone = int(request.form.get('work_phone', 0))
        phone = int(request.form.get('phone', 0))
        email = int(request.form.get('email', 0))
        
        occupation = request.form.get('occupation', 'Unknown')
        family_members = float(request.form.get('family_members', 1.0))
        
        # Construct DataFrame in the exact expected structure
        input_data = pd.DataFrame([{
            'CODE_GENDER': gender,
            'FLAG_OWN_CAR': own_car,
            'FLAG_OWN_REALTY': own_realty,
            'CNT_CHILDREN': children,
            'AMT_INCOME_TOTAL': income,
            'NAME_INCOME_TYPE': income_type,
            'NAME_EDUCATION_TYPE': education,
            'NAME_FAMILY_STATUS': family_status,
            'NAME_HOUSING_TYPE': housing,
            'FLAG_WORK_PHONE': work_phone,
            'FLAG_PHONE': phone,
            'FLAG_EMAIL': email,
            'OCCUPATION_TYPE': occupation,
            'CNT_FAM_MEMBERS': family_members,
            'AGE': age,
            'YEARS_EMPLOYED': years_employed,
            'IS_UNEMPLOYED': is_unemployed
        }])
        
        # Reorder columns to match X_train exactly
        input_data = input_data[EXPECTED_COLUMNS]
        
        # Apply preprocessing pipeline (Scaling & One-Hot Encoding)
        processed_input = preprocessor.transform(input_data)
        
        # Run prediction
        # Output is binary (0 = Approved, 1 = Rejected)
        prediction = int(model.predict(processed_input)[0])
        
        # Get probability (score) of default
        prob_default = model.predict_proba(processed_input)[0][1]
        
        # Calculate approval score (out of 100)
        # If prob_default is 0.1, approval score is 90%
        approval_score = round((1 - prob_default) * 100, 1)
        
        # Decide final status
        # Since we optimize for catching defaults, we might have a strict default probability threshold.
        # Standard threshold is 0.5, but banks are conservative (e.g. rejecting if default probability > 0.25)
        # We will use 0.5 as standard, but display the risk scale.
        status = "Approved" if prediction == 0 else "Rejected"
        risk_level = "Low Risk"
        if prob_default > 0.6:
            risk_level = "High Risk"
        elif prob_default > 0.25:
            risk_level = "Medium Risk"
            
        applicant_summary = {
            'income': f"${income:,.2f}",
            'age': int(age),
            'years_employed': f"{years_employed:.1f}" if is_unemployed == 0 else "Unemployed",
            'education': education.split('/')[0].strip(), # clean secondary / secondary special to secondary
            'status': status,
            'approval_score': approval_score,
            'prob_default': round(prob_default * 100, 1),
            'risk_level': risk_level
        }
        
        return render_template('result.html', applicant=applicant_summary)
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        return render_template('result.html', error=f"An error occurred during prediction: {str(e)}")

# Health check route
@app.route('/health')
def health():
    return jsonify({"status": "healthy", "model": MODEL_NAME})

if __name__ == '__main__':
    # Running locally on port 5000
    app.run(debug=True, host='0.0.0.0', port=5000)
