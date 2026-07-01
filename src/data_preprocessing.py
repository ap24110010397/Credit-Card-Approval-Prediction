import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def preprocess_pipeline(data_path="data/processed/merged_data.csv", output_dir="data/processed", models_dir="models"):
    """
    Cleans raw columns, performs a stratified train-test split, creates a 
    Scikit-learn ColumnTransformer pipeline, fits it on training data,
    saves the preprocessor, and saves the cleaned splits.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    # 1. Load merged dataset
    if not os.path.exists(data_path):
        logger.error(f"Merged dataset not found at {data_path}")
        return
        
    df = pd.read_csv(data_path)
    logger.info(f"Loaded merged dataset. Shape: {df.shape}")
    
    # 2. Raw Feature Engineering (Conversions)
    logger.info("Applying age and employment duration conversions...")
    # Age: Convert DAYS_BIRTH (negative) to positive years
    df['AGE'] = -df['DAYS_BIRTH'] / 365.25
    
    # Years Employed: Convert DAYS_EMPLOYED (negative) to positive years, set 365243 (unemployed) to 0
    df['YEARS_EMPLOYED'] = df['DAYS_EMPLOYED'].apply(lambda x: 0.0 if x > 0 else -x / 365.25)
    # Create Unemployment indicator flag
    df['IS_UNEMPLOYED'] = df['DAYS_EMPLOYED'].apply(lambda x: 1 if x > 0 else 0)
    
    # Drop raw columns that are replaced or not useful for modeling
    # ID is dropped because it's a unique identifier and has no predictive power (prevents leakage/overfitting)
    # FLAG_MOBIL is dropped because 100% of applicants have a mobile phone (no variance)
    # DAYS_BIRTH and DAYS_EMPLOYED are replaced by AGE and YEARS_EMPLOYED
    cols_to_drop = ['ID', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'FLAG_MOBIL']
    df = df.drop(columns=cols_to_drop, errors='ignore')
    
    # 3. Define target and features
    X = df.drop(columns=['TARGET'])
    y = df['TARGET']
    
    logger.info(f"Feature set shape: {X.shape}, Target shape: {y.shape}")
    
    # 4. Stratified Train-Test Split (80% Train, 20% Test)
    # Stratified split ensures the proportion of default cases is identical in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    logger.info(f"Train set shape: {X_train.shape}, Test set shape: {X_test.shape}")
    logger.info(f"Train target mean: {y_train.mean():.4f}, Test target mean: {y_test.mean():.4f}")
    
    # 5. Define Feature Types for Pipeline
    numerical_features = ['AMT_INCOME_TOTAL', 'AGE', 'YEARS_EMPLOYED', 'CNT_CHILDREN', 'CNT_FAM_MEMBERS']
    
    # Binary categories (M/F, Y/N) and nominal categories
    categorical_features = [
        'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 
        'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 
        'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE'
    ]
    
    # Indicator columns (phone flags) that are already 1/0
    binary_indicators = ['FLAG_WORK_PHONE', 'FLAG_PHONE', 'FLAG_EMAIL', 'IS_UNEMPLOYED']
    
    # 6. Build the Pipelines
    # Numerical Pipeline: Impute missing numericals with median, then standard scale
    num_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical Pipeline: Impute missing categoricals (e.g. occupation type) with 'Unknown', then One-Hot Encode
    cat_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine everything using ColumnTransformer
    # Columns not specified (e.g., binary indicators that are already 0/1) will pass through as-is
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, numerical_features),
            ('cat', cat_pipeline, categorical_features)
        ],
        remainder='passthrough'  # Keep indicators as they are
    )
    
    # 7. Fit and Transform
    logger.info("Fitting preprocessing pipeline on training data...")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Retrieve feature names for tracking
    # Get numeric feature names
    cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
    # Re-retrieve fitted category names for onehot encoded columns
    cat_feature_names = cat_encoder.get_feature_names_out(categorical_features).tolist()
    all_feature_names = numerical_features + cat_feature_names + binary_indicators
    
    logger.info(f"Processed feature names count: {len(all_feature_names)}")
    logger.info(f"Transformed train shape: {X_train_processed.shape}")
    logger.info(f"Transformed test shape:  {X_test_processed.shape}")
    
    # Convert processed arrays back to DataFrames for easier viewing and loading
    X_train_processed_df = pd.DataFrame(X_train_processed, columns=all_feature_names)
    X_test_processed_df = pd.DataFrame(X_test_processed, columns=all_feature_names)
    
    # 8. Save Outputs
    # Save the fitted preprocessor pipeline to models/
    preprocessor_file = os.path.join(models_dir, "preprocessor.joblib")
    joblib.dump(preprocessor, preprocessor_file)
    logger.info(f"Fitted preprocessor pipeline saved to {preprocessor_file}")
    
    # Save target variable and clean data splits to data/processed/
    X_train_processed_df.to_csv(os.path.join(output_dir, "X_train_clean.csv"), index=False)
    X_test_processed_df.to_csv(os.path.join(output_dir, "X_test_clean.csv"), index=False)
    y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)
    
    # Save feature names list to models/ for model tracking/inference
    feature_names_file = os.path.join(models_dir, "feature_names.joblib")
    joblib.dump(all_feature_names, feature_names_file)
    
    logger.info("Preprocessing complete! Cleaned splits and preprocessor saved.")

if __name__ == "__main__":
    preprocess_pipeline()
