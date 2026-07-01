import pandas as pd
import os
import joblib
import logging
from sklearn.ensemble import RandomForestClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def train_production_model(data_dir="data/processed", models_dir="models"):
    """
    Trains only the champion model (Random Forest) with the optimized hyperparameters
    found during local grid search. This is fast and memory-efficient for build servers.
    """
    logger.info("Starting production model training...")
    
    # 1. Load clean datasets
    X_train = pd.read_csv(os.path.join(data_dir, "X_train_clean.csv"))
    y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv")).values.ravel()
    
    # 2. Initialize Random Forest with optimized parameters
    # max_depth=15, min_samples_leaf=2, n_estimators=200, class_weight='balanced'
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    # 3. Fit model on the full training set
    logger.info("Fitting champion Random Forest on training data...")
    model.fit(X_train, y_train)
    
    # 4. Save model weights
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "best_model.joblib")
    joblib.dump(model, model_path)
    logger.info(f"Production model successfully saved to {model_path}")
    
    # Save best model name text file
    with open(os.path.join(models_dir, "best_model_name.txt"), "w") as f:
        f.write("Random Forest")
        
    logger.info("Production model build completed.")

if __name__ == "__main__":
    train_production_model()
