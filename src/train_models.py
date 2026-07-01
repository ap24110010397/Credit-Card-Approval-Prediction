import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import logging

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def load_data(data_dir="data/processed"):
    """Loads the preprocessed splits."""
    logger.info("Loading preprocessed training and testing datasets...")
    X_train = pd.read_csv(os.path.join(data_dir, "X_train_clean.csv"))
    X_test = pd.read_csv(os.path.join(data_dir, "X_test_clean.csv"))
    y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv")).values.ravel()
    y_test = pd.read_csv(os.path.join(data_dir, "y_test.csv")).values.ravel()
    return X_train, X_test, y_train, y_test

def train_and_tune_models(X_train, X_test, y_train, y_test, models_dir="models", plots_dir="static/images/plots"):
    """
    Trains and tunes Logistic Regression, Decision Tree, Random Forest, and XGBoost.
    Evaluates them on the test set and saves the best model.
    """
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    # 3-Fold Stratified Cross Validation
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    # Define models and their hyperparameter search spaces
    models_config = {
        'Logistic Regression': {
            'model': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
            'params': {
                'C': [0.01, 0.1, 1.0, 10.0],
                'solver': ['lbfgs', 'saga']
            }
        },
        'Decision Tree': {
            'model': DecisionTreeClassifier(random_state=42, class_weight='balanced'),
            'params': {
                'max_depth': [5, 10, 15],
                'min_samples_split': [2, 5, 10]
            }
        },
        'Random Forest': {
            'model': RandomForestClassifier(random_state=42, class_weight='balanced'),
            'params': {
                'n_estimators': [100, 200],
                'max_depth': [8, 12, 15],
                'min_samples_leaf': [2, 5]
            }
        },
        'XGBoost': {
            # scale_pos_weight ~ 7.5 to handle 88%/12% class imbalance (weight of negative / positive samples)
            'model': XGBClassifier(random_state=42, eval_metric='logloss', scale_pos_weight=7.5),
            'params': {
                'n_estimators': [100, 200],
                'max_depth': [4, 6, 8],
                'learning_rate': [0.05, 0.1]
            }
        }
    }
    
    performance_metrics = {}
    best_models = {}
    
    logger.info("Starting model training and hyperparameter tuning...")
    
    for name, config in models_config.items():
        logger.info(f"Tuning hyperparameters for {name}...")
        
        # Grid Search optimizing for F1-score or ROC-AUC to balance precision & recall
        # We optimize for roc_auc here as credit scoring prefers robust probability sorting
        grid_search = GridSearchCV(
            estimator=config['model'],
            param_grid=config['params'],
            cv=cv,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        best_clf = grid_search.best_estimator_
        best_models[name] = best_clf
        
        logger.info(f"Best parameters for {name}: {grid_search.best_params_}")
        
        # Predict on test set
        y_pred = best_clf.predict(X_test)
        
        # Get probability outputs for ROC-AUC
        if hasattr(best_clf, "predict_proba"):
            y_proba = best_clf.predict_proba(X_test)[:, 1]
        else:
            y_proba = y_pred
            
        # Calculate Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_proba)
        cm = confusion_matrix(y_test, y_pred)
        
        performance_metrics[name] = {
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1,
            'ROC-AUC': roc_auc,
            'Confusion Matrix': cm
        }
        
        logger.info(f"{name} Evaluation: Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}, ROC-AUC={roc_auc:.4f}")
        
    # Create Model Comparison DataFrame
    comparison_df = pd.DataFrame(performance_metrics).T.drop(columns=['Confusion Matrix'])
    print("\n" + "="*80)
    print("MODEL COMPARISON TABLE")
    print("="*80)
    print(comparison_df.round(4))
    print("="*80)
    
    # Save comparison dataframe to a CSV for documentation/frontend
    comparison_df.to_csv(os.path.join("data/processed", "model_comparison.csv"))
    
    # Select Best Model based on F1-Score (balances Catching Defaults vs false rejections)
    best_model_name = comparison_df['F1-Score'].idxmax()
    best_model_obj = best_models[best_model_name]
    
    logger.info(f"Champion Model Selected: {best_model_name} (F1-Score: {performance_metrics[best_model_name]['F1-Score']:.4f})")
    
    # Save champion model to models/best_model.joblib
    best_model_path = os.path.join(models_dir, "best_model.joblib")
    joblib.dump(best_model_obj, best_model_path)
    logger.info(f"Saved best model ({best_model_name}) to {best_model_path}")
    
    # Save the model name string to text file for reference
    with open(os.path.join(models_dir, "best_model_name.txt"), "w") as f:
        f.write(best_model_name)
        
    # --- Generate Model Comparison Visualizations ---
    # Plot metric comparisons
    comparison_melted = comparison_df.reset_index().rename(columns={'index': 'Model'}).melt(
        id_vars='Model', var_name='Metric', value_name='Score'
    )
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=comparison_melted, x='Metric', y='Score', hue='Model', palette='Set2')
    plt.title("Performance Metric Comparison Across Models")
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "07_model_comparison.png"), dpi=150)
    plt.close()
    
    # Plot Confusion Matrix of the Best Model
    best_cm = performance_metrics[best_model_name]['Confusion Matrix']
    plt.figure(figsize=(6, 5))
    sns.heatmap(best_cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Approved (0)', 'Rejected (1)'], 
                yticklabels=['Approved (0)', 'Rejected (1)'])
    plt.title(f"Confusion Matrix - {best_model_name} (Best Model)")
    plt.ylabel("Actual Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "08_best_model_confusion_matrix.png"), dpi=150)
    plt.close()
    
    # XGBoost specific feature importance if selected
    if best_model_name in ['XGBoost', 'Random Forest', 'Decision Tree']:
        logger.info(f"Extracting feature importances from {best_model_name}...")
        feature_names = joblib.load(os.path.join(models_dir, "feature_names.joblib"))
        importances = best_model_obj.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Plot top 15 features
        plt.figure(figsize=(10, 6))
        top_n = min(15, len(feature_names))
        sns.barplot(
            x=importances[indices[:top_n]], 
            y=[feature_names[i] for i in indices[:top_n]], 
            palette='viridis',
            hue=[feature_names[i] for i in indices[:top_n]],
            legend=False
        )
        plt.title(f"Top {top_n} Feature Importances ({best_model_name})")
        plt.xlabel("Relative Importance Score")
        plt.ylabel("Feature")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "09_feature_importances.png"), dpi=150)
        plt.close()
        logger.info("Saved feature importances plot.")

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()
    train_and_tune_models(X_train, X_test, y_train, y_test)
