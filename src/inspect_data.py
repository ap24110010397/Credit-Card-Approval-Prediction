import pandas as pd
import os

def inspect_datasets():
    raw_dir = "data/raw"
    app_path = os.path.join(raw_dir, "application_record.csv")
    credit_path = os.path.join(raw_dir, "credit_record.csv")
    
    print("=" * 60)
    print("INSPECTING DATASETS")
    print("=" * 60)
    
    # Check if files exist
    if not os.path.exists(app_path) or not os.path.exists(credit_path):
        print("Error: Dataset files not found in data/raw/")
        return
        
    # Load datasets
    print("Loading application record...")
    app_df = pd.read_csv(app_path)
    print("Loading credit record...")
    credit_df = pd.read_csv(credit_path)
    
    # 1. Dataset shapes
    print(f"\n[Shape of Datasets]")
    print(f"Application Record Shape: {app_df.shape} (Rows, Columns)")
    print(f"Credit Record Shape:      {credit_df.shape} (Rows, Columns)")
    
    # 2. Columns and Data Types
    print("\n[Application Record Columns & Types]")
    print(app_df.dtypes)
    
    print("\n[Credit Record Columns & Types]")
    print(credit_df.dtypes)
    
    # 3. Missing Values
    print("\n[Missing Values - Application Record]")
    missing_app = app_df.isnull().sum()
    print(missing_app[missing_app > 0] if missing_app.sum() > 0 else "No missing values")
    
    print("\n[Missing Values - Credit Record]")
    missing_credit = credit_df.isnull().sum()
    print(missing_credit[missing_credit > 0] if missing_credit.sum() > 0 else "No missing values")
    
    # 4. Duplicate IDs
    print("\n[Duplicate IDs]")
    print(f"Duplicates in Application ID: {app_df.duplicated(subset=['ID']).sum()} out of {len(app_df)}")
    print(f"Unique IDs in Application:    {app_df['ID'].nunique()}")
    print(f"Unique IDs in Credit:         {credit_df['ID'].nunique()}")
    
    # 5. Target Variable Candidates (Status distribution)
    print("\n[Credit Record Status Distribution]")
    print(credit_df['STATUS'].value_counts(dropna=False))
    
    # 6. Sample records
    print("\n[Sample Application Record]")
    print(app_df.head(2))
    
    print("\n[Sample Credit Record]")
    print(credit_df.head(2))

if __name__ == "__main__":
    inspect_datasets()
