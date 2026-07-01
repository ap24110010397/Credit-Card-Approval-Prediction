import pandas as pd
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def process_and_merge_data(raw_dir="data/raw", processed_dir="data/processed"):
    """
    Cleans raw application records, engineers a credit target label from credit records,
    and merges the two datasets to create a single master dataset for training.
    """
    os.makedirs(processed_dir, exist_ok=True)
    
    app_path = os.path.join(raw_dir, "application_record.csv")
    credit_path = os.path.join(raw_dir, "credit_record.csv")
    merged_path = os.path.join(processed_dir, "merged_data.csv")
    
    logger.info("Loading raw datasets...")
    app_df = pd.read_csv(app_path)
    credit_df = pd.read_csv(credit_path)
    
    # 1. Clean Application Record Duplicates
    logger.info(f"Initial application record shape: {app_df.shape}")
    app_df = app_df.drop_duplicates(subset=["ID"], keep="first")
    logger.info(f"Application record shape after dropping duplicate IDs: {app_df.shape}")
    
    # 2. Target Engineering on Credit Record
    # Definition of High Risk (1) vs Low Risk (0):
    # If the customer's payment status was ever 1, 2, 3, 4, or 5 (overdue by 30+ days), they are high risk.
    # Otherwise (statuses: C, X, 0), they are low risk.
    logger.info("Engineering target labels from credit records...")
    
    # Convert STATUS to numeric categories or flag bad months
    # Bad statuses: '1' (30-59 days), '2' (60-89 days), '3' (90-119 days), '4' (120-149 days), '5' (150+ days)
    bad_statuses = {'1', '2', '3', '4', '5'}
    
    # Flag each month's status
    credit_df['IS_BAD'] = credit_df['STATUS'].apply(lambda x: 1 if x in bad_statuses else 0)
    
    # Group by ID and check if they ever had a bad month
    # We aggregate using max() to see if 'IS_BAD' was ever 1
    grouped = credit_df.groupby('ID')['IS_BAD'].max().reset_index()
    grouped.rename(columns={'IS_BAD': 'TARGET'}, inplace=True)
    
    logger.info(f"Target labels generated. Unique IDs: {len(grouped)}")
    logger.info(f"Target class distribution:\n{grouped['TARGET'].value_counts(normalize=True)}")
    
    # 3. Merge datasets
    logger.info("Merging application records with credit target labels...")
    merged_df = pd.merge(app_df, grouped, on="ID", how="inner")
    
    logger.info(f"Merged dataset shape: {merged_df.shape}")
    
    # Save the processed dataset
    merged_df.to_csv(merged_path, index=False)
    logger.info(f"Merged dataset saved to {merged_path}")
    
    return merged_df

if __name__ == "__main__":
    process_and_merge_data()
