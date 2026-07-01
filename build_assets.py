import os
import sys
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Ensure current directory is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils import setup_datasets
from src.data_pipeline import process_and_merge_data
from src.data_preprocessing import preprocess_pipeline
from src.train_champion import train_production_model

def build():
    """
    Executes the entire end-to-end assets build pipeline on the server:
    1. Downloads raw data.
    2. Runs target engineering and merge.
    3. Fits the preprocessor pipeline.
    4. Trains the champion model.
    """
    logger.info("==================================================")
    logger.info("RUNNING PRODUCTION ASSETS BUILD")
    logger.info("==================================================")
    
    # Step 1: Download datasets
    logger.info("Step 1/4: Downloading datasets...")
    success = setup_datasets()
    if not success:
        logger.error("Dataset download failed. Aborting build.")
        sys.exit(1)
        
    # Step 2: Clean and Merge data
    logger.info("Step 2/4: Cleaning and merging datasets...")
    process_and_merge_data()
    
    # Step 3: Fit Preprocessor Pipeline
    logger.info("Step 3/4: Fitting preprocessor pipeline...")
    preprocess_pipeline()
    
    # Step 4: Train Production Model
    logger.info("Step 4/4: Training production model...")
    train_production_model()
    
    logger.info("==================================================")
    logger.info("ASSETS BUILD COMPLETED SUCCESSFULLY")
    logger.info("==================================================")

if __name__ == "__main__":
    build()
