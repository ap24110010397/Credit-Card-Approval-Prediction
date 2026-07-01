import os
import urllib.request
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Target URLs for the Kaggle Credit Card Approval Prediction dataset
URL_APPLICATION_RECORD = "https://raw.githubusercontent.com/shiraz-30/Credit-Card-Approval-Prediction-Model/master/application_record.csv"
URL_CREDIT_RECORD = "https://raw.githubusercontent.com/shiraz-30/Credit-Card-Approval-Prediction-Model/master/credit_record.csv"

def download_file(url, destination):
    """
    Downloads a file from a URL to a target destination.
    """
    try:
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        if os.path.exists(destination):
            logger.info(f"File already exists at {destination}, skipping download.")
            return True
        
        logger.info(f"Downloading from {url} to {destination}...")
        # Add User-Agent header to avoid HTTP 403 Forbidden error from some servers
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        with urllib.request.urlopen(req) as response, open(destination, 'wb') as out_file:
            data = response.read()
            out_file.write(data)
        logger.info(f"Downloaded successfully: {destination}")
        return True
    except Exception as e:
        logger.error(f"Failed to download from {url}: {e}")
        return False

def setup_datasets(data_dir="data/raw"):
    """
    Sets up the raw datasets by downloading them if not already present.
    """
    app_path = os.path.join(data_dir, "application_record.csv")
    credit_path = os.path.join(data_dir, "credit_record.csv")
    
    success_app = download_file(URL_APPLICATION_RECORD, app_path)
    success_credit = download_file(URL_CREDIT_RECORD, credit_path)
    
    return success_app and success_credit

if __name__ == "__main__":
    setup_datasets()
