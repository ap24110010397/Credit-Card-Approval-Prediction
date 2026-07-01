# Credit Card Approval Prediction System

An end-to-end, production-ready machine learning system that evaluates credit card applications and predicts default risk. It features a complete data pipeline (repayment target engineering, cleaning, scaling, and encoding), tuned tree ensemble models (Logistic Regression, Decision Trees, XGBoost, and Random Forests), a Flask backend web application, and a responsive classic corporate dashboard.

---

## 🚀 Key Features

* **Data Engineering**: Cleans duplicates and programmatically derives binary target risk labels (`TARGET=1` for delinquency overdue by 30+ days, `TARGET=0` for good repayment records).
* **Exploratory Data Analysis (EDA)**: Automatic visual plots of client demographics, income density distributions, default ratios, and correlation heatmaps.
* **Modular Preprocessing**: Bundles numerical imputation/scaling and categorical imputation/One-Hot encoding into a serialized Scikit-learn `ColumnTransformer` pipeline.
* **Model Selection**: Compares four tuned classifiers using accuracy, precision, recall, F1-score, and ROC-AUC, selecting the optimized **Random Forest** as the champion.
* **Web Dashboard**: Features a vertical slate sidebar navigation that toggles dynamic center panels (Application Form, Model Metrics, Dataset Distribution plots) and renders an animated circular credit meter for applicant risk evaluation.
* **Production Build Automation**: Features a single-command build file (`build_assets.py`) that sets up databases, pipelines, and fits model weights on the fly on the server.

---

## 📂 Project Structure

```
credit-card-approval-prediction/
│
├── .gitignore                  # Excluded folders (venv, raw CSVs, weights)
├── README.md                   # System documentation
├── requirements.txt            # Package dependencies
├── runtime.txt                 # Target cloud environment runtime
├── render.yaml                 # Render Blueprint deployment definition
├── build_assets.py             # Server build pipeline script
├── app.py                      # Flask web application entrypoint
│
├── data/
│   ├── raw/                    # Application and credit history CSV files
│   └── processed/              # Merged master dataset and comparison CSVs
│
├── models/
│   ├── best_model.joblib       # Fitted Random Forest model weights
│   ├── best_model_name.txt     # Deployed model name reference
│   ├── feature_names.joblib    # Sorted preprocessed feature labels
│   └── preprocessor.joblib     # Pre-fitted ColumnTransformer pipeline
│
├── notebooks/                  # Interactive notebooks (development drafts)
│
├── src/
│   ├── __init__.py
│   ├── utils.py                # Raw dataset download utilities
│   ├── inspect_data.py         # Schema and null value verification checks
│   ├── data_pipeline.py        # Repayment target engineering and merging
│   ├── data_preprocessing.py   # Dataset splitting and preprocessing fits
│   ├── train_models.py         # Multi-model grid search tuning & plotting
│   └── train_champion.py       # Speed-fitted production weights trainer
│
├── static/
│   ├── css/
│   │   └── style.css           # Premium classic dashboard layout stylesheet
│   ├── js/
│   │   └── app.js              # Client validation and panel toggling script
│   └── images/plots/           # pre-generated EDA and model assessment plots
│
└── templates/
    ├── index.html              # Dashboard split panel application layout
    └── result.html             # Gauge credit meter evaluation result view
```

---

## 🛠️ Local Installation & Setup

Follow these steps to run the application locally on your machine:

### 1. Clone the Workspace
```powershell
git clone <your-repository-url>
cd credit-card-approval-prediction
```

### 2. Configure the Virtual Environment
Create a local Python virtual environment to isolate the package files:
```powershell
python -m venv venv
```
Activate the environment:
* **PowerShell**: `.\venv\Scripts\Activate.ps1`
* **Git Bash / Linux**: `source venv/bin/activate`
* **Command Prompt**: `venv\Scripts\activate.bat`

### 3. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 4. Run the Production Build Pipeline
Compile all assets, fetch datasets, merge tables, generate the preprocessing pipeline, and fit the model weights:
```powershell
python build_assets.py
```

### 5. Start the Web Server
Launch the Flask development server:
```powershell
python app.py
```
Open **[http://127.0.0.1:5000](http://127.0.0.1:5000)** in your web browser to access the dashboard!

---

## 📊 Model Tuning & Evaluation Table

Grid search tuning was conducted using **3-Fold Stratified Cross-Validation** to optimize ROC-AUC. Model performance evaluated on the test split:

| Model Classifier | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 57.89% | 13.66% | 48.48% | 0.2132 | 0.5501 |
| **Decision Tree** | 67.40% | 21.22% | 65.27% | 0.3203 | 0.7181 |
| **XGBoost** | 78.77% | 29.56% | 58.16% | 0.3920 | 0.7510 |
| **Random Forest (Champion)**| **80.44%** | **30.99%** | **53.96%** | **0.3937** | **0.7559** |

*Note: While catching defaults demands high **Recall** (catching bad applicants), we optimized for **F1-Score** and **ROC-AUC** to balance catching defaults with the bank's profitability (avoiding excessive False Positive rejections of good clients).*

---

## ☁️ Cloud Deployment (Render)

This repository is optimized for free deployment on **Render** using the provided `render.yaml` blueprint.

### Automatic Blueprint Setup
1. Create a free account on **[Render.com](https://render.com/)**.
2. Connect your **GitHub account**.
3. Push this project folder to a public or private GitHub repository.
4. On the Render Dashboard, click **New** -> **Blueprint**.
5. Select your GitHub repository. Render will automatically parse the `render.yaml` file and configure the Flask web service with the correct build and start commands.
6. Click **Apply**. Once the build completes, your app will be live at the provided `.onrender.com` URL!

### Manual Setup
If you choose not to use Blueprints, configure a **Web Service** on Render with:
* **Environment**: `Python`
* **Build Command**: `pip install -r requirements.txt && python build_assets.py`
* **Start Command**: `gunicorn app:app`

---

## 📈 Key Technical Takeaways

1. **Window path limit bypass**: By deploying packages in a local project virtual environment (`venv/`), we bypass Windows Store Python AppData path limit exceptions (`MAX_PATH`).
2. **Feature Imbalance management**: Adjusting class weights (`class_weight='balanced'` in Random Forest and `scale_pos_weight=7.5` in XGBoost) prevents the models from overfitting to the majority class (88% approved).
3. **Data Leakage Prevention**: We fit the preprocessing pipeline (`preprocessor.joblib`) strictly on the training partition and load it during production inference, ensuring no test characteristics influence training.
