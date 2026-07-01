import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def run_eda(data_path="data/processed/merged_data.csv", output_dir="static/images/plots"):
    """
    Performs exploratory data analysis on the merged dataset and generates plots.
    """
    print("=" * 60)
    print("RUNNING EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 60)
    
    # 1. Load merged dataset
    if not os.path.exists(data_path):
        print(f"Error: Merged dataset not found at {data_path}")
        return
        
    df = pd.read_csv(data_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # Set styling
    sns.set_theme(style="whitegrid")
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["font.size"] = 12
    
    # --- Data Cleaning transformations needed for plotting ---
    # Convert DAYS_BIRTH to AGE in years (DAYS_BIRTH is negative count of days from current date)
    df['AGE'] = -df['DAYS_BIRTH'] / 365.25
    
    # Convert DAYS_EMPLOYED to YEARS_EMPLOYED. Note that positive value 365243 represents unemployed.
    df['YEARS_EMPLOYED'] = df['DAYS_EMPLOYED'].apply(lambda x: 0 if x > 0 else -x / 365.25)
    df['IS_UNEMPLOYED'] = df['DAYS_EMPLOYED'].apply(lambda x: 1 if x > 0 else 0)
    
    # 2. Descriptive statistics
    print("\n[Descriptive Statistics - Numerical Columns]")
    numerical_cols = ['AMT_INCOME_TOTAL', 'AGE', 'YEARS_EMPLOYED', 'CNT_CHILDREN', 'CNT_FAM_MEMBERS']
    print(df[numerical_cols].describe().round(2))
    
    print("\n[Categorical Value Counts (Top Categories)]")
    for col in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE']:
        print(f"\nValue counts for {col}:")
        print(df[col].value_counts(normalize=True).round(3) * 100)
        
    # 3. Generating Visualizations
    
    # Plot 1: Target class distribution (Count Plot)
    plt.figure(figsize=(6, 5))
    ax = sns.countplot(data=df, x='TARGET', hue='TARGET', palette='Set2', legend=False)
    plt.title("Distribution of Credit Approval Target\n(0 = Approved/Good, 1 = Rejected/Bad)")
    plt.xlabel("Target Status")
    plt.ylabel("Number of Applicants")
    
    # Add percentage labels
    total = len(df)
    for p in ax.patches:
        percentage = f'{100 * p.get_height() / total:.1f}%'
        x_coord = p.get_x() + p.get_width() / 2 - 0.1
        y_coord = p.get_height() + 300
        ax.annotate(percentage, (x_coord, y_coord))
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "01_target_distribution.png"), dpi=150)
    plt.close()
    print("Saved: 01_target_distribution.png")

    # Plot 2: Distribution of Income (Histogram & Density)
    plt.figure(figsize=(10, 6))
    # Filter out extreme income outliers for better visualization
    income_cutoff = df['AMT_INCOME_TOTAL'].quantile(0.99)
    filtered_income = df[df['AMT_INCOME_TOTAL'] <= income_cutoff]
    
    sns.histplot(data=filtered_income, x='AMT_INCOME_TOTAL', hue='TARGET', 
                 kde=True, element='step', stat='density', common_norm=False, palette='Set2')
    plt.title("Income Distribution by Credit Approval Status (99th percentile)")
    plt.xlabel("Total Annual Income ($)")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "02_income_distribution.png"), dpi=150)
    plt.close()
    print("Saved: 02_income_distribution.png")

    # Plot 3: Distribution of Age (Histogram)
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='AGE', hue='TARGET', kde=True, multiple='stack', palette='Set1', bins=30)
    plt.title("Age Distribution of Applicants by Approval Status")
    plt.xlabel("Age (Years)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "03_age_distribution.png"), dpi=150)
    plt.close()
    print("Saved: 03_age_distribution.png")

    # Plot 4: Boxplot of Years Employed
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df[df['IS_UNEMPLOYED'] == 0], x='TARGET', y='YEARS_EMPLOYED', hue='TARGET', palette='Set2', legend=False)
    plt.title("Years Employed of Employed Applicants by Approval Status")
    plt.xlabel("Target Status (0=Approved, 1=Rejected)")
    plt.ylabel("Years of Employment")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "04_employment_years_boxplot.png"), dpi=150)
    plt.close()
    print("Saved: 04_employment_years_boxplot.png")

    # Plot 5: Delinquency (Rejection) Rate by Education Type (Bar Chart)
    plt.figure(figsize=(10, 6))
    # Calculate target rate (rejection rate) per category
    edu_rates = df.groupby('NAME_EDUCATION_TYPE')['TARGET'].mean().reset_index()
    edu_rates = edu_rates.sort_values(by='TARGET', ascending=False)
    
    sns.barplot(data=edu_rates, x='TARGET', y='NAME_EDUCATION_TYPE', hue='NAME_EDUCATION_TYPE', palette='viridis', legend=False)
    plt.title("Rejection (High-Risk) Rate by Education Level")
    plt.xlabel("Rejection Rate (Proportion of Target = 1)")
    plt.ylabel("Education Level")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "05_rejection_by_education.png"), dpi=150)
    plt.close()
    print("Saved: 05_rejection_by_education.png")

    # Plot 6: Correlation Heatmap for Numerical Columns
    plt.figure(figsize=(8, 6))
    corr_cols = numerical_cols + ['TARGET']
    corr_matrix = df[corr_cols].corr()
    
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, vmin=-1, vmax=1)
    plt.title("Correlation Heatmap of Numerical Features & Target")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "06_correlation_heatmap.png"), dpi=150)
    plt.close()
    print("Saved: 06_correlation_heatmap.png")
    
    # 4. Outlier analysis
    print("\n[Outlier Analysis]")
    # Outlier in DAYS_EMPLOYED: check value 365243 (meaning unemployed)
    unemployed_count = (df['DAYS_EMPLOYED'] == 365243).sum()
    print(f"Number of rows with DAYS_EMPLOYED = 365243 (Unemployed code): {unemployed_count} ({unemployed_count / len(df) * 100:.2f}%)")
    
    # 5. Missing values report
    print("\n[Missing Values Report]")
    missing_counts = df.isnull().sum()
    print(missing_counts[missing_counts > 0])
    
    print("\nEDA completed successfully.")

if __name__ == "__main__":
    run_eda()
