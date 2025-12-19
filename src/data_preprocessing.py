"""
Data preprocessing and feature engineering module for credit scoring.
Handles data loading, cleaning, feature engineering, and train-test splitting.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')


def generate_synthetic_data(n_samples=5000, random_state=42):
    """
    Generate synthetic credit data for model training and testing.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    random_state : int
        Random seed for reproducibility
    
    Returns:
    --------
    pd.DataFrame
        Synthetic credit dataset
    """
    np.random.seed(random_state)
    
    data = {
        'Income': np.random.uniform(20, 200, n_samples),  # in thousands
        'Credit_Utilization': np.random.uniform(0, 100, n_samples),  # percentage
        'Payment_History': np.random.randint(0, 120, n_samples),  # months
        'Delinquent_Accounts': np.random.randint(0, 10, n_samples),
        'Total_Debt': np.random.uniform(0, 500, n_samples),  # in thousands
        'Years_Credit_History': np.random.uniform(0, 50, n_samples),
        'Recent_Inquiries': np.random.randint(0, 15, n_samples),
        'Num_Credit_Accounts': np.random.randint(1, 20, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Create target variable with some logic
    # Better credit profile -> higher chance of good credit
    df['Credit_Score'] = (
        (df['Income'] > 40) * 0.3 +
        (df['Credit_Utilization'] < 50) * 0.2 +
        (df['Payment_History'] > 60) * 0.2 +
        (df['Delinquent_Accounts'] == 0) * 0.15 +
        (df['Years_Credit_History'] > 5) * 0.15
    )
    
    # Add some noise
    noise = np.random.binomial(1, 0.15, n_samples)
    df['Credit_Score'] = ((df['Credit_Score'] > 0.5) ^ (noise == 1)).astype(int)
    
    # Add some missing values randomly
    missing_idx = np.random.choice(n_samples, size=int(0.02 * n_samples), replace=False)
    df.loc[missing_idx, 'Payment_History'] = np.nan
    
    return df


def load_and_prepare_data(filepath, test_size=0.2, random_state=42):
    """
    Load and prepare data for model training.
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file
    test_size : float
        Proportion of data for testing
    random_state : int
        Random seed for reproducibility
    
    Returns:
    --------
    tuple
        (X_train, X_test, y_train, y_test, scaler)
    """
    # Load data
    df = pd.read_csv(filepath)
    
    # Handle missing values
    df = handle_missing_values(df)
    
    # Feature engineering
    df = engineer_features(df)
    
    # Separate features and target
    X = df.drop('Credit_Score', axis=1)
    y = df['Credit_Score']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame for easier handling
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)
    
    return X_train_scaled, X_test_scaled, y_train.reset_index(drop=True), y_test.reset_index(drop=True), scaler


def handle_missing_values(df):
    """
    Handle missing values in the dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with missing values handled
    """
    df = df.copy()
    
    # Fill numeric columns with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)
    
    return df


def engineer_features(df):
    """
    Engineer new features from existing ones.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with engineered features
    """
    df = df.copy()
    
    # Loan-to-income ratio (avoid division by zero)
    df['Loan_to_Income_Ratio'] = df['Total_Debt'] / (df['Income'] + 1)
    
    # Credit utilization squared (non-linear relationship)
    df['Credit_Utilization_Squared'] = df['Credit_Utilization'] ** 2
    
    # Payment history score (normalized)
    df['Payment_History_Score'] = df['Payment_History'] / (df['Payment_History'].max() + 1)
    
    # Debt-to-accounts ratio
    df['Debt_per_Account'] = df['Total_Debt'] / (df['Num_Credit_Accounts'] + 1)
    
    # Age of credit history indicator
    df['Is_Long_Credit_History'] = (df['Years_Credit_History'] > 10).astype(int)
    
    # Recent credit activity indicator
    df['High_Recent_Inquiries'] = (df['Recent_Inquiries'] > 5).astype(int)
    
    # Drop original features that were used for engineering
    # Keep original for interpretability
    df = df.drop(['Num_Credit_Accounts'], axis=1)
    
    return df


def get_feature_importance_info():
    """
    Return information about feature importance and meaning.
    
    Returns:
    --------
    dict
        Feature importance descriptions
    """
    return {
        'Income': 'Annual income - higher income correlates with better creditworthiness',
        'Credit_Utilization': 'Percentage of credit limit used - lower is better',
        'Payment_History': 'Months with good payment history - higher is better',
        'Delinquent_Accounts': 'Number of delinquent accounts - lower is better',
        'Total_Debt': 'Total outstanding debt - lower is better',
        'Years_Credit_History': 'Length of credit history - longer is typically better',
        'Recent_Inquiries': 'Recent credit inquiries - fewer is better',
        'Loan_to_Income_Ratio': 'Total debt relative to income - lower is better',
        'Credit_Utilization_Squared': 'Squared credit utilization for non-linear effect',
        'Payment_History_Score': 'Normalized payment history score',
        'Debt_per_Account': 'Average debt per credit account',
        'Is_Long_Credit_History': 'Binary indicator for long credit history',
        'High_Recent_Inquiries': 'Binary indicator for many recent inquiries',
    }


if __name__ == '__main__':
    # Example usage
    print("Generating synthetic credit data...")
    df = generate_synthetic_data(n_samples=5000)
    
    print(f"\nDataset shape: {df.shape}")
    print(f"\nFirst few rows:\n{df.head()}")
    print(f"\nData types:\n{df.dtypes}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nCredit Score distribution:\n{df['Credit_Score'].value_counts()}")
    print(f"\nBasic statistics:\n{df.describe()}")
    
    # Save to CSV
    df.to_csv('../data/credit_data.csv', index=False)
    print("\nData saved to '../data/credit_data.csv'")
