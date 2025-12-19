# Credit Scoring Model

A machine learning project to predict individual creditworthiness using financial data. This project implements multiple classification algorithms (Logistic Regression, Decision Trees, and Random Forest) with comprehensive feature engineering and evaluation metrics.

## Overview

Credit scoring is a critical component of financial risk assessment. This project demonstrates a complete pipeline for building, training, and evaluating credit scoring models using realistic financial features.

**Key Features:**
- ✅ Feature engineering from financial history (income, debts, payment history, credit utilization)
- ✅ Multiple classification algorithms (Logistic Regression, Decision Trees, Random Forest)
- ✅ Comprehensive evaluation metrics (Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix)
- ✅ Data preprocessing and normalization
- ✅ Model persistence and comparison
- ✅ Jupyter notebook for exploration and visualization

## Dataset

The dataset includes the following features:
- **Income**: Annual income in thousands
- **Credit Utilization**: Percentage of credit limit used
- **Payment History**: Number of months with good payment history
- **Delinquent Accounts**: Number of delinquent accounts
- **Total Debt**: Total outstanding debt in thousands
- **Loan-to-Income Ratio**: Ratio of total debt to annual income
- **Years of Credit History**: Length of credit history
- **Recent Inquiries**: Number of recent credit inquiries
- **Target Variable**: Credit Score (0 = Poor/Default, 1 = Good/Approved)

## Project Structure

```
credit-scoring-model/
├── data/
│   └── credit_data.csv              # Synthetic dataset
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py        # Data loading, cleaning, feature engineering
│   ├── model_training.py            # Model implementation and training
│   ├── model_evaluation.py          # Evaluation metrics and model comparison
│   └── utils.py                     # Utility functions
├── models/
│   └── (trained models saved here)
├── results/
│   └── (evaluation results and plots)
├── notebooks/
│   └── credit_scoring_analysis.ipynb # Exploration and visualization
├── requirements.txt                 # Project dependencies
├── .gitignore                       # Git ignore rules
├── run.py                           # Main execution script
└── README.md                        # This file
```

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/credit-scoring-model.git
   cd credit-scoring-model
   ```

2. **Create a virtual environment (optional but recommended)**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Run the complete pipeline

Execute the main script to generate data, train models, and evaluate performance:

```bash
python run.py
```

This will:
1. Generate synthetic credit data
2. Preprocess and engineer features
3. Train multiple models
4. Evaluate and compare model performance
5. Save results and visualizations

### Generate synthetic data

```python
from src.data_preprocessing import generate_synthetic_data

# Generate 10,000 samples
df = generate_synthetic_data(n_samples=10000, random_state=42)
df.to_csv('data/credit_data.csv', index=False)
```

### Train a single model

```python
from src.model_training import CreditScoringModel
from src.data_preprocessing import load_and_prepare_data

# Load data
X_train, X_test, y_train, y_test = load_and_prepare_data('data/credit_data.csv')

# Train Logistic Regression model
model = CreditScoringModel(model_type='logistic_regression')
model.train(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)
```

### Evaluate models

```python
from src.model_evaluation import ModelEvaluator

evaluator = ModelEvaluator(y_test, predictions)
results = evaluator.get_metrics()
print(results)
```

## Model Performance

The project evaluates models using:

| Metric | Description |
|--------|-------------|
| **Accuracy** | Overall correctness of predictions |
| **Precision** | True Positives / (True Positives + False Positives) |
| **Recall** | True Positives / (True Positives + False Negatives) |
| **F1-Score** | Harmonic mean of Precision and Recall |
| **ROC-AUC** | Area under the Receiver Operating Characteristic curve |
| **Confusion Matrix** | True/False Positives and Negatives breakdown |

## Key Features

### Data Preprocessing
- Missing value handling
- Feature scaling and normalization
- Categorical encoding
- Train-test split (80-20)

### Feature Engineering
- Loan-to-income ratio calculation
- Payment history score aggregation
- Credit utilization percentage
- Delinquency indicators

### Models Implemented

1. **Logistic Regression**
   - Interpretable, baseline model
   - Fast training and inference
   - Good for understanding feature importance

2. **Decision Tree**
   - Non-parametric approach
   - Captures non-linear relationships
   - Easy to interpret decision boundaries

3. **Random Forest**
   - Ensemble of decision trees
   - Robust to outliers
   - High predictive power

## Results

After running `run.py`, you'll find:
- Model comparison metrics in `results/model_comparison.csv`
- Confusion matrices and ROC curves in `results/`
- Trained models in `models/`

Example output:
```
Model Performance Comparison:
┌─────────────────────┬──────────┬──────────┬──────────┬──────────┐
│ Model               │ Accuracy │ Precision│ Recall   │ ROC-AUC  │
├─────────────────────┼──────────┼──────────┼──────────┼──────────┤
│ Logistic Regression │ 0.8450   │ 0.8320   │ 0.8610   │ 0.9120   │
│ Decision Tree       │ 0.8320   │ 0.8190   │ 0.8480   │ 0.8950   │
│ Random Forest       │ 0.8720   │ 0.8640   │ 0.8790   │ 0.9380   │
└─────────────────────┴──────────┴──────────┴──────────┴──────────┘
```

## Jupyter Notebook

For interactive exploration and visualization, open the Jupyter notebook:

```bash
jupyter notebook notebooks/credit_scoring_analysis.ipynb
```

The notebook includes:
- Data exploration and visualization
- Distribution analysis
- Correlation heatmaps
- Model performance plots
- ROC curves and confusion matrices

## Dependencies

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning algorithms and evaluation metrics
- **matplotlib**: Data visualization
- **seaborn**: Statistical data visualization
- **jupyter**: Interactive notebooks

See `requirements.txt` for exact versions.

## Model Comparison

The Random Forest model typically performs best due to its ensemble nature and ability to capture complex patterns in credit data. However, Logistic Regression remains valuable for its interpretability.

### When to use each model:

- **Logistic Regression**: When interpretability is critical, or for regulatory compliance
- **Decision Tree**: When you need a single, easily explainable tree
- **Random Forest**: When maximizing predictive performance is the priority

## Future Enhancements

- [ ] Hyperparameter tuning with GridSearchCV
- [ ] XGBoost and Gradient Boosting implementations
- [ ] Feature importance analysis
- [ ] Handling class imbalance (SMOTE)
- [ ] Cross-validation for more robust evaluation
- [ ] Model explainability (SHAP values)
- [ ] API deployment with Flask/FastAPI

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

Created as a machine learning demonstration project.

## References

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [ROC-AUC Explained](https://en.wikipedia.org/wiki/Receiver_operating_characteristic)
- [Credit Scoring Models](https://en.wikipedia.org/wiki/Credit_score)
- [Feature Engineering Best Practices](https://developers.google.com/machine-learning/guides/rules-of-ml)

## Contact

For questions or suggestions, please open an issue on GitHub.

---

**Note**: This project uses synthetic data for demonstration purposes. Real credit scoring models require careful consideration of fairness, discrimination laws, and regulatory requirements.
