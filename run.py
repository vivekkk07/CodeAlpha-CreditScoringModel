"""
Main execution script for the credit scoring model project.
Orchestrates data generation, preprocessing, model training, and evaluation.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.data_preprocessing import (
    generate_synthetic_data, load_and_prepare_data, get_feature_importance_info
)
from src.model_training import ModelFactory, CreditScoringModel
from src.model_evaluation import ModelEvaluator, ModelComparator, print_evaluation_summary


def ensure_directories():
    """Ensure required directories exist."""
    dirs = ['data', 'models', 'results', 'results/confusion_matrices', 'results/roc_curves']
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
    print("✓ Directory structure ready")


def generate_data(n_samples=5000):
    """Generate synthetic credit data."""
    print("\n" + "="*70)
    print("STEP 1: GENERATING SYNTHETIC DATA")
    print("="*70)
    
    data_path = 'data/credit_data.csv'
    
    if os.path.exists(data_path):
        print(f"Data file already exists at {data_path}")
        response = input("Regenerate? (y/n): ").lower()
        if response != 'y':
            print("Using existing data file")
            return data_path
    
    print(f"\nGenerating {n_samples} samples of synthetic credit data...")
    df = generate_synthetic_data(n_samples=n_samples, random_state=42)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Features: {list(df.columns)}")
    print(f"\nCredit Score Distribution:")
    print(df['Credit_Score'].value_counts())
    print(f"\nBasic Statistics:")
    print(df.describe().round(2))
    
    df.to_csv(data_path, index=False)
    print(f"\n✓ Data saved to {data_path}")
    
    return data_path


def prepare_data(data_path):
    """Load and prepare data."""
    print("\n" + "="*70)
    print("STEP 2: DATA PREPARATION & FEATURE ENGINEERING")
    print("="*70)
    
    print("\nLoading and preparing data...")
    X_train, X_test, y_train, y_test, scaler = load_and_prepare_data(
        data_path, test_size=0.2, random_state=42
    )
    
    print(f"\nTraining set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")
    print(f"\nFeatures used: {list(X_train.columns)}")
    print(f"\nFeature descriptions:")
    feature_info = get_feature_importance_info()
    for feature, description in feature_info.items():
        if feature in X_train.columns:
            print(f"  • {feature}: {description}")
    
    print(f"\n✓ Data preparation complete")
    
    return X_train, X_test, y_train, y_test, scaler


def train_models(X_train, y_train):
    """Train all models."""
    print("\n" + "="*70)
    print("STEP 3: MODEL TRAINING")
    print("="*70 + "\n")
    
    models = ModelFactory.create_models(random_state=42)
    ModelFactory.train_all_models(models, X_train, y_train)
    
    # Save models
    print("\n" + "-"*70)
    print("Saving trained models...")
    for name, model in models.items():
        filename = f"models/{name.lower().replace(' ', '_')}.pkl"
        model.save_model(filename)
    
    return models


def evaluate_models(models, X_test, y_test):
    """Evaluate all models."""
    print("\n" + "="*70)
    print("STEP 4: MODEL EVALUATION")
    print("="*70)
    
    comparator = ModelComparator()
    evaluators = {}
    
    for name, model in models.items():
        print(f"\n{'─'*70}")
        print(f"Evaluating {name}...")
        print('─'*70)
        
        # Get predictions
        predictions = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Create evaluator
        evaluator = ModelEvaluator(y_test, predictions, y_pred_proba)
        evaluators[name] = evaluator
        
        # Print summary
        print_evaluation_summary(name, evaluator)
        
        # Add to comparator
        comparator.add_model_result(name, evaluator)
        
        # Get feature importance
        importance = model.get_feature_importance(X_test.columns.tolist())
        print(f"Top 5 Important Features:")
        print(importance.head().to_string(index=False))
    
    return evaluators, comparator


def generate_visualizations(evaluators, comparator):
    """Generate evaluation visualizations."""
    print("\n" + "="*70)
    print("STEP 5: GENERATING VISUALIZATIONS")
    print("="*70 + "\n")
    
    # Confusion matrices
    print("Generating confusion matrices...")
    for name, evaluator in evaluators.items():
        filepath = f"results/confusion_matrices/{name.lower().replace(' ', '_')}_cm.png"
        evaluator.plot_confusion_matrix(filepath=filepath, title=f'Confusion Matrix - {name}')
    
    # ROC curves
    print("Generating ROC curves...")
    for name, evaluator in evaluators.items():
        filepath = f"results/roc_curves/{name.lower().replace(' ', '_')}_roc.png"
        evaluator.plot_roc_curve(filepath=filepath, title=f'ROC Curve - {name}')
    
    # Metrics bars
    print("Generating metrics bar charts...")
    for name, evaluator in evaluators.items():
        filepath = f"results/{name.lower().replace(' ', '_')}_metrics.png"
        evaluator.plot_metrics_bar(filepath=filepath, title=f'Metrics - {name}')
    
    # Model comparison
    print("Generating model comparison plot...")
    comparator.plot_comparison(
        filepath='results/model_comparison.png',
        title='Model Performance Comparison'
    )
    
    print("\n✓ All visualizations generated")


def print_summary(comparator):
    """Print final summary."""
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    comparator.print_comparison()
    
    best_model = comparator.get_best_model(metric='F1-Score')
    print(f"\n🏆 Best Model (by F1-Score): {best_model}")
    
    print("\n📊 Generated Outputs:")
    print("  • Trained models saved to: models/")
    print("  • Confusion matrices saved to: results/confusion_matrices/")
    print("  • ROC curves saved to: results/roc_curves/")
    print("  • Metrics plots saved to: results/")
    print("  • Model comparison saved to: results/model_comparison.csv")


def main():
    """Main execution function."""
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*15 + "CREDIT SCORING MODEL PROJECT" + " "*25 + "║")
    print("║" + " "*14 + "Machine Learning Classification Pipeline" + " "*14 + "║")
    print("╚" + "="*68 + "╝")
    
    try:
        # Setup
        ensure_directories()
        
        # Step 1: Generate data
        data_path = generate_data(n_samples=5000)
        
        # Step 2: Prepare data
        X_train, X_test, y_train, y_test, scaler = prepare_data(data_path)
        
        # Step 3: Train models
        models = train_models(X_train, y_train)
        
        # Step 4: Evaluate models
        evaluators, comparator = evaluate_models(models, X_test, y_test)
        
        # Step 5: Generate visualizations
        generate_visualizations(evaluators, comparator)
        
        # Save comparison
        comparator.save_comparison('results/model_comparison.csv')
        
        # Print summary
        print_summary(comparator)
        
        print("\n" + "="*70)
        print("✓ PROJECT COMPLETED SUCCESSFULLY!")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n✗ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
