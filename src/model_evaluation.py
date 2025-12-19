"""
Model evaluation module for credit scoring.
Computes comprehensive evaluation metrics and visualizations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
    auc
)
import warnings

warnings.filterwarnings('ignore')

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)


class ModelEvaluator:
    """
    Comprehensive model evaluation and metrics computation.
    """
    
    def __init__(self, y_true, y_pred, y_pred_proba=None):
        """
        Initialize the evaluator.
        
        Parameters:
        -----------
        y_true : array-like
            True labels
        y_pred : array-like
            Predicted labels
        y_pred_proba : array-like, optional
            Predicted probabilities for positive class
        """
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.y_pred_proba = np.array(y_pred_proba) if y_pred_proba is not None else None
    
    def get_metrics(self):
        """
        Compute all evaluation metrics.
        
        Returns:
        --------
        dict
            Dictionary of all metrics
        """
        metrics = {
            'Accuracy': accuracy_score(self.y_true, self.y_pred),
            'Precision': precision_score(self.y_true, self.y_pred),
            'Recall': recall_score(self.y_true, self.y_pred),
            'F1-Score': f1_score(self.y_true, self.y_pred),
        }
        
        # Add ROC-AUC if probabilities are available
        if self.y_pred_proba is not None:
            metrics['ROC-AUC'] = roc_auc_score(self.y_true, self.y_pred_proba)
        
        return metrics
    
    def get_confusion_matrix(self):
        """
        Get confusion matrix.
        
        Returns:
        --------
        np.ndarray
            Confusion matrix
        """
        return confusion_matrix(self.y_true, self.y_pred)
    
    def get_confusion_matrix_dict(self):
        """
        Get confusion matrix as a dictionary with labels.
        
        Returns:
        --------
        dict
            Confusion matrix with labels
        """
        cm = self.get_confusion_matrix()
        return {
            'True Negatives': cm[0, 0],
            'False Positives': cm[0, 1],
            'False Negatives': cm[1, 0],
            'True Positives': cm[1, 1]
        }
    
    def get_classification_report(self):
        """
        Get detailed classification report.
        
        Returns:
        --------
        str
            Classification report
        """
        return classification_report(
            self.y_true, self.y_pred,
            target_names=['Poor Credit (0)', 'Good Credit (1)']
        )
    
    def plot_confusion_matrix(self, filepath=None, title='Confusion Matrix'):
        """
        Plot confusion matrix heatmap.
        
        Parameters:
        -----------
        filepath : str, optional
            Path to save the plot
        title : str
            Title of the plot
        """
        cm = self.get_confusion_matrix()
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                    xticklabels=['Poor (0)', 'Good (1)'],
                    yticklabels=['Poor (0)', 'Good (1)'])
        plt.title(title, fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        if filepath:
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"✓ Confusion matrix saved to {filepath}")
        
        return plt
    
    def plot_roc_curve(self, filepath=None, title='ROC Curve'):
        """
        Plot ROC curve.
        
        Parameters:
        -----------
        filepath : str, optional
            Path to save the plot
        title : str
            Title of the plot
        """
        if self.y_pred_proba is None:
            print("Warning: Probabilities not available for ROC curve")
            return None
        
        fpr, tpr, _ = roc_curve(self.y_true, self.y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=11)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if filepath:
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"✓ ROC curve saved to {filepath}")
        
        return plt
    
    def plot_metrics_bar(self, filepath=None, title='Model Metrics'):
        """
        Plot metrics as a bar chart.
        
        Parameters:
        -----------
        filepath : str, optional
            Path to save the plot
        title : str
            Title of the plot
        """
        metrics = self.get_metrics()
        
        plt.figure(figsize=(10, 6))
        colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
        bars = plt.bar(metrics.keys(), metrics.values(), color=colors[:len(metrics)])
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=11)
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.ylabel('Score', fontsize=12)
        plt.ylim([0, 1.1])
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        if filepath:
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"✓ Metrics bar chart saved to {filepath}")
        
        return plt


class ModelComparator:
    """
    Compare multiple models and generate comparison reports.
    """
    
    def __init__(self):
        """Initialize the comparator."""
        self.results = []
    
    def add_model_result(self, model_name, evaluator):
        """
        Add evaluation results from a model.
        
        Parameters:
        -----------
        model_name : str
            Name of the model
        evaluator : ModelEvaluator
            Evaluator instance with results
        """
        metrics = evaluator.get_metrics()
        metrics['Model'] = model_name
        self.results.append(metrics)
    
    def get_comparison_dataframe(self):
        """
        Get comparison results as a DataFrame.
        
        Returns:
        --------
        pd.DataFrame
            Model comparison results
        """
        df = pd.DataFrame(self.results)
        df = df.set_index('Model')
        return df
    
    def print_comparison(self):
        """Print formatted model comparison."""
        df = self.get_comparison_dataframe()
        
        print("\n" + "="*80)
        print("MODEL PERFORMANCE COMPARISON")
        print("="*80)
        print(df.to_string())
        print("="*80 + "\n")
    
    def save_comparison(self, filepath):
        """
        Save comparison results to CSV.
        
        Parameters:
        -----------
        filepath : str
            Path to save the CSV
        """
        df = self.get_comparison_dataframe()
        df.to_csv(filepath)
        print(f"✓ Comparison saved to {filepath}")
    
    def plot_comparison(self, filepath=None, title='Model Performance Comparison'):
        """
        Plot model comparison.
        
        Parameters:
        -----------
        filepath : str, optional
            Path to save the plot
        title : str
            Title of the plot
        """
        df = self.get_comparison_dataframe()
        
        plt.figure(figsize=(12, 6))
        x = np.arange(len(df.columns))
        width = 0.25
        
        for i, model in enumerate(df.index):
            plt.bar(x + i*width, df.loc[model], width, label=model)
        
        plt.xlabel('Metrics', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xticks(x + width, df.columns, rotation=45)
        plt.legend(fontsize=11)
        plt.ylim([0, 1.1])
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        if filepath:
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"✓ Comparison plot saved to {filepath}")
        
        return plt
    
    def get_best_model(self, metric='ROC-AUC'):
        """
        Get the best performing model.
        
        Parameters:
        -----------
        metric : str
            Metric to use for comparison
        
        Returns:
        --------
        str
            Name of best model
        """
        df = self.get_comparison_dataframe()
        if metric not in df.columns:
            metric = 'F1-Score'
        return df[metric].idxmax()


def print_evaluation_summary(model_name, evaluator):
    """
    Print comprehensive evaluation summary for a model.
    
    Parameters:
    -----------
    model_name : str
        Name of the model
    evaluator : ModelEvaluator
        Evaluator instance
    """
    print(f"\n{'='*70}")
    print(f"EVALUATION REPORT: {model_name}")
    print(f"{'='*70}\n")
    
    # Metrics
    metrics = evaluator.get_metrics()
    print("Metrics:")
    print("-" * 70)
    for metric, value in metrics.items():
        print(f"  {metric:.<40} {value:.4f}")
    
    # Confusion Matrix
    cm_dict = evaluator.get_confusion_matrix_dict()
    print("\nConfusion Matrix:")
    print("-" * 70)
    for label, value in cm_dict.items():
        print(f"  {label:.<40} {value}")
    
    # Classification Report
    print("\nDetailed Classification Report:")
    print("-" * 70)
    print(evaluator.get_classification_report())
    print(f"{'='*70}\n")


if __name__ == '__main__':
    # Example usage
    from model_training import ModelFactory
    from data_preprocessing import load_and_prepare_data
    
    print("Loading data...")
    X_train, X_test, y_train, y_test, scaler = load_and_prepare_data(
        '../data/credit_data.csv'
    )
    
    # Train models
    models = ModelFactory.create_models()
    ModelFactory.train_all_models(models, X_train, y_train)
    
    # Evaluate models
    print("\n" + "="*70)
    print("EVALUATING MODELS")
    print("="*70)
    
    comparator = ModelComparator()
    
    for name, model in models.items():
        predictions = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        evaluator = ModelEvaluator(y_test, predictions, y_pred_proba)
        print_evaluation_summary(name, evaluator)
        
        comparator.add_model_result(name, evaluator)
    
    # Print comparison
    comparator.print_comparison()
