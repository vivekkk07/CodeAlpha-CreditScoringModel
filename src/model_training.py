"""
Model training module for credit scoring.
Implements Logistic Regression, Decision Trees, and Random Forest models.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import joblib
import warnings

warnings.filterwarnings('ignore')


class CreditScoringModel:
    """
    Wrapper class for credit scoring models.
    """
    
    def __init__(self, model_type='logistic_regression', random_state=42):
        """
        Initialize the model.
        
        Parameters:
        -----------
        model_type : str
            Type of model: 'logistic_regression', 'decision_tree', or 'random_forest'
        random_state : int
            Random seed for reproducibility
        """
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.is_trained = False
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the selected model with optimal hyperparameters."""
        if self.model_type == 'logistic_regression':
            self.model = LogisticRegression(
                max_iter=1000,
                random_state=self.random_state,
                solver='lbfgs',
                class_weight='balanced'
            )
        elif self.model_type == 'decision_tree':
            self.model = DecisionTreeClassifier(
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=self.random_state,
                class_weight='balanced'
            )
        elif self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=self.random_state,
                class_weight='balanced',
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(self, X_train, y_train):
        """
        Train the model.
        
        Parameters:
        -----------
        X_train : pd.DataFrame or np.ndarray
            Training features
        y_train : pd.Series or np.ndarray
            Training labels
        """
        print(f"Training {self.model_type}...")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        print(f"✓ Training completed")
    
    def predict(self, X):
        """
        Make predictions.
        
        Parameters:
        -----------
        X : pd.DataFrame or np.ndarray
            Features for prediction
        
        Returns:
        --------
        np.ndarray
            Predicted class labels
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Predict class probabilities.
        
        Parameters:
        -----------
        X : pd.DataFrame or np.ndarray
            Features for prediction
        
        Returns:
        --------
        np.ndarray
            Predicted probabilities for each class
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict_proba(X)
    
    def get_feature_importance(self, feature_names=None):
        """
        Get feature importance scores.
        
        Parameters:
        -----------
        feature_names : list
            Names of features (optional)
        
        Returns:
        --------
        pd.DataFrame
            Feature importance scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting feature importance")
        
        if self.model_type == 'logistic_regression':
            importance = np.abs(self.model.coef_[0])
        elif self.model_type in ['decision_tree', 'random_forest']:
            importance = self.model.feature_importances_
        else:
            return None
        
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(len(importance))]
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        return importance_df
    
    def save_model(self, filepath):
        """
        Save trained model to disk.
        
        Parameters:
        -----------
        filepath : str
            Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        joblib.dump(self.model, filepath)
        print(f"✓ Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load a trained model from disk.
        
        Parameters:
        -----------
        filepath : str
            Path to the model file
        """
        self.model = joblib.load(filepath)
        self.is_trained = True
        print(f"✓ Model loaded from {filepath}")
    
    def get_model_info(self):
        """
        Get information about the model.
        
        Returns:
        --------
        dict
            Model information
        """
        return {
            'Model Type': self.model_type,
            'Is Trained': self.is_trained,
            'Model Object': self.model,
            'Parameters': self.model.get_params() if self.is_trained else None
        }


class ModelFactory:
    """
    Factory class to easily create and train multiple models.
    """
    
    @staticmethod
    def create_models(random_state=42):
        """
        Create all three models.
        
        Parameters:
        -----------
        random_state : int
            Random seed
        
        Returns:
        --------
        dict
            Dictionary of models
        """
        models = {
            'Logistic Regression': CreditScoringModel('logistic_regression', random_state),
            'Decision Tree': CreditScoringModel('decision_tree', random_state),
            'Random Forest': CreditScoringModel('random_forest', random_state)
        }
        return models
    
    @staticmethod
    def train_all_models(models, X_train, y_train):
        """
        Train all models.
        
        Parameters:
        -----------
        models : dict
            Dictionary of models
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training labels
        """
        print("\n" + "="*60)
        print("TRAINING ALL MODELS")
        print("="*60)
        
        for name, model in models.items():
            print(f"\n{name}:")
            model.train(X_train, y_train)


if __name__ == '__main__':
    # Example usage
    from data_preprocessing import load_and_prepare_data
    
    print("Loading and preparing data...")
    X_train, X_test, y_train, y_test, scaler = load_and_prepare_data(
        '../data/credit_data.csv'
    )
    
    # Create and train models
    models = ModelFactory.create_models()
    ModelFactory.train_all_models(models, X_train, y_train)
    
    # Make predictions
    print("\n" + "="*60)
    print("MAKING PREDICTIONS")
    print("="*60)
    for name, model in models.items():
        predictions = model.predict(X_test)
        print(f"\n{name}: {predictions[:5]}")
        
        # Get feature importance
        importance = model.get_feature_importance(X_train.columns.tolist())
        print(f"\nTop 5 Important Features:")
        print(importance.head())
