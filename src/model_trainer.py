"""
ML Model Trainer for Toxicity Detection
Trains machine learning models to detect toxic content with context awareness.
"""

import yaml
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Optional
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, classification_report, confusion_matrix
)
import warnings
warnings.filterwarnings('ignore')


class ToxicityModelTrainer:
    """
    Trainer for toxicity detection models using various ML algorithms.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the trainer with configuration.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.ml_config = self.config.get('ml_model', {})
        self.data_config = self.config.get('data', {})
        
        self.model = None
        self.vectorizer = None
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        config_file = Path(config_path)
        if config_file.exists():
            with open(config_file, 'r') as f:
                return yaml.safe_load(f)
        return {}
    
    def load_data(self, data_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load training data from CSV file.
        Expected columns: 'text', 'label' (0 for non-toxic, 1 for toxic)
        
        Args:
            data_path: Path to data file (optional)
            
        Returns:
            DataFrame with text and labels
        """
        if data_path is None:
            data_path = self.data_config.get('sample_data', 'data/sample_dataset.csv')
        
        df = pd.read_csv(data_path)
        
        # Validate required columns
        if 'text' not in df.columns or 'label' not in df.columns:
            raise ValueError("Dataset must contain 'text' and 'label' columns")
        
        return df
    
    def preprocess_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess data for training.
        
        Args:
            df: DataFrame with text and labels
            
        Returns:
            Tuple of (X, y) arrays
        """
        # Drop any missing values
        df = df.dropna()
        
        # Extract features and labels
        X = df['text'].values
        y = df['label'].values
        
        return X, y
    
    def create_vectorizer(self) -> object:
        """
        Create text vectorizer based on configuration.
        
        Returns:
            Vectorizer instance
        """
        vectorizer_type = self.ml_config.get('vectorizer', 'tfidf')
        max_features = self.ml_config.get('max_features', 5000)
        ngram_range = tuple(self.ml_config.get('ngram_range', [1, 3]))
        
        if vectorizer_type == 'tfidf':
            return TfidfVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                lowercase=True,
                stop_words='english'
            )
        else:
            return CountVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                lowercase=True,
                stop_words='english'
            )
    
    def create_model(self) -> object:
        """
        Create ML model based on configuration.
        
        Returns:
            Model instance
        """
        model_type = self.ml_config.get('model_type', 'logistic_regression')
        
        if model_type == 'logistic_regression':
            return LogisticRegression(max_iter=1000, random_state=42)
        elif model_type == 'naive_bayes':
            return MultinomialNB()
        elif model_type == 'random_forest':
            return RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == 'svm':
            return SVC(kernel='linear', random_state=42)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Train the toxicity detection model.
        
        Args:
            X_train: Training texts
            y_train: Training labels
        """
        print("Creating vectorizer...")
        self.vectorizer = self.create_vectorizer()
        X_train_vec = self.vectorizer.fit_transform(X_train)
        
        print(f"Training {self.ml_config.get('model_type', 'logistic_regression')} model...")
        self.model = self.create_model()
        self.model.fit(X_train_vec, y_train)
        
        print("Training completed!")
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Evaluate model performance.
        
        Args:
            X_test: Test texts
            y_test: Test labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None or self.vectorizer is None:
            raise ValueError("Model not trained yet!")
        
        # Vectorize test data
        X_test_vec = self.vectorizer.transform(X_test)
        
        # Make predictions
        y_pred = self.model.predict(X_test_vec)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0)
        }
        
        return metrics
    
    def save_model(self, model_path: Optional[str] = None, 
                   vectorizer_path: Optional[str] = None) -> None:
        """
        Save trained model and vectorizer.
        
        Args:
            model_path: Path to save model
            vectorizer_path: Path to save vectorizer
        """
        if self.model is None or self.vectorizer is None:
            raise ValueError("Model not trained yet!")
        
        if model_path is None:
            model_path = self.ml_config.get('model_path', 'models/toxicity_model.joblib')
        if vectorizer_path is None:
            vectorizer_path = self.ml_config.get('vectorizer_path', 'models/vectorizer.joblib')
        
        # Ensure directories exist
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        Path(vectorizer_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save model and vectorizer
        joblib.dump(self.model, model_path)
        joblib.dump(self.vectorizer, vectorizer_path)
        
        print(f"Model saved to: {model_path}")
        print(f"Vectorizer saved to: {vectorizer_path}")
    
    def load_model(self, model_path: Optional[str] = None,
                   vectorizer_path: Optional[str] = None) -> None:
        """
        Load pre-trained model and vectorizer.
        
        Args:
            model_path: Path to model file
            vectorizer_path: Path to vectorizer file
        """
        if model_path is None:
            model_path = self.ml_config.get('model_path', 'models/toxicity_model.joblib')
        if vectorizer_path is None:
            vectorizer_path = self.ml_config.get('vectorizer_path', 'models/vectorizer.joblib')
        
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)
        
        print("Model and vectorizer loaded successfully!")
    
    def train_pipeline(self, data_path: Optional[str] = None) -> Dict:
        """
        Complete training pipeline: load, preprocess, train, evaluate.
        
        Args:
            data_path: Path to training data
            
        Returns:
            Dictionary with evaluation metrics
        """
        print("=" * 80)
        print("TOXICITY DETECTION MODEL TRAINING PIPELINE")
        print("=" * 80)
        
        # Load data
        print("\n1. Loading data...")
        df = self.load_data(data_path)
        print(f"   Loaded {len(df)} samples")
        print(f"   Toxic: {sum(df['label'] == 1)}, Non-toxic: {sum(df['label'] == 0)}")
        
        # Preprocess
        print("\n2. Preprocessing data...")
        X, y = self.preprocess_data(df)
        
        # Split data
        test_size = self.ml_config.get('test_size', 0.2)
        random_state = self.ml_config.get('random_state', 42)
        
        print(f"\n3. Splitting data (test_size={test_size})...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Train
        print(f"\n4. Training model...")
        self.train(X_train, y_train)
        
        # Evaluate
        print(f"\n5. Evaluating model...")
        metrics = self.evaluate(X_test, y_test)
        
        print("\n" + "=" * 80)
        print("EVALUATION RESULTS")
        print("=" * 80)
        for metric, value in metrics.items():
            print(f"{metric.capitalize()}: {value:.4f}")
        
        # Save model
        print("\n6. Saving model...")
        self.save_model()
        
        print("\n" + "=" * 80)
        print("Training pipeline completed successfully!")
        print("=" * 80)
        
        return metrics


def main():
    """Demo usage of the model trainer."""
    trainer = ToxicityModelTrainer()
    
    # Check if sample data exists
    sample_data_path = 'data/sample_dataset.csv'
    if not Path(sample_data_path).exists():
        print(f"Sample dataset not found at {sample_data_path}")
        print("Please create a sample dataset first or provide a data path.")
        return
    
    # Train model
    metrics = trainer.train_pipeline()


if __name__ == "__main__":
    main()
