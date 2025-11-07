"""
Toxicity Classifier
Uses trained ML model to predict toxicity in text with context awareness.
"""

import yaml
import joblib
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')


class ToxicityClassifier:
    """
    Classifier for detecting toxicity in text using trained ML models.
    Provides context-aware analysis for better detection of subtle toxicity.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the classifier with configuration.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.ml_config = self.config.get('ml_model', {})
        
        self.model = None
        self.vectorizer = None
        
        # Try to load pre-trained model
        self._try_load_model()
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        config_file = Path(config_path)
        if config_file.exists():
            with open(config_file, 'r') as f:
                return yaml.safe_load(f)
        return {}
    
    def _try_load_model(self) -> None:
        """Try to load pre-trained model if available."""
        model_path = self.ml_config.get('model_path', 'models/toxicity_model.joblib')
        vectorizer_path = self.ml_config.get('vectorizer_path', 'models/vectorizer.joblib')
        
        if Path(model_path).exists() and Path(vectorizer_path).exists():
            try:
                self.model = joblib.load(model_path)
                self.vectorizer = joblib.load(vectorizer_path)
                print("Pre-trained model loaded successfully!")
            except Exception as e:
                print(f"Warning: Could not load pre-trained model: {e}")
    
    def load_model(self, model_path: Optional[str] = None,
                   vectorizer_path: Optional[str] = None) -> None:
        """
        Load trained model and vectorizer.
        
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
    
    def predict(self, text: str) -> int:
        """
        Predict if text is toxic (1) or non-toxic (0).
        
        Args:
            text: Input text
            
        Returns:
            Prediction: 1 for toxic, 0 for non-toxic
        """
        if self.model is None or self.vectorizer is None:
            raise ValueError("Model not loaded! Please train or load a model first.")
        
        # Vectorize text
        text_vec = self.vectorizer.transform([text])
        
        # Make prediction
        prediction = self.model.predict(text_vec)[0]
        
        return int(prediction)
    
    def predict_proba(self, text: str) -> float:
        """
        Predict toxicity probability.
        
        Args:
            text: Input text
            
        Returns:
            Probability of text being toxic (0-1)
        """
        if self.model is None or self.vectorizer is None:
            raise ValueError("Model not loaded! Please train or load a model first.")
        
        # Check if model supports probability prediction
        if not hasattr(self.model, 'predict_proba'):
            # For models without predict_proba, use binary prediction
            return float(self.predict(text))
        
        # Vectorize text
        text_vec = self.vectorizer.transform([text])
        
        # Get probability of toxic class
        proba = self.model.predict_proba(text_vec)[0]
        
        # Return probability of toxic class (index 1)
        return proba[1] if len(proba) > 1 else proba[0]
    
    def analyze_with_context(self, text: str, context: Optional[List[str]] = None) -> Dict:
        """
        Analyze text with surrounding context for better toxicity detection.
        
        Args:
            text: Main text to analyze
            context: Optional list of surrounding texts for context
            
        Returns:
            Dictionary with analysis results
        """
        # Predict main text
        prediction = self.predict(text)
        probability = self.predict_proba(text)
        
        result = {
            'text': text,
            'is_toxic': bool(prediction),
            'toxicity_probability': probability,
            'severity': self._get_severity(probability)
        }
        
        # If context provided, analyze context as well
        if context:
            context_scores = []
            for ctx_text in context:
                try:
                    ctx_prob = self.predict_proba(ctx_text)
                    context_scores.append(ctx_prob)
                except (ValueError, RuntimeError, Exception) as e:
                    # Skip context texts that cause errors
                    continue
            
            if context_scores:
                avg_context_score = np.mean(context_scores)
                
                # Adjust main score based on context
                # If context is toxic, increase suspicion of main text
                if avg_context_score > 0.3:
                    adjusted_prob = (probability + avg_context_score) / 2
                    result['adjusted_probability'] = adjusted_prob
                    result['context_influenced'] = True
                    result['context_toxicity'] = avg_context_score
        
        return result
    
    def _get_severity(self, probability: float) -> str:
        """
        Determine severity level based on probability.
        
        Args:
            probability: Toxicity probability
            
        Returns:
            Severity level: 'low', 'medium', or 'high'
        """
        if probability >= 0.7:
            return 'high'
        elif probability >= 0.4:
            return 'medium'
        else:
            return 'low'
    
    def batch_predict(self, texts: List[str]) -> List[Dict]:
        """
        Predict toxicity for multiple texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        
        for text in texts:
            try:
                prediction = self.predict(text)
                probability = self.predict_proba(text)
                
                results.append({
                    'text': text,
                    'is_toxic': bool(prediction),
                    'toxicity_probability': probability,
                    'severity': self._get_severity(probability)
                })
            except Exception as e:
                results.append({
                    'text': text,
                    'error': str(e)
                })
        
        return results


def main():
    """Demo usage of the toxicity classifier."""
    classifier = ToxicityClassifier()
    
    if classifier.model is None:
        print("No pre-trained model found!")
        print("Please train a model first using model_trainer.py")
        return
    
    # Test cases
    test_texts = [
        "You are a wonderful person!",
        "I hate everything about this.",
        "You should go away and never come back.",
        "Yeah right, you're so smart.",
        "Oh how wonderful, nobody likes you.",
        "Have a great day!",
        "You stupid idiot.",
        "I think you should reconsider your approach.",
    ]
    
    print("=" * 80)
    print("ML-BASED TOXICITY DETECTION DEMO")
    print("=" * 80)
    
    results = classifier.batch_predict(test_texts)
    
    for result in results:
        if 'error' in result:
            print(f"\nText: {result['text']}")
            print(f"Error: {result['error']}")
        else:
            print(f"\nText: {result['text']}")
            print(f"Toxic: {result['is_toxic']}")
            print(f"Probability: {result['toxicity_probability']:.3f}")
            print(f"Severity: {result['severity']}")
        print("-" * 80)
    
    # Demo context-aware analysis
    print("\n" + "=" * 80)
    print("CONTEXT-AWARE ANALYSIS DEMO")
    print("=" * 80)
    
    main_text = "You're really something."
    context = [
        "You always mess things up.",
        "Everyone knows you're incompetent.",
        "I can't believe I have to work with you."
    ]
    
    result = classifier.analyze_with_context(main_text, context)
    
    print(f"\nMain Text: {result['text']}")
    print(f"Toxic: {result['is_toxic']}")
    print(f"Base Probability: {result['toxicity_probability']:.3f}")
    
    if 'context_influenced' in result:
        print(f"Context Toxicity: {result['context_toxicity']:.3f}")
        print(f"Adjusted Probability: {result['adjusted_probability']:.3f}")
        print("Note: Context suggests potential hidden toxicity!")


if __name__ == "__main__":
    main()
