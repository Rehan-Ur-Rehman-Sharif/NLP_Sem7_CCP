"""
NLP Toxicity Detection Package
Tools for detecting toxicity in text using rule-based and ML approaches.
"""

from .rule_based_detector import RuleBasedDetector
from .toxicity_classifier import ToxicityClassifier
from .model_trainer import ToxicityModelTrainer
from .utils import (
    clean_text,
    preprocess_text,
    create_sample_dataset,
    load_and_preprocess_dataset,
    get_dataset_statistics
)

__version__ = "1.0.0"

__all__ = [
    'RuleBasedDetector',
    'ToxicityClassifier',
    'ToxicityModelTrainer',
    'clean_text',
    'preprocess_text',
    'create_sample_dataset',
    'load_and_preprocess_dataset',
    'get_dataset_statistics'
]
