"""
Main Entry Point for Toxicity Detection System
Provides unified interface for both rule-based and ML-based detection.
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.rule_based_detector import RuleBasedDetector
from src.toxicity_classifier import ToxicityClassifier
from src.model_trainer import ToxicityModelTrainer
from src.utils import create_sample_dataset


def run_rule_based_detection(text: str):
    """Run rule-based toxicity detection."""
    print("=" * 80)
    print("RULE-BASED TOXICITY DETECTION")
    print("=" * 80)
    
    detector = RuleBasedDetector()
    result = detector.analyze(text)
    
    print(f"\nText: {result['text']}")
    print(f"Toxic: {result['is_toxic']}")
    print(f"Score: {result['toxicity_score']:.3f}")
    print(f"Severity: {result['details']['severity']}")
    
    if result['details']['toxic_keywords']:
        print(f"Toxic Keywords: {', '.join(result['details']['toxic_keywords'])}")
    
    if result['details']['context_patterns']:
        print(f"Context Patterns Detected: {len(result['details']['context_patterns'])}")
    
    if result['details']['has_sarcasm']:
        print("⚠ Sarcasm detected - potential hidden toxicity!")
    
    print("=" * 80)


def run_ml_detection(text: str):
    """Run ML-based toxicity detection."""
    print("=" * 80)
    print("ML-BASED TOXICITY DETECTION")
    print("=" * 80)
    
    classifier = ToxicityClassifier()
    
    if classifier.model is None:
        print("\n⚠ No trained model found!")
        print("Please train a model first using: python main.py --train")
        return
    
    result = classifier.analyze_with_context(text)
    
    print(f"\nText: {result['text']}")
    print(f"Toxic: {result['is_toxic']}")
    print(f"Probability: {result['toxicity_probability']:.3f}")
    print(f"Severity: {result['severity']}")
    
    print("=" * 80)


def run_combined_detection(text: str):
    """Run both rule-based and ML-based detection."""
    print("\n" + "=" * 80)
    print("COMBINED TOXICITY DETECTION")
    print("=" * 80)
    
    # Rule-based
    print("\n[1] Rule-Based Analysis:")
    print("-" * 80)
    detector = RuleBasedDetector()
    rule_result = detector.analyze(text)
    
    print(f"Toxic: {rule_result['is_toxic']}")
    print(f"Score: {rule_result['toxicity_score']:.3f}")
    print(f"Severity: {rule_result['details']['severity']}")
    
    # ML-based
    print("\n[2] ML-Based Analysis:")
    print("-" * 80)
    classifier = ToxicityClassifier()
    
    if classifier.model is not None:
        ml_result = classifier.analyze_with_context(text)
        print(f"Toxic: {ml_result['is_toxic']}")
        print(f"Probability: {ml_result['toxicity_probability']:.3f}")
        print(f"Severity: {ml_result['severity']}")
        
        # Combined decision
        print("\n[3] Combined Decision:")
        print("-" * 80)
        
        combined_toxic = rule_result['is_toxic'] or ml_result['is_toxic']
        combined_score = (rule_result['toxicity_score'] + ml_result['toxicity_probability']) / 2
        
        print(f"Final Decision: {'TOXIC' if combined_toxic else 'NON-TOXIC'}")
        print(f"Combined Score: {combined_score:.3f}")
        
        if rule_result['is_toxic'] and ml_result['is_toxic']:
            print("⚠ Both methods agree: HIGH CONFIDENCE")
        elif rule_result['is_toxic'] or ml_result['is_toxic']:
            print("⚠ Methods disagree: MODERATE CONFIDENCE")
    else:
        print("⚠ No trained ML model available")
        print("Using only rule-based detection")
    
    print("=" * 80)


def train_model(data_path: str = None):
    """Train a new toxicity detection model."""
    print("=" * 80)
    print("TRAINING TOXICITY DETECTION MODEL")
    print("=" * 80)
    
    # Create sample dataset if no data path provided
    if data_path is None:
        sample_path = "data/sample_dataset.csv"
        if not Path(sample_path).exists():
            print("\nCreating sample dataset...")
            create_sample_dataset(num_samples=100)
        data_path = sample_path
    
    # Train model
    trainer = ToxicityModelTrainer()
    trainer.train_pipeline(data_path)


def demo_mode():
    """Run demonstration with sample texts."""
    print("\n" + "=" * 80)
    print("TOXICITY DETECTION SYSTEM - DEMO MODE")
    print("=" * 80)
    
    # Sample texts showing various types of toxicity
    test_cases = [
        {
            'text': "You are a wonderful person!",
            'description': "Clearly non-toxic"
        },
        {
            'text': "I hate everything about this.",
            'description': "Direct toxic keyword"
        },
        {
            'text': "You should go away and never come back.",
            'description': "Context-based toxicity"
        },
        {
            'text': "Yeah right, you're so smart.",
            'description': "Sarcastic toxicity"
        },
        {
            'text': "Oh how wonderful, nobody likes you.",
            'description': "Sarcasm + toxic pattern"
        },
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'=' * 80}")
        print(f"TEST CASE {i}: {test_case['description']}")
        print(f"{'=' * 80}")
        print(f"Text: \"{test_case['text']}\"")
        run_combined_detection(test_case['text'])
        print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Toxicity Detection System - Detect toxic content with context awareness"
    )
    
    parser.add_argument(
        '--text',
        type=str,
        help='Text to analyze for toxicity'
    )
    
    parser.add_argument(
        '--method',
        choices=['rule', 'ml', 'both'],
        default='both',
        help='Detection method: rule-based, ML-based, or both (default: both)'
    )
    
    parser.add_argument(
        '--train',
        action='store_true',
        help='Train a new ML model'
    )
    
    parser.add_argument(
        '--data',
        type=str,
        help='Path to training data CSV (for --train)'
    )
    
    parser.add_argument(
        '--demo',
        action='store_true',
        help='Run demonstration with sample texts'
    )
    
    parser.add_argument(
        '--create-dataset',
        action='store_true',
        help='Create a sample dataset'
    )
    
    args = parser.parse_args()
    
    # Handle different commands
    if args.create_dataset:
        print("Creating sample dataset...")
        create_sample_dataset(num_samples=100)
        print("Sample dataset created at: data/sample_dataset.csv")
        return
    
    if args.train:
        train_model(args.data)
        return
    
    if args.demo:
        demo_mode()
        return
    
    if args.text:
        if args.method == 'rule':
            run_rule_based_detection(args.text)
        elif args.method == 'ml':
            run_ml_detection(args.text)
        else:
            run_combined_detection(args.text)
    else:
        # If no arguments, show help
        parser.print_help()
        print("\n" + "=" * 80)
        print("QUICK START EXAMPLES:")
        print("=" * 80)
        print("\n1. Create sample dataset:")
        print("   python main.py --create-dataset")
        print("\n2. Train ML model:")
        print("   python main.py --train")
        print("\n3. Analyze text:")
        print("   python main.py --text \"Your text here\"")
        print("\n4. Run demo:")
        print("   python main.py --demo")
        print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
