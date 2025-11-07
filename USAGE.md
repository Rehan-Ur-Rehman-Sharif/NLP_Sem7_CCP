# Usage Examples

This document provides detailed examples of how to use the Toxicity Detection System.

## Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### 2. Create Sample Dataset

```bash
python main.py --create-dataset
```

This creates a sample dataset at `data/sample_dataset.csv` with 100 balanced samples.

### 3. Train ML Model

```bash
python main.py --train
```

This trains a machine learning model using the sample dataset and saves it to `models/`.

### 4. Run Demo

```bash
python main.py --demo
```

This runs demonstrations showing various types of toxicity detection.

## Analyzing Text

### Basic Analysis (Both Methods)

```bash
python main.py --text "You stupid idiot, I hate you"
```

Output:
```
================================================================================
COMBINED TOXICITY DETECTION
================================================================================

[1] Rule-Based Analysis:
--------------------------------------------------------------------------------
Toxic: True
Score: 0.900
Severity: high

[2] ML-Based Analysis:
--------------------------------------------------------------------------------
Toxic: True
Probability: 0.540
Severity: medium

[3] Combined Decision:
--------------------------------------------------------------------------------
Final Decision: TOXIC
Combined Score: 0.720
⚠ Both methods agree: HIGH CONFIDENCE
================================================================================
```

### Rule-Based Only

```bash
python main.py --text "Oh great, you messed up again" --method rule
```

### ML-Based Only

```bash
python main.py --text "I think you're wonderful" --method ml
```

## Using Python API

### Rule-Based Detection

```python
from src.rule_based_detector import RuleBasedDetector

# Initialize detector
detector = RuleBasedDetector()

# Analyze text
result = detector.analyze("Yeah right, you're so smart")

# Access results
print(f"Is toxic: {result['is_toxic']}")
print(f"Toxicity score: {result['toxicity_score']}")
print(f"Severity: {result['details']['severity']}")
print(f"Has sarcasm: {result['details']['has_sarcasm']}")

# Get just the toxicity score
score, details = detector.calculate_toxicity_score("Some text")
```

### ML-Based Detection

```python
from src.toxicity_classifier import ToxicityClassifier

# Initialize classifier (loads pre-trained model)
classifier = ToxicityClassifier()

# Simple prediction
is_toxic = classifier.predict("I hate everything")
probability = classifier.predict_proba("You are wonderful")

# Detailed analysis
result = classifier.analyze_with_context("Your text here")
print(f"Toxic: {result['is_toxic']}")
print(f"Probability: {result['toxicity_probability']}")
print(f"Severity: {result['severity']}")

# Batch prediction
texts = ["Text 1", "Text 2", "Text 3"]
results = classifier.batch_predict(texts)
```

### Context-Aware Analysis

```python
from src.toxicity_classifier import ToxicityClassifier

classifier = ToxicityClassifier()

# Analyze text with surrounding context
main_text = "You're really something"
context = [
    "You always mess things up",
    "Everyone knows you're incompetent",
    "I can't believe I have to work with you"
]

result = classifier.analyze_with_context(main_text, context)

print(f"Base probability: {result['toxicity_probability']}")
if 'adjusted_probability' in result:
    print(f"Context-adjusted: {result['adjusted_probability']}")
    print(f"Context toxicity: {result['context_toxicity']}")
```

### Training Custom Models

```python
from src.model_trainer import ToxicityModelTrainer

# Initialize trainer
trainer = ToxicityModelTrainer()

# Option 1: Use sample data
metrics = trainer.train_pipeline()

# Option 2: Use custom dataset
metrics = trainer.train_pipeline("path/to/your/data.csv")

# Access evaluation metrics
print(f"Accuracy: {metrics['accuracy']}")
print(f"F1 Score: {metrics['f1_score']}")
```

### Data Preprocessing

```python
from src.utils import (
    clean_text,
    preprocess_text,
    create_sample_dataset,
    load_and_preprocess_dataset,
    get_dataset_statistics
)

# Clean a single text
text = "Check this http://example.com and email@test.com"
cleaned = clean_text(text)

# Full preprocessing
processed = preprocess_text(text, clean=True, expand_contract=True)

# Create sample dataset
create_sample_dataset(output_path="data/my_dataset.csv", num_samples=200)

# Load and preprocess dataset
df = load_and_preprocess_dataset("data/sample_dataset.csv")

# Get statistics
stats = get_dataset_statistics(df)
print(f"Total samples: {stats['total_samples']}")
print(f"Toxic samples: {stats['toxic_samples']}")
```

## Custom Configuration

Edit `config/config.yaml` to customize detection behavior:

```yaml
rule_based:
  toxic_keywords:
    - "hate"
    - "stupid"
    - "idiot"
    # Add your own keywords
  
  context_patterns:
    - ["you", "should", "die"]
    # Add your own patterns
  
  severity_weights:
    high: 1.0
    medium: 0.6
    low: 0.3

ml_model:
  model_type: 'logistic_regression'  # or 'naive_bayes', 'random_forest', 'svm'
  vectorizer: 'tfidf'  # or 'count'
  max_features: 5000
  ngram_range: [1, 3]
```

## Training with Your Own Data

### 1. Prepare Your Dataset

Create a CSV file with two columns:
- `text`: The text content
- `label`: 0 for non-toxic, 1 for toxic

Example `my_data.csv`:
```csv
text,label
"Have a great day!",0
"I hate you",1
"You are wonderful",0
"You stupid idiot",1
```

### 2. Train Model

```bash
python main.py --train --data my_data.csv
```

Or use Python:

```python
from src.model_trainer import ToxicityModelTrainer

trainer = ToxicityModelTrainer()
metrics = trainer.train_pipeline("my_data.csv")
```

### 3. Use Trained Model

The trained model is automatically saved and loaded:

```bash
python main.py --text "Your text" --method ml
```

## Advanced Examples

### Detect Sarcastic Toxicity

```python
from src.rule_based_detector import RuleBasedDetector

detector = RuleBasedDetector()

texts = [
    "Oh great, another stupid mistake",
    "Yeah right, like you know anything",
    "How wonderful, you failed again"
]

for text in texts:
    result = detector.analyze(text)
    if result['details']['has_sarcasm']:
        print(f"Sarcastic toxicity detected: {text}")
        print(f"Score: {result['toxicity_score']}")
```

### Compare Detection Methods

```python
from src.rule_based_detector import RuleBasedDetector
from src.toxicity_classifier import ToxicityClassifier

rule_detector = RuleBasedDetector()
ml_classifier = ToxicityClassifier()

text = "You should reconsider your life choices"

# Rule-based
rule_result = rule_detector.analyze(text)
print(f"Rule-based: {rule_result['is_toxic']} ({rule_result['toxicity_score']})")

# ML-based
ml_result = ml_classifier.analyze_with_context(text)
print(f"ML-based: {ml_result['is_toxic']} ({ml_result['toxicity_probability']})")

# Combined decision
combined_toxic = rule_result['is_toxic'] or ml_result['is_toxic']
print(f"Combined: {combined_toxic}")
```

## Running Tests

```bash
# Run all tests
python tests/test_basic.py

# Or use pytest if installed
pytest tests/
```

## Common Issues

### Model Not Found

If you get "No pre-trained model found", you need to train a model first:

```bash
python main.py --create-dataset
python main.py --train
```

### Import Errors

Make sure all dependencies are installed:

```bash
pip install -r requirements.txt
```

### Low Model Accuracy

The sample dataset is small (100 samples) and meant for demonstration. For better accuracy:

1. Collect a larger, more diverse dataset
2. Ensure balanced classes (similar number of toxic and non-toxic samples)
3. Experiment with different model types in `config/config.yaml`
4. Adjust `max_features` and `ngram_range` parameters
