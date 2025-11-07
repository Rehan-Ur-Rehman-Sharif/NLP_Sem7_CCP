# NLP Toxicity Detection System

A comprehensive tool for toxicity evaluation based on context, featuring both rule-based implementation and machine learning approaches to detect toxicity instances from text.

## 🎯 Problem Statement

This project evaluates the intent behind texts that appear non-toxic but, given appropriate context and proper analysis, can be sinister. The system provides:

- **Context-aware detection**: Identifies toxicity that depends on surrounding context
- **Sarcasm detection**: Recognizes sarcastic statements that hide toxic intent
- **Pattern matching**: Detects multi-word patterns that become toxic in combination
- **Machine learning**: Trains models on datasets for improved accuracy

## 📁 Directory Structure

```
NLP_Sem7_CCP/
├── config/              # Configuration files
│   └── config.yaml      # Main configuration for detection settings
├── data/                # Dataset storage (properly labeled and managed)
│   ├── raw/            # Raw, unprocessed datasets
│   ├── processed/      # Preprocessed datasets
│   └── sample_dataset.csv  # Sample data for testing
├── models/              # Trained ML models
│   ├── toxicity_model.joblib    # Trained model
│   └── vectorizer.joblib        # Feature vectorizer
├── src/                 # Source code
│   ├── rule_based_detector.py   # Rule-based toxicity detection
│   ├── model_trainer.py         # ML model training
│   ├── toxicity_classifier.py   # ML-based prediction
│   └── utils.py                 # Data preprocessing utilities
├── tests/               # Test files
├── main.py             # Main entry point
└── requirements.txt    # Python dependencies
```

## 🚀 Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Rehan-Ur-Rehman-Sharif/NLP_Sem7_CCP.git
cd NLP_Sem7_CCP
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Usage

#### 1. Create Sample Dataset
```bash
python main.py --create-dataset
```

#### 2. Train ML Model
```bash
python main.py --train
```

#### 3. Analyze Text (Combined Detection)
```bash
python main.py --text "Your text here"
```

#### 4. Run Demo
```bash
python main.py --demo
```

## 🔍 Detection Methods

### 1. Rule-Based Detection

Uses pattern matching and linguistic rules to detect toxicity:

- **Toxic keywords**: Identifies offensive words and phrases
- **Context patterns**: Detects word combinations that are toxic together
- **Sarcasm indicators**: Recognizes sarcastic statements that hide toxicity
- **Severity scoring**: Calculates toxicity scores (low/medium/high)

Example:
```python
from src.rule_based_detector import RuleBasedDetector

detector = RuleBasedDetector()
result = detector.analyze("Oh great, you messed up again")

print(f"Toxic: {result['is_toxic']}")
print(f"Score: {result['toxicity_score']}")
print(f"Has sarcasm: {result['details']['has_sarcasm']}")
```

### 2. ML-Based Detection

Trains machine learning models on labeled datasets:

- **Feature extraction**: TF-IDF or Count Vectorization with n-grams
- **Multiple algorithms**: Logistic Regression, Naive Bayes, Random Forest, SVM
- **Context-aware**: Analyzes surrounding text for better predictions
- **Probability scores**: Returns confidence levels for predictions

Example:
```python
from src.toxicity_classifier import ToxicityClassifier

classifier = ToxicityClassifier()
result = classifier.analyze_with_context(
    text="You're really something",
    context=["You always mess things up", "I can't believe you"]
)

print(f"Toxic: {result['is_toxic']}")
print(f"Probability: {result['toxicity_probability']}")
```

### 3. Combined Detection

Uses both methods for higher accuracy and confidence:

```bash
python main.py --text "Your text" --method both
```

## 📊 Model Training

### Using Custom Dataset

Prepare a CSV file with `text` and `label` columns:

```csv
text,label
"Have a great day!",0
"I hate you",1
```

Train the model:
```bash
python main.py --train --data path/to/your/dataset.csv
```

### Configuration

Modify `config/config.yaml` to customize:

- Detection thresholds
- Toxic keywords and patterns
- ML model parameters
- Feature extraction settings

## 🎓 Features

### Context-Aware Analysis

The system understands that toxicity can depend on context:

```python
# Text alone: "You're really something"
# Might seem neutral, but with toxic context...

classifier.analyze_with_context(
    text="You're really something",
    context=["You always fail", "Nobody likes you"]
)
# Result: Adjusted toxicity score based on context
```

### Sarcasm Detection

Identifies sarcastic statements that hide toxicity:

- "Yeah right, you're so smart" → Detected as sarcastic + toxic
- "Oh how wonderful, you messed up again" → Sarcasm indicator

### Pattern Matching

Detects multi-word patterns:

- ["you", "should", "die"] → Toxic pattern
- ["nobody", "likes", "you"] → Toxic pattern

## 📝 Command-Line Options

```bash
# Analyze text with different methods
python main.py --text "text" --method rule    # Rule-based only
python main.py --text "text" --method ml      # ML-based only
python main.py --text "text" --method both    # Combined (default)

# Training and setup
python main.py --create-dataset               # Create sample data
python main.py --train                        # Train with sample data
python main.py --train --data custom.csv      # Train with custom data

# Demo mode
python main.py --demo                         # Run demonstration
```

## 🧪 Testing

Run individual components:

```bash
# Test rule-based detector
python src/rule_based_detector.py

# Test ML classifier
python src/toxicity_classifier.py

# Test data utilities
python src/utils.py
```

## 🔧 Configuration

Edit `config/config.yaml` to customize:

```yaml
rule_based:
  toxic_keywords:
    - "hate"
    - "stupid"
  severity_weights:
    high: 1.0
    medium: 0.6
    low: 0.3

ml_model:
  model_type: 'logistic_regression'
  vectorizer: 'tfidf'
  max_features: 5000
  ngram_range: [1, 3]
```

## 📋 Requirements

- Python 3.7+
- numpy
- pandas
- scikit-learn
- nltk
- pyyaml
- Other dependencies in requirements.txt

## 🎯 Use Cases

1. **Social Media Moderation**: Detect toxic comments and posts
2. **Content Filtering**: Filter harmful content from user-generated text
3. **Chatbot Safety**: Prevent chatbots from responding to toxic inputs
4. **Research**: Study toxicity patterns and context-dependent toxicity
5. **Educational Tools**: Teach about online safety and toxic behavior

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues.

## 📄 License

This project is for educational purposes as part of NLP coursework.

## 👥 Authors

- Rehan Ur Rehman Sharif

## 🙏 Acknowledgments

- Built for NLP Semester 7 CCP project
- Focuses on context-aware toxicity detection
- Implements both traditional and ML approaches
