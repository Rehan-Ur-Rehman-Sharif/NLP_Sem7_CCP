# Data Directory

This directory contains datasets for training and testing toxicity detection models.

## Structure

```
data/
├── raw/              # Raw, unprocessed datasets
├── processed/        # Preprocessed datasets ready for training
└── sample_dataset.csv # Sample dataset for quick testing
```

## Creating Sample Dataset

To create a sample dataset for testing:

```bash
python main.py --create-dataset
```

Or use the utility directly:

```bash
python src/utils.py
```

## Dataset Format

Datasets should be in CSV format with the following columns:

- `text`: The text content to analyze
- `label`: Binary label (0 = non-toxic, 1 = toxic)

Example:

```csv
text,label
"Have a great day!",0
"I hate you",1
"You are wonderful",0
```

## Data Files

Dataset CSV files are excluded from git via `.gitignore` to avoid committing large data files.
