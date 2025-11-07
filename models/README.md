# Models Directory

This directory stores trained machine learning models for toxicity detection.

## Files

- `toxicity_model.joblib`: Trained ML model (created after training)
- `vectorizer.joblib`: Text vectorizer (TF-IDF or Count Vectorizer)

## Training Models

To train a new model, run:

```bash
python main.py --train
```

Or use the model trainer directly:

```bash
python src/model_trainer.py
```

## Model Files

Model files are automatically saved here after training and are excluded from git via `.gitignore`.
