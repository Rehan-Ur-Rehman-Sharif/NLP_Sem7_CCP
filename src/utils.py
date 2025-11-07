"""
Data Preprocessing Utilities
Functions for preparing and processing text data for toxicity detection.
"""

import re
import pandas as pd
import numpy as np
from typing import List, Dict
from pathlib import Path


def clean_text(text: str) -> str:
    """
    Clean and normalize text.
    
    Args:
        text: Input text
        
    Returns:
        Cleaned text
    """
    if not isinstance(text, str):
        return ""
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text


def expand_contractions(text: str) -> str:
    """
    Expand common English contractions.
    
    Args:
        text: Input text
        
    Returns:
        Text with expanded contractions
    """
    contractions = {
        "ain't": "am not",
        "aren't": "are not",
        "can't": "cannot",
        "can't've": "cannot have",
        "could've": "could have",
        "couldn't": "could not",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "hadn't": "had not",
        "hasn't": "has not",
        "haven't": "have not",
        "he'd": "he would",
        "he'll": "he will",
        "he's": "he is",
        "I'd": "I would",
        "I'll": "I will",
        "I'm": "I am",
        "I've": "I have",
        "isn't": "is not",
        "it'd": "it would",
        "it'll": "it will",
        "it's": "it is",
        "let's": "let us",
        "shouldn't": "should not",
        "that's": "that is",
        "there's": "there is",
        "they'd": "they would",
        "they'll": "they will",
        "they're": "they are",
        "they've": "they have",
        "wasn't": "was not",
        "we'd": "we would",
        "we'll": "we will",
        "we're": "we are",
        "we've": "we have",
        "weren't": "were not",
        "won't": "will not",
        "wouldn't": "would not",
        "you'd": "you would",
        "you'll": "you will",
        "you're": "you are",
        "you've": "you have"
    }
    
    for contraction, expansion in contractions.items():
        text = re.sub(r'\b' + contraction + r'\b', expansion, text, flags=re.IGNORECASE)
    
    return text


def remove_special_characters(text: str, keep_punctuation: bool = True) -> str:
    """
    Remove special characters from text.
    
    Args:
        text: Input text
        keep_punctuation: Whether to keep basic punctuation
        
    Returns:
        Text with special characters removed
    """
    if keep_punctuation:
        # Keep letters, numbers, and basic punctuation
        text = re.sub(r'[^a-zA-Z0-9\s.,!?;:\'\"-]', '', text)
    else:
        # Keep only letters, numbers, and spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    return text


def preprocess_text(text: str, 
                   clean: bool = True,
                   expand_contract: bool = True,
                   remove_special: bool = False) -> str:
    """
    Complete text preprocessing pipeline.
    
    Args:
        text: Input text
        clean: Whether to clean text (URLs, emails, etc.)
        expand_contract: Whether to expand contractions
        remove_special: Whether to remove special characters
        
    Returns:
        Preprocessed text
    """
    if clean:
        text = clean_text(text)
    
    if expand_contract:
        text = expand_contractions(text)
    
    if remove_special:
        text = remove_special_characters(text, keep_punctuation=True)
    
    return text


def create_sample_dataset(output_path: str = "data/sample_dataset.csv", 
                         num_samples: int = 100) -> None:
    """
    Create a sample dataset for training.
    
    Args:
        output_path: Path to save the dataset
        num_samples: Number of samples to generate
    """
    # Sample toxic and non-toxic texts
    toxic_texts = [
        "I hate you so much",
        "You are stupid and worthless",
        "Go die in a hole",
        "Nobody likes you, loser",
        "You're an idiot",
        "Shut up, you fool",
        "I wish you were dead",
        "You disgust me",
        "Kill yourself",
        "You're pathetic and useless",
        "What a complete moron",
        "I hope bad things happen to you",
        "You make me sick",
        "Drop dead",
        "You're garbage",
        "Oh great, another stupid comment from you",
        "Yeah right, like you know anything",
        "How wonderful, you messed up again",
        "You should just give up",
        "Nobody wants you here",
        "You're a terrible person",
        "I can't stand you",
        "Get lost, idiot",
        "You're so dumb it hurts",
        "What an absolute waste",
    ]
    
    non_toxic_texts = [
        "Have a great day!",
        "You did an excellent job!",
        "Thank you for your help",
        "I appreciate your effort",
        "That's a good point",
        "Well done!",
        "I agree with you",
        "This is interesting",
        "Thanks for sharing",
        "I learned something new today",
        "You're very talented",
        "Keep up the good work",
        "That's helpful, thank you",
        "I understand your perspective",
        "Great suggestion!",
        "This is really useful",
        "I'm glad to hear that",
        "You make a valid point",
        "That's a creative idea",
        "I respect your opinion",
        "This is well explained",
        "Thank you for clarifying",
        "I see what you mean",
        "That's informative",
        "I appreciate your input",
    ]
    
    # Create balanced dataset
    data = []
    
    # Add toxic samples
    toxic_count = min(num_samples // 2, len(toxic_texts))
    for i in range(toxic_count):
        data.append({
            'text': toxic_texts[i % len(toxic_texts)],
            'label': 1
        })
    
    # Add non-toxic samples
    non_toxic_count = num_samples - toxic_count
    for i in range(non_toxic_count):
        data.append({
            'text': non_toxic_texts[i % len(non_toxic_texts)],
            'label': 0
        })
    
    # Create DataFrame and shuffle
    df = pd.DataFrame(data)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save to CSV
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"Sample dataset created: {output_path}")
    print(f"Total samples: {len(df)}")
    print(f"Toxic: {sum(df['label'] == 1)}, Non-toxic: {sum(df['label'] == 0)}")


def load_and_preprocess_dataset(data_path: str, 
                                preprocess: bool = True) -> pd.DataFrame:
    """
    Load and preprocess a dataset.
    
    Args:
        data_path: Path to dataset CSV
        preprocess: Whether to preprocess text
        
    Returns:
        Preprocessed DataFrame
    """
    df = pd.read_csv(data_path)
    
    if preprocess and 'text' in df.columns:
        df['text'] = df['text'].apply(lambda x: preprocess_text(x))
    
    return df


def get_dataset_statistics(df: pd.DataFrame) -> Dict:
    """
    Get statistics about a dataset.
    
    Args:
        df: Dataset DataFrame
        
    Returns:
        Dictionary with statistics
    """
    stats = {
        'total_samples': len(df),
        'toxic_samples': sum(df['label'] == 1) if 'label' in df.columns else 0,
        'non_toxic_samples': sum(df['label'] == 0) if 'label' in df.columns else 0,
        'avg_text_length': df['text'].str.len().mean() if 'text' in df.columns else 0,
        'max_text_length': df['text'].str.len().max() if 'text' in df.columns else 0,
        'min_text_length': df['text'].str.len().min() if 'text' in df.columns else 0,
    }
    
    if 'label' in df.columns:
        stats['class_balance'] = stats['toxic_samples'] / stats['total_samples'] if stats['total_samples'] > 0 else 0
    
    return stats


def main():
    """Demo usage of preprocessing utilities."""
    # Create sample dataset
    print("Creating sample dataset...")
    create_sample_dataset(num_samples=100)
    
    # Load and display statistics
    print("\nDataset Statistics:")
    df = load_and_preprocess_dataset('data/sample_dataset.csv')
    stats = get_dataset_statistics(df)
    
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Demo text preprocessing
    print("\n" + "=" * 80)
    print("TEXT PREPROCESSING DEMO")
    print("=" * 80)
    
    sample_texts = [
        "I can't believe you'd do that! Visit http://example.com",
        "You're such an idiot!!!",
        "Contact me at email@example.com for more info"
    ]
    
    for text in sample_texts:
        print(f"\nOriginal: {text}")
        print(f"Cleaned: {preprocess_text(text)}")


if __name__ == "__main__":
    main()
