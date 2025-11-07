"""
Rule-Based Toxicity Detector
Detects toxic content using pattern matching and context-aware rules.
Focuses on detecting texts that appear non-toxic but can be sinister with context.
"""

import re
import yaml
from typing import List, Dict, Tuple
from pathlib import Path


class RuleBasedDetector:
    """
    A rule-based toxicity detector that uses keyword matching and context analysis
    to identify potentially toxic or harmful content.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the detector with configuration.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.toxic_keywords = self.config.get('rule_based', {}).get('toxic_keywords', [])
        self.context_patterns = self.config.get('rule_based', {}).get('context_patterns', [])
        self.severity_weights = self.config.get('rule_based', {}).get('severity_weights', {})
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        config_file = Path(config_path)
        if config_file.exists():
            with open(config_file, 'r') as f:
                return yaml.safe_load(f)
        return {}
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for analysis.
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text
        """
        # Convert to lowercase
        text = text.lower()
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
    
    def detect_toxic_keywords(self, text: str) -> List[str]:
        """
        Detect toxic keywords in text.
        
        Args:
            text: Input text
            
        Returns:
            List of detected toxic keywords
        """
        text = self.preprocess_text(text)
        detected = []
        
        for keyword in self.toxic_keywords:
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
            if re.search(pattern, text):
                detected.append(keyword)
        
        return detected
    
    def detect_context_patterns(self, text: str) -> List[List[str]]:
        """
        Detect context-aware toxic patterns.
        These are combinations of words that become toxic in context.
        
        Args:
            text: Input text
            
        Returns:
            List of detected patterns
        """
        text = self.preprocess_text(text)
        detected_patterns = []
        
        for pattern in self.context_patterns:
            # Check if all words in pattern appear in text (in order)
            pattern_lower = [word.lower() for word in pattern]
            
            # Create regex for flexible matching (allows words in between)
            pattern_regex = r'\b' + r'\b.*\b'.join([re.escape(word) for word in pattern_lower]) + r'\b'
            
            if re.search(pattern_regex, text):
                detected_patterns.append(pattern)
        
        return detected_patterns
    
    def analyze_sarcasm_indicators(self, text: str) -> bool:
        """
        Detect potential sarcasm indicators that might hide toxicity.
        
        Args:
            text: Input text
            
        Returns:
            True if sarcasm indicators detected
        """
        sarcasm_markers = [
            r'yeah right\b',
            r'\bsure\b.*\b(you|it)\b',
            r'\boh.*\b(great|wonderful|perfect)\b',
            r'\bhow.*\b(nice|lovely|wonderful)\b',
            r'totally\b',
            r'absolutely\b.*\bnot\b'
        ]
        
        text = self.preprocess_text(text)
        
        for marker in sarcasm_markers:
            if re.search(marker, text, re.IGNORECASE):
                return True
        
        return False
    
    def calculate_toxicity_score(self, text: str) -> Tuple[float, Dict]:
        """
        Calculate overall toxicity score for text.
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (toxicity_score, details_dict)
        """
        details = {
            'toxic_keywords': [],
            'context_patterns': [],
            'has_sarcasm': False,
            'severity': 'low'
        }
        
        # Detect toxic keywords
        toxic_keywords = self.detect_toxic_keywords(text)
        details['toxic_keywords'] = toxic_keywords
        
        # Detect context patterns
        context_patterns = self.detect_context_patterns(text)
        details['context_patterns'] = context_patterns
        
        # Detect sarcasm
        has_sarcasm = self.analyze_sarcasm_indicators(text)
        details['has_sarcasm'] = has_sarcasm
        
        # Calculate score
        score = 0.0
        
        # Add points for keywords
        if toxic_keywords:
            score += len(toxic_keywords) * 0.3
        
        # Add points for context patterns (weighted more heavily)
        if context_patterns:
            score += len(context_patterns) * 0.5
        
        # Add points for sarcasm (potential hidden toxicity)
        if has_sarcasm and (toxic_keywords or context_patterns):
            score += 0.3
        
        # Normalize score to 0-1 range
        score = min(score, 1.0)
        
        # Determine severity
        if score >= 0.7:
            details['severity'] = 'high'
        elif score >= 0.4:
            details['severity'] = 'medium'
        else:
            details['severity'] = 'low'
        
        return score, details
    
    def is_toxic(self, text: str, threshold: float = 0.4) -> bool:
        """
        Determine if text is toxic based on threshold.
        
        Args:
            text: Input text
            threshold: Toxicity threshold (0-1)
            
        Returns:
            True if text is considered toxic
        """
        score, _ = self.calculate_toxicity_score(text)
        return score >= threshold
    
    def analyze(self, text: str) -> Dict:
        """
        Perform complete analysis of text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with analysis results
        """
        score, details = self.calculate_toxicity_score(text)
        
        return {
            'text': text,
            'toxicity_score': score,
            'is_toxic': score >= 0.4,
            'details': details
        }


def main():
    """Demo usage of the rule-based detector."""
    detector = RuleBasedDetector()
    
    # Test cases including context-aware toxicity
    test_texts = [
        "You are a wonderful person!",
        "I hate everything about this.",
        "You should go away and never come back.",
        "Yeah right, you're so smart.",  # Sarcastic
        "Oh how wonderful, nobody likes you.",  # Sarcastic + toxic
        "Have a great day!",
        "You stupid idiot, go die.",
        "I think you should reconsider your approach.",
    ]
    
    print("=" * 80)
    print("RULE-BASED TOXICITY DETECTION DEMO")
    print("=" * 80)
    
    for text in test_texts:
        result = detector.analyze(text)
        print(f"\nText: {result['text']}")
        print(f"Toxic: {result['is_toxic']}")
        print(f"Score: {result['toxicity_score']:.3f}")
        print(f"Severity: {result['details']['severity']}")
        if result['details']['toxic_keywords']:
            print(f"Toxic Keywords: {result['details']['toxic_keywords']}")
        if result['details']['context_patterns']:
            print(f"Context Patterns: {result['details']['context_patterns']}")
        if result['details']['has_sarcasm']:
            print("Sarcasm detected!")
        print("-" * 80)


if __name__ == "__main__":
    main()
