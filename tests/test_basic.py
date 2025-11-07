"""
Basic tests for toxicity detection system
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rule_based_detector import RuleBasedDetector
from src.utils import clean_text, preprocess_text, expand_contractions


def test_rule_based_detector():
    """Test rule-based detector basic functionality."""
    print("Testing Rule-Based Detector...")
    
    detector = RuleBasedDetector()
    
    # Test non-toxic text
    result = detector.analyze("Have a great day!")
    assert result['is_toxic'] == False, "Non-toxic text marked as toxic"
    print("✓ Non-toxic text detection passed")
    
    # Test toxic text
    result = detector.analyze("I hate you stupid idiot")
    assert result['is_toxic'] == True, "Toxic text not detected"
    print("✓ Toxic text detection passed")
    
    # Test toxicity score calculation
    score, details = detector.calculate_toxicity_score("You are wonderful")
    assert score >= 0 and score <= 1, "Invalid toxicity score"
    print("✓ Toxicity score calculation passed")
    
    print("Rule-Based Detector: ALL TESTS PASSED ✓\n")


def test_text_preprocessing():
    """Test text preprocessing utilities."""
    print("Testing Text Preprocessing...")
    
    # Test clean_text
    text = "Check this http://example.com and email@test.com"
    cleaned = clean_text(text)
    assert "http://" not in cleaned, "URL not removed"
    assert "@" not in cleaned, "Email not removed"
    print("✓ URL and email removal passed")
    
    # Test expand_contractions
    text = "I can't believe you won't help"
    expanded = expand_contractions(text)
    assert "cannot" in expanded or "can not" in expanded, "Contraction not expanded"
    print("✓ Contraction expansion passed")
    
    # Test preprocess_text
    text = "  I can't   believe this!  http://test.com  "
    processed = preprocess_text(text)
    assert len(processed) < len(text), "Text not preprocessed"
    print("✓ Full preprocessing passed")
    
    print("Text Preprocessing: ALL TESTS PASSED ✓\n")


def test_detector_context_patterns():
    """Test context pattern detection."""
    print("Testing Context Pattern Detection...")
    
    detector = RuleBasedDetector()
    
    # Test pattern detection
    text = "you should go die"
    patterns = detector.detect_context_patterns(text)
    assert len(patterns) > 0, "Context pattern not detected"
    print("✓ Context pattern detection passed")
    
    # Test sarcasm detection
    text = "Yeah right, you're so smart"
    has_sarcasm = detector.analyze_sarcasm_indicators(text)
    assert has_sarcasm == True, "Sarcasm not detected"
    print("✓ Sarcasm detection passed")
    
    print("Context Pattern Detection: ALL TESTS PASSED ✓\n")


def run_all_tests():
    """Run all tests."""
    print("=" * 80)
    print("RUNNING TOXICITY DETECTION SYSTEM TESTS")
    print("=" * 80 + "\n")
    
    try:
        test_text_preprocessing()
        test_rule_based_detector()
        test_detector_context_patterns()
        
        print("=" * 80)
        print("ALL TESTS PASSED SUCCESSFULLY! ✓")
        print("=" * 80)
        return True
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        print("=" * 80)
        return False
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        print("=" * 80)
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
