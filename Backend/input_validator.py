# -*- coding: utf-8 -*-
"""
Input Validator - Multi-layer Defense System
Provides functions for input validation, including length, URL, SQL injection, profanity,
gibberish, and basic structural checks.
"""
import re
import logging
from typing import Tuple, Dict, Any, Optional

logger = logging.getLogger(__name__)

# Global variable for the semantic model, to be set by ModelWrapper
# This is a placeholder and will be accessed by the validator instance.
# The actual SentenceTransformer model instance will be passed/accessed dynamically.
global_semantic_model = None 

class InputValidator:
    """Input Validator - Multi-layer Defense System"""

    def __init__(self):
        """Initialize validator (including optional semantic model)"""
        # Semantic model is loaded lazily and accessed via global_semantic_model
        self.semantic_check_enabled = False
        self.ref_sentences = [
            "This is a normal sentence about politics.",
            "The unemployment rate has increased.",
            "Climate change is affecting the world.",
            "The government announced new policies.",
            "He made a statement about healthcare.",
            "Scientists have developed a new vaccine.",
            "The research shows promising results.",
            "Technology companies are investing in AI.",
            "Education reform will take several years.",
            "The economy is showing signs of recovery.",
            "COVID-19 vaccines were approved by regulators.",
            "Medical research has made significant progress.",
            "This is a normal statement about politics.",
            "The economic growth rate is slowing down.",
            "Scientists have discovered new treatments."
        ]
        self.ref_embeddings = None
        self.semantic_model = None # Placeholder for the actual model instance

    def set_semantic_model(self, model_instance, ref_embeddings):
        """Set the semantic model instance and pre-calculated embeddings."""
        self.semantic_model = model_instance
        self.ref_embeddings = ref_embeddings
        self.semantic_check_enabled = (model_instance is not None and ref_embeddings is not None)
        if self.semantic_check_enabled:
            logger.info("✓ Semantic model and reference embeddings initialized for InputValidator.")
        else:
            logger.warning("⚠️ Semantic model not set or embeddings are missing. Semantic check disabled.")

    def is_valid_length(self, text: str) -> Tuple[bool, str]:
        """Length check"""
        text = text.strip()
        if len(text) < 5:
            return False, "Input text is too short (minimum 5 characters)"
        if len(text) > 1000:
            return False, "Input text is too long (maximum 1000 characters)"
        return True, ""

    def contains_url(self, text: str) -> Tuple[bool, str]:
        """URL detection"""
        if re.search(r'(https?://|www\.|ftp://|mailto:|file://)', text, re.IGNORECASE):
            return False, "Input cannot contain URL links"
        if re.search(r'\b[a-z0-9]{8,}\.(com|net|org|cn|io)\b', text, re.IGNORECASE):
            return False, "Input appears to contain links"
        return True, ""

    def contains_sql_injection(self, text: str) -> Tuple[bool, str]:
        """SQL Injection detection"""
        sql_patterns = [
            r'\b(SELECT|DROP|INSERT|DELETE|UPDATE|UNION|ALTER|CREATE|TABLE|DATABASE|EXEC|EXECUTE)\b',
            r'(--|;|\/\*|\*\/|xp_|sp_)',
            r'(\bOR\b.*=.*|AND.*=.*|\d+\s*=\s*\d+)',
        ]
        for pattern in sql_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return False, "Input contains suspicious SQL code"
        return True, ""

    def contains_profanity(self, text: str) -> Tuple[bool, str]:
        """Profanity and inappropriate content detection"""
        en_profanity = [
            'fuck', 'shit', 'bitch', 'asshole', 'damn', 'crap',
            'bastard', 'dick', 'pussy', 'cock', 'slut', 'whore'
        ]
        cn_profanity_pinyin = [
            'cao', 'mabi', 'shabi', 'nima', 'caonima', 'wo cao',
            'tama', 'goushi', 'laji', 'feiwu', 'baichi', 'zhizhang'
        ]
        spam_words = [
            'advertisement', 'promotion', 'pornography', 'gambling', 'wechat', 'qq', 'vx',
            'add me', 'contact me', 'consultation', 'agency', 'certification', 'loan',
            'part-time', 'recruitment', 'make money', 'discount', 'offer', '💰', '🤑', '💵'
        ]

        text_lower = text.lower()
        for word in en_profanity:
            if word in text_lower:
                return False, "Input contains inappropriate language"
        
        # Check for pinyin-based profanity (simple check)
        for word in cn_profanity_pinyin:
            if word in text_lower:
                return False, "Input contains inappropriate language (pinyin)"

        for word in spam_words:
            if word in text_lower:
                return False, "Input appears to be advertising or marketing content"

        return True, ""

    def is_random_gibberish(self, text: str) -> Tuple[bool, str]:
        """Random gibberish detection"""
        text = text.strip()

        # Pure symbols or numbers
        if re.fullmatch(r'^[!@#$%^&*()_+=\-\d\s]+$', text):
            return False, "Input is meaningless (pure symbols or numbers)"

        # Excessive repetition
        if re.search(r'(.)\1{6,}', text):
            return False, "Input contains excessive repeated characters"

        # Keyboard smash
        keyboard_patterns = [
            'asdfghjkl', 'qwertyuiop', 'zxcvbnm',
            'qazwsxedc', 'plokimjunhybgtvfrcdxesz'
        ]
        text_clean = re.sub(r'\s+', '', text.lower())
        for pattern in keyboard_patterns:
            if pattern in text_clean or pattern[::-1] in text_clean:
                return False, "Input appears to be a keyboard smash"

        # Too few unique characters
        unique_chars = len(set(text.replace(' ', '')))
        if unique_chars < 3 and len(text) > 10:
            return False, "Input has too few unique character types"

        return True, ""

    def has_valid_structure(self, text: str) -> Tuple[bool, str]:
        """Basic grammatical structure check"""
        text = text.strip()

        # Must contain at least letters
        if not re.search(r'[a-zA-Z]', text):
            return False, "Input lacks valid text content"

        # Find words (allowing hyphens and numbers, e.g., COVID-19)
        words = re.findall(r'\b[a-zA-Z][-a-zA-Z0-9]*[a-zA-Z0-9]\b|\b[a-zA-Z]+\b', text)
        
        if len(words) < 2:
            logger.debug(f"Insufficient word count: Detected {len(words)} valid words, requires at least 2")
            logger.debug(f"Detected words: {words}")
            return False, f"Input is too brief or lacks complete words (Detected {len(words)} valid words)"

        # Excessive punctuation (excluding hyphens)
        punct_count = len(re.findall(r'[^\w\s\-]', text))
        if punct_count > len(text) * 0.5:
            return False, "Input contains excessive punctuation"

        return True, ""

    def is_semantically_meaningful(self, text: str) -> Tuple[bool, str]:
        """Semantic similarity check"""
        if not self.semantic_check_enabled or self.semantic_model is None or self.ref_embeddings is None:
            logger.debug("ℹ️ Semantic check is disabled or model is not initialized.")
            return True, ""

        try:
            from sentence_transformers import util
            
            text_embedding = self.semantic_model.encode(text, convert_to_tensor=True)
            cosine_scores = util.cos_sim(text_embedding, self.ref_embeddings)
            max_score = float(cosine_scores.max())

            logger.debug(f"Semantic similarity score: {max_score:.3f}")

            # Threshold set to 0.10 for high tolerance
            if max_score < 0.10:
                return False, f"Input does not resemble a meaningful statement (Similarity: {max_score:.3f})"

            return True, ""
        except Exception as e:
            logger.warning(f"Semantic check failed during runtime: {e}")
            return True, ""

    def validate(self, text: str) -> Tuple[bool, str]:
        """
        Comprehensive validation
        Returns: (is_valid, error_message)
        """
        checks = [
            self.is_valid_length(text),
            self.contains_url(text),
            self.contains_sql_injection(text),
            self.contains_profanity(text),
            self.is_random_gibberish(text),
            self.has_valid_structure(text),
        ]

        # Execute all rule checks
        for valid, msg in checks:
            if not valid:
                return False, msg

        # Semantic check (mandatory)
        valid, msg = self.is_semantically_meaningful(text)
        if not valid:
            return False, msg

        return True, ""

# Instantiate global validator
validator = InputValidator()

def is_safe_input(text: str) -> Tuple[bool, str]:
    """
    Validate if the input is safe
    Returns: (is_safe, error_message)
    """
    try:
        valid, msg = validator.validate(text)
        if not valid:
            logger.debug(f"Input rejected by validator: {msg} | Text: {text[:60]}...")
        return valid, msg
    except Exception as e:
        logger.warning(f"is_safe_input runtime exception: {e}")
        return False, f"Validation system error: {str(e)}"