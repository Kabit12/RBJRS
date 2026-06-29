"""
Text Preprocessor Module
==========================
NLP preprocessing pipeline for resume and job description text.

Pipeline Steps:
1. Lowercase normalization
2. URL and email removal
3. Special character cleaning
4. Tokenization (word-level)
5. Stopword removal (NLTK English stopwords + custom domain stopwords)
6. Lemmatization (WordNet Lemmatizer for root-word normalization)

Design Decisions:
- NLTK is used for tokenization, stopwords, and lemmatization because:
  1. It's lightweight compared to spaCy for this specific task
  2. WordNet lemmatizer produces cleaner results for technical terms
  3. The proposal lists NLTK as a required library
- Custom stopwords are added for job/resume-specific noise words
  (e.g., "resume", "curriculum", "vitae", "references")
- The preprocessor returns both a cleaned string and a list of tokens,
  as different downstream tasks need different formats.

This is the second step in the processing pipeline:
    Upload → Parse → [Preprocess] → Extract Features → Classify → Recommend
"""

import re
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Add project-local nltk_data directory to search path
_project_nltk = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'nltk_data')
if os.path.isdir(_project_nltk) and _project_nltk not in nltk.data.path:
    nltk.data.path.insert(0, _project_nltk)

# Download required NLTK data (one-time setup)
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)


# Custom stopwords for the resume/job domain
DOMAIN_STOPWORDS = {
    'resume', 'curriculum', 'vitae', 'cv', 'references', 'available',
    'upon', 'request', 'page', 'objective', 'summary', 'contact',
    'address', 'phone', 'email', 'name', 'date', 'birth',
    'nationality', 'gender', 'marital', 'status', 'dear', 'sir',
    'madam', 'sincerely', 'regards', 'thank', 'please', 'apply',
    'position', 'application', 'candidate', 'applicant'
}


class TextPreprocessor:
    """
    NLP text preprocessing pipeline for resumes and job descriptions.

    Usage:
        preprocessor = TextPreprocessor()
        cleaned_text = preprocessor.preprocess("Raw resume text here...")
        tokens = preprocessor.tokenize("Raw text...")
    """

    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english')) | DOMAIN_STOPWORDS

    def preprocess(self, text):
        """
        Full preprocessing pipeline: clean → tokenize → remove stopwords → lemmatize → rejoin.

        Args:
            text: Raw text string from resume or job description.

        Returns:
            Cleaned and preprocessed text string.
        """
        if not text:
            return ''

        # Step 1: Basic cleaning
        text = self._clean_text(text)

        # Step 2: Tokenize
        tokens = word_tokenize(text)

        # Step 3: Remove stopwords and short tokens
        tokens = [t for t in tokens if t not in self.stop_words and len(t) > 1]

        # Step 4: Lemmatize
        tokens = [self.lemmatizer.lemmatize(t) for t in tokens]

        return ' '.join(tokens)

    def tokenize(self, text):
        """
        Tokenize and clean text, returning a list of processed tokens.

        Args:
            text: Raw text string.

        Returns:
            List of cleaned, lemmatized tokens.
        """
        if not text:
            return []

        text = self._clean_text(text)
        tokens = word_tokenize(text)
        tokens = [t for t in tokens if t not in self.stop_words and len(t) > 1]
        tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
        return tokens

    def _clean_text(self, text):
        """
        Basic text cleaning operations.

        Steps:
        1. Convert to lowercase
        2. Remove URLs
        3. Remove email addresses
        4. Remove phone numbers
        5. Remove special characters (keep alphanumeric and spaces)
        6. Remove extra whitespace
        """
        # Lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)

        # Remove email addresses
        text = re.sub(r'\S+@\S+\.\S+', '', text)

        # Remove phone numbers (various formats)
        text = re.sub(r'[\+]?[\d\-\(\)\s]{7,15}', ' ', text)

        # Remove special characters but keep alphanumeric, spaces, and common punctuation
        text = re.sub(r'[^a-zA-Z0-9\s\.\+\#]', ' ', text)

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def extract_keywords(self, text, top_n=30):
        """
        Extract the most important keywords from text.
        Simple approach: tokenize, count, return top N by frequency.

        Args:
            text: Input text string.
            top_n: Number of top keywords to return.

        Returns:
            List of (keyword, count) tuples sorted by frequency.
        """
        tokens = self.tokenize(text)
        from collections import Counter
        freq = Counter(tokens)
        return freq.most_common(top_n)


# Singleton instance
text_preprocessor = TextPreprocessor()
