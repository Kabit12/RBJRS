"""
Job Classifier Module
======================
Classifies job descriptions into the same category taxonomy as resumes.

This ensures that both resumes and jobs use the same classification scheme,
enabling category-based matching in the recommendation engine.

Design Decision:
- Reuses the same TF-IDF vectorizer and classification approach as the resume classifier.
- When no dedicated job classifier model exists, it falls back to using
  the resume classifier (since both use similar text domains).
- Job text = title + description + requirements + responsibilities
"""

import os
import joblib
from app.ml.resume_classifier import ResumeClassifier


class JobClassifier:
    """
    Classifies job descriptions into the same categories as resumes.

    Falls back to the resume classifier if no dedicated job model exists,
    since the category taxonomy is shared.

    Usage:
        classifier = JobClassifier()
        classifier.load_model('ml_models/')
        category, confidence = classifier.predict("Senior Python Developer...")
    """

    def __init__(self):
        self.vectorizer = None
        self.classifier = None
        self.is_loaded = False
        self._resume_classifier = None

    def load_model(self, model_dir):
        """
        Load job classifier model. Falls back to resume classifier.

        Args:
            model_dir: Directory containing model files.

        Returns:
            True if loaded successfully.
        """
        # Try loading a dedicated job classifier first
        job_vec_path = os.path.join(model_dir, 'job_tfidf_vectorizer.pkl')
        job_clf_path = os.path.join(model_dir, 'job_classifier.pkl')

        if os.path.exists(job_vec_path) and os.path.exists(job_clf_path):
            try:
                self.vectorizer = joblib.load(job_vec_path)
                self.classifier = joblib.load(job_clf_path)
                self.is_loaded = True
                return True
            except Exception as e:
                print(f'Error loading job classifier: {e}')

        # Fall back to resume classifier
        self._resume_classifier = ResumeClassifier()
        if self._resume_classifier.load_model(model_dir):
            self.is_loaded = True
            return True

        return False

    def predict(self, text):
        """
        Predict the job category for a job description.

        Args:
            text: Combined job text (title + description + requirements).

        Returns:
            Tuple of (category: str, confidence: float)
        """
        if not self.is_loaded:
            return 'Unknown', 0.0

        # Use dedicated model if available, else fall back to resume classifier
        if self.classifier and self.vectorizer:
            try:
                features = self.vectorizer.transform([text])
                category = self.classifier.predict(features)[0]
                if hasattr(self.classifier, 'predict_proba'):
                    confidence = float(max(self.classifier.predict_proba(features)[0]))
                else:
                    confidence = 0.8
                return category, confidence
            except Exception as e:
                print(f'Job prediction error: {e}')
                return 'Unknown', 0.0
        elif self._resume_classifier:
            return self._resume_classifier.predict(text)

        return 'Unknown', 0.0


# Singleton instance
job_classifier = JobClassifier()
