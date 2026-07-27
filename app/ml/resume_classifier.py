"""
Resume Classifier Module
==========================
Classifies resumes into job domain categories using supervised ML.

Architecture:
- TF-IDF Vectorizer converts cleaned resume text into feature vectors
- SVM (Support Vector Machine) classifier with linear kernel for category prediction

Training Data:
- Currently trained on ~450 SYNTHETIC samples generated in train_classifier.py
- NOT trained on real-world resumes — accuracy on real data is unverified
- The model should be retrained on real data (e.g., Kaggle Resume Dataset)
  for production use

Categories (25):
    Data Science, HR, Advocate, Arts, Web Designing, Mechanical Engineer,
    Sales, Health and Fitness, Civil Engineer, Java Developer, Business Analyst,
    SAP Developer, Automation Testing, Electrical Engineering, Operations Manager,
    Python Developer, DevOps Engineer, Network Security Engineer, PMO, Database,
    Hadoop, ETL Developer, DotNet Developer, Blockchain, Testing

Design Decisions:
- SVM with linear kernel works well with high-dimensional TF-IDF features
- Confidence threshold: predictions below 0.35 confidence return 'Unknown'
  to avoid misleading low-confidence guesses
- TF-IDF max_features=5000 to balance vocabulary coverage vs. noise
- The model and vectorizer are saved as pickle files for fast loading

This is the fourth step:
    Upload → Parse → Preprocess → Extract → [Classify] → Recommend
"""

import os
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV


class ResumeClassifier:
    """
    Classifies resume text into job domain categories.

    Usage:
        classifier = ResumeClassifier()
        classifier.load_model('ml_models/')
        category, confidence = classifier.predict("Python developer with 5 years experience...")
    """

    # The 25 resume categories from the Kaggle dataset
    CATEGORIES = [
        'Advocate', 'Arts', 'Automation Testing', 'Blockchain',
        'Business Analyst', 'Civil Engineer', 'Data Science',
        'Database', 'DevOps Engineer', 'DotNet Developer',
        'ETL Developer', 'Electrical Engineering', 'HR',
        'Hadoop', 'Health and Fitness', 'Java Developer',
        'Mechanical Engineer', 'Network Security Engineer',
        'Operations Manager', 'PMO', 'Python Developer',
        'SAP Developer', 'Sales', 'Testing', 'Web Designing',
    ]

    def __init__(self):
        self.vectorizer = None
        self.classifier = None
        self.is_loaded = False

    def load_model(self, model_dir):
        """
        Load pre-trained model and vectorizer from disk.

        Args:
            model_dir: Directory containing the saved model files.

        Returns:
            True if loaded successfully, False otherwise.
        """
        try:
            vectorizer_path = os.path.join(model_dir, 'tfidf_vectorizer.pkl')
            classifier_path = os.path.join(model_dir, 'resume_classifier.pkl')

            if os.path.exists(vectorizer_path) and os.path.exists(classifier_path):
                self.vectorizer = joblib.load(vectorizer_path)
                self.classifier = joblib.load(classifier_path)
                self.is_loaded = True
                return True
            else:
                print(f'Model files not found in {model_dir}')
                return False
        except Exception as e:
            print(f'Error loading model: {e}')
            return False

    # Minimum confidence threshold — below this, return 'Unknown'
    # to avoid misleading low-confidence category assignments
    CONFIDENCE_THRESHOLD = 0.35

    def predict(self, text):
        """
        Predict the job category for a resume text.

        Args:
            text: Cleaned/preprocessed resume text.

        Returns:
            Tuple of (category: str, confidence: float)
            Returns ('Unknown', 0.0) if model is not loaded or confidence
            is below the threshold.
        """
        if not self.is_loaded:
            return 'Unknown', 0.0

        if not text or len(text.strip()) < 10:
            return 'Unknown', 0.0

        try:
            # Vectorize
            features = self.vectorizer.transform([text])

            # Predict
            category = self.classifier.predict(features)[0]

            # Get confidence (probability)
            if hasattr(self.classifier, 'predict_proba'):
                probabilities = self.classifier.predict_proba(features)[0]
                confidence = float(max(probabilities))
            elif hasattr(self.classifier, 'decision_function'):
                decisions = self.classifier.decision_function(features)[0]
                # Convert decision function to pseudo-probability using softmax
                exp_decisions = np.exp(decisions - np.max(decisions))
                probabilities = exp_decisions / exp_decisions.sum()
                confidence = float(max(probabilities))
            else:
                confidence = 0.8  # Default confidence

            # Reject low-confidence predictions
            if confidence < self.CONFIDENCE_THRESHOLD:
                return 'Unknown', confidence

            return category, confidence

        except Exception as e:
            print(f'Prediction error: {e}')
            return 'Unknown', 0.0

    def get_top_categories(self, text, top_n=5):
        """
        Get top N predicted categories with their confidence scores.

        Useful for showing the user alternative category matches.

        Args:
            text: Cleaned resume text.
            top_n: Number of top categories to return.

        Returns:
            List of (category, confidence) tuples sorted by confidence.
        """
        if not self.is_loaded or not text:
            return []

        try:
            features = self.vectorizer.transform([text])

            if hasattr(self.classifier, 'predict_proba'):
                probabilities = self.classifier.predict_proba(features)[0]
                classes = self.classifier.classes_
            elif hasattr(self.classifier, 'decision_function'):
                decisions = self.classifier.decision_function(features)[0]
                exp_decisions = np.exp(decisions - np.max(decisions))
                probabilities = exp_decisions / exp_decisions.sum()
                classes = self.classifier.classes_
            else:
                return [(self.classifier.predict(features)[0], 0.8)]

            # Sort by probability
            indices = np.argsort(probabilities)[::-1][:top_n]
            return [(classes[i], float(probabilities[i])) for i in indices]

        except Exception as e:
            print(f'Error getting top categories: {e}')
            return []

    @staticmethod
    def train_and_save(texts, labels, model_dir):
        """
        Train the classifier on labeled resume data and save the model.

        Args:
            texts: List of preprocessed resume text strings.
            labels: List of category labels.
            model_dir: Directory to save the trained model.

        Returns:
            Dictionary with training metrics.
        """
        os.makedirs(model_dir, exist_ok=True)

        # TF-IDF Vectorization
        vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),  # Unigrams + bigrams
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,  # Apply sublinear TF scaling (1 + log(tf))
        )
        features = vectorizer.fit_transform(texts)

        # Train SVM with calibration for probability estimates
        base_svm = LinearSVC(max_iter=10000, C=1.0, class_weight='balanced')
        classifier = CalibratedClassifierCV(base_svm, cv=3)
        classifier.fit(features, labels)

        # Save models
        joblib.dump(vectorizer, os.path.join(model_dir, 'tfidf_vectorizer.pkl'))
        joblib.dump(classifier, os.path.join(model_dir, 'resume_classifier.pkl'))

        # Compute accuracy on training data (for initial verification)
        train_accuracy = classifier.score(features, labels)

        return {
            'accuracy': train_accuracy,
            'n_samples': len(texts),
            'n_features': features.shape[1],
            'n_categories': len(set(labels)),
        }


# Singleton instance
resume_classifier = ResumeClassifier()
