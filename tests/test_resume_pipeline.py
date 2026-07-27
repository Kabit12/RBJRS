"""
Resume Pipeline Tests
========================
Tests for the resume upload → parse → classify pipeline.

Tests the individual components of the pipeline:
1. ResumeParser: PDF/DOCX text extraction
2. TextPreprocessor: NLP text cleaning
3. ResumeClassifier: Category prediction
4. EmbeddingService: Vector encoding
5. RecommendationEngine: Score computation
"""

import os
import pytest
import numpy as np


class TestResumeParser:
    """Tests for the ResumeParser class."""

    def test_parse_unsupported_format(self):
        """Parser should reject unsupported file types."""
        from app.ml.resume_parser import resume_parser
        with pytest.raises(ValueError, match='Unsupported file type'):
            resume_parser.parse('test.txt')

    def test_normalize_whitespace(self):
        """Whitespace normalization should clean multi-spaces and newlines."""
        from app.ml.resume_parser import ResumeParser
        parser = ResumeParser()
        messy = '  Hello    World  \n\n\n\nNext    paragraph  '
        result = parser._normalize_whitespace(messy)
        assert '    ' not in result  # No quadruple spaces
        assert '\n\n\n' not in result  # No triple newlines
        assert result.startswith('Hello')  # Leading whitespace stripped

    def test_parse_nonexistent_pdf(self):
        """Parser should return empty string for non-existent PDF."""
        from app.ml.resume_parser import resume_parser
        result = resume_parser.parse('nonexistent_file.pdf')
        assert result == ''

    def test_parse_nonexistent_docx(self):
        """Parser should return empty string for non-existent DOCX."""
        from app.ml.resume_parser import resume_parser
        result = resume_parser.parse('nonexistent_file.docx')
        assert result == ''


class TestTextPreprocessor:
    """Tests for text preprocessing."""

    def test_preprocess_cleans_text(self):
        """Preprocessing should lowercase, strip, and clean text."""
        from app.ml.text_preprocessor import text_preprocessor
        result = text_preprocessor.preprocess(
            'PYTHON Developer with 5+ Years EXPERIENCE in Machine Learning'
        )
        # Should be lowercase and cleaned
        assert 'PYTHON' not in result
        assert len(result) > 10

    def test_preprocess_empty_text(self):
        """Preprocessing empty text should return empty string."""
        from app.ml.text_preprocessor import text_preprocessor
        result = text_preprocessor.preprocess('')
        assert result == '' or result.strip() == ''

    def test_preprocess_special_characters(self):
        """Preprocessing should handle special characters gracefully."""
        from app.ml.text_preprocessor import text_preprocessor
        result = text_preprocessor.preprocess('C++ | C# | .NET | React.js')
        # Should not crash
        assert isinstance(result, str)


class TestResumeClassifier:
    """Tests for the resume classifier."""

    def test_predict_without_model(self):
        """Prediction without loaded model should return Unknown."""
        from app.ml.resume_classifier import ResumeClassifier
        classifier = ResumeClassifier()
        # Don't load any model
        category, confidence = classifier.predict('Python developer with ML experience')
        assert category == 'Unknown'
        assert confidence == 0.0

    def test_predict_empty_text(self):
        """Prediction on empty text should return Unknown."""
        from app.ml.resume_classifier import ResumeClassifier
        classifier = ResumeClassifier()
        category, confidence = classifier.predict('')
        assert category == 'Unknown'

    def test_predict_short_text(self):
        """Prediction on very short text should return Unknown."""
        from app.ml.resume_classifier import ResumeClassifier
        classifier = ResumeClassifier()
        category, confidence = classifier.predict('hi')
        assert category == 'Unknown'

    def test_categories_list(self):
        """Classifier should have exactly 25 categories."""
        from app.ml.resume_classifier import ResumeClassifier
        assert len(ResumeClassifier.CATEGORIES) == 25

    def test_load_nonexistent_model(self):
        """Loading from nonexistent path should return False."""
        from app.ml.resume_classifier import ResumeClassifier
        classifier = ResumeClassifier()
        result = classifier.load_model('/nonexistent/path/to/models')
        assert result is False
        assert classifier.is_loaded is False


class TestEmbeddingService:
    """Tests for the embedding service (unit tests only, no model loading)."""

    def test_encode_empty_text(self):
        """Encoding empty text should return a zero vector."""
        from app.ml.embedding_service import EmbeddingService
        service = EmbeddingService()
        result = service.encode('')
        assert isinstance(result, np.ndarray)
        assert result.shape == (384,)
        assert np.allclose(result, 0)

    def test_encode_none_text(self):
        """Encoding None should return a zero vector."""
        from app.ml.embedding_service import EmbeddingService
        service = EmbeddingService()
        result = service.encode(None)
        assert isinstance(result, np.ndarray)
        assert result.shape == (384,)

    def test_encode_batch_empty(self):
        """Encoding empty list should return empty array."""
        from app.ml.embedding_service import EmbeddingService
        service = EmbeddingService()
        result = service.encode_batch([])
        assert isinstance(result, np.ndarray)
        assert result.shape == (0, 384)


class TestRecommendationEngine:
    """Tests for the recommendation engine scoring logic."""

    def test_skill_score_empty_jobs(self):
        """No required skills → neutral score."""
        from app.ml.recommendation_engine import RecommendationEngine
        engine = RecommendationEngine()
        score = engine._compute_skill_score(['python', 'java'], [])
        assert score == 0.5

    def test_skill_score_perfect_match(self):
        """All skills matched → score of 1.0."""
        from app.ml.recommendation_engine import RecommendationEngine
        engine = RecommendationEngine()
        score = engine._compute_skill_score(
            ['python', 'flask', 'sql'],
            ['python', 'flask', 'sql']
        )
        assert score == 1.0

    def test_skill_score_no_match(self):
        """No skills matched → low score."""
        from app.ml.recommendation_engine import RecommendationEngine
        engine = RecommendationEngine()
        score = engine._compute_skill_score(
            ['java', 'spring'],
            ['python', 'flask']
        )
        assert score < 0.3

    def test_skill_score_synonym_matching(self):
        """Synonym skills should be matched (e.g., reactjs → react)."""
        from app.ml.recommendation_engine import RecommendationEngine
        engine = RecommendationEngine()
        score = engine._compute_skill_score(
            ['reactjs', 'nodejs'],
            ['react', 'node.js']
        )
        assert score >= 0.85

    def test_education_score_no_data(self):
        """No education data → 0.4 (moderate)."""
        from app.ml.recommendation_engine import RecommendationEngine
        engine = RecommendationEngine()
        score = engine._compute_education_score([], '')
        assert score == 0.4

    def test_education_score_bachelors(self):
        """Bachelor's degree → 0.7."""
        from app.ml.recommendation_engine import RecommendationEngine
        engine = RecommendationEngine()
        score = engine._compute_education_score(
            [{'degree': "Bachelor's", 'field_of_study': 'CS'}], ''
        )
        assert score == 0.7

    def test_education_score_phd(self):
        """PhD → 1.0."""
        from app.ml.recommendation_engine import RecommendationEngine
        engine = RecommendationEngine()
        score = engine._compute_education_score(
            [{'degree': 'PhD', 'field_of_study': 'ML'}], ''
        )
        assert score == 1.0

    def test_experience_score_no_data(self):
        """No experience data → 0.4 (moderate)."""
        from app.ml.recommendation_engine import RecommendationEngine
        engine = RecommendationEngine()
        score = engine._compute_experience_score([], '')
        assert score == 0.4

    def test_experience_score_with_dates(self):
        """Experience with date range should compute years."""
        from app.ml.recommendation_engine import RecommendationEngine
        engine = RecommendationEngine()
        score = engine._compute_experience_score(
            [{'start_date': '2018', 'end_date': '2023', 'title': 'Dev'}],
            'mid-level'
        )
        # 5 years of experience for mid-level should score well
        assert score >= 0.7

    def test_generate_feedback(self):
        """Feedback should return non-empty strengths and improvements."""
        from app.ml.recommendation_engine import RecommendationEngine
        engine = RecommendationEngine()
        strengths, improvements = engine._generate_feedback(
            {'skills': ['python', 'flask'], 'category': 'Python Developer'},
            {'required_skills': ['python', 'flask', 'docker'], 'category': 'Python Developer'},
            skill_score=0.7, edu_score=0.7, exp_score=0.7,
        )
        assert len(strengths) > 0
        assert len(improvements) > 0

    def test_recommend_jobs_empty_input(self):
        """Empty inputs should return empty recommendations."""
        from app.ml.recommendation_engine import RecommendationEngine
        engine = RecommendationEngine()
        result = engine.recommend_jobs({'cleaned_text': ''}, [])
        assert result == []

    def test_rank_candidates_empty_input(self):
        """Empty inputs should return empty rankings."""
        from app.ml.recommendation_engine import RecommendationEngine
        engine = RecommendationEngine()
        result = engine.rank_candidates({'combined_text': ''}, [])
        assert result == []


class TestScoringUtils:
    """Tests for scoring utility functions."""

    def test_score_label_excellent(self):
        from app.ml.scoring import get_score_label
        assert get_score_label(0.90) == 'Excellent Match'

    def test_score_label_low(self):
        from app.ml.scoring import get_score_label
        assert get_score_label(0.15) == 'Low Match'

    def test_score_color_success(self):
        from app.ml.scoring import get_score_color
        assert get_score_color(0.80) == 'success'

    def test_score_color_danger(self):
        from app.ml.scoring import get_score_color
        assert get_score_color(0.20) == 'danger'

    def test_format_percentage(self):
        from app.ml.scoring import format_score_percentage
        assert format_score_percentage(0.852) == '85.2%'
        assert format_score_percentage(None) == 'N/A'

    def test_compute_overall_score(self):
        from app.ml.scoring import compute_overall_score
        score = compute_overall_score(1.0, 1.0, 1.0, 1.0)
        assert score == 1.0

        score_zero = compute_overall_score(0.0, 0.0, 0.0, 0.0)
        assert score_zero == 0.0


class TestJobIndex:
    """Tests for the FAISS job index."""

    def test_empty_index(self):
        """Empty index should return empty results."""
        from app.ml.job_index import JobIndex
        index = JobIndex()
        scores, ids = index.search(np.zeros(384, dtype=np.float32))
        assert scores == []
        assert ids == []

    def test_build_and_search(self):
        """Build index and search should return results."""
        from app.ml.job_index import JobIndex
        index = JobIndex(embedding_dim=384)

        # Create some fake embeddings
        np.random.seed(42)
        n_jobs = 5
        job_ids = list(range(1, n_jobs + 1))
        embeddings = np.random.randn(n_jobs, 384).astype(np.float32)
        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms

        index.build(job_ids, embeddings)
        assert index.is_built
        assert index.size == n_jobs

        # Search with a query
        query = embeddings[0]  # Should match itself
        scores, result_ids = index.search(query, top_k=3)
        assert len(result_ids) == 3
        assert result_ids[0] == 1  # First job should match itself

    def test_clear_index(self):
        """Clear should reset the index."""
        from app.ml.job_index import JobIndex
        index = JobIndex()
        index.build([1], np.random.randn(1, 384).astype(np.float32))
        assert index.is_built
        index.clear()
        assert not index.is_built
