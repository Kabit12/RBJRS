"""
ML Package
============
Machine Learning and NLP components for the Resume-Based Job Recommendation System.

Modules:
    - resume_parser: PDF/DOCX text extraction
    - text_preprocessor: NLP text cleaning pipeline
    - feature_extractor: Skill/education/experience extraction
    - resume_classifier: Resume category classification (TF-IDF + SVM)
    - job_classifier: Job category classification
    - embedding_service: Sentence-transformer embeddings (NEW — replaces TF-IDF refit)
    - job_index: FAISS-based job vector index (NEW — replaces brute-force KNN)
    - recommendation_engine: Multi-signal matching (embedding + skills + education + experience)
    - scoring: Job Fit Score computation
"""
