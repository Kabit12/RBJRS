"""
Recommendation Engine Module (v2 — Embedding-Based)
=====================================================
Core matching engine that recommends jobs to candidates and ranks candidates for jobs.

REWRITTEN from the old TF-IDF refit-per-request approach to use:
1. Sentence-transformer embeddings (384-dim, computed ONCE per text)
2. FAISS index for fast nearest neighbor search (milliseconds, not seconds)
3. Multi-signal scoring: embedding similarity + skill matching + education + experience

Architecture Change (v1 → v2):
    v1: TF-IDF fit_transform() on EVERY request → inconsistent vectors, O(N²), slow
    v2: Pre-computed embeddings → FAISS search → structured signal scoring → fast, consistent

Score Formula:
    overall = 0.40 × embedding_similarity
            + 0.30 × skill_score
            + 0.15 × education_score
            + 0.15 × experience_score
            + category_bonus (0.10 if category matches)

Design Decisions:
- Embedding similarity gets the highest weight (0.40) because sentence-transformers
  capture semantic meaning far better than TF-IDF keyword overlap.
- Skill score weight (0.30) is still high because explicit skill matching
  is a strong signal that candidates and recruiters care about.
- Category bonus reduced from 0.15 to 0.10 because embeddings already capture
  domain similarity, making the explicit category bonus less necessary.
- The engine is split into focused components:
    - EmbeddingService: text → vector (embedding_service.py)
    - JobIndex: vector search (job_index.py)
    - RecommendationEngine: orchestrates scoring (this file)
"""

import re
import numpy as np
import logging

logger = logging.getLogger(__name__)


# Score component weights (must sum to 1.0)
WEIGHT_EMBEDDING = 0.40
WEIGHT_SKILL = 0.30
WEIGHT_EDUCATION = 0.15
WEIGHT_EXPERIENCE = 0.15

# Category bonus when resume and job share the same classified category
CATEGORY_BONUS = 0.10

# Education level hierarchy for scoring
EDUCATION_LEVELS = {
    'certificate': 1,
    'diploma': 2,
    'associate': 2,
    "bachelor's": 3,
    "bachelor": 3,
    "master's": 4,
    "master": 4,
    'doctorate': 5,
    'phd': 5,
    'other': 2,
}

# Experience level mapping (years)
EXPERIENCE_LEVELS = {
    'entry': (0, 2),
    'entry level': (0, 2),
    'junior': (0, 3),
    'fresher': (0, 1),
    'mid': (2, 5),
    'mid level': (2, 5),
    'mid-level': (2, 5),
    'senior': (5, 10),
    'lead': (7, 15),
    'principal': (10, 20),
    'staff': (8, 15),
    'director': (10, 20),
    'vp': (12, 25),
}

# Skill synonyms — maps variant names to a canonical form
SKILL_SYNONYMS = {
    'react.js': 'react', 'reactjs': 'react', 'react js': 'react',
    'vue.js': 'vue', 'vuejs': 'vue', 'vue js': 'vue',
    'node.js': 'nodejs', 'node js': 'nodejs',
    'next.js': 'nextjs', 'next js': 'nextjs',
    'angular.js': 'angular', 'angularjs': 'angular',
    'express.js': 'express', 'expressjs': 'express',
    'three.js': 'threejs',
    'c++': 'cpp', 'cplusplus': 'cpp',
    'c#': 'csharp', 'c sharp': 'csharp',
    'objective-c': 'objectivec', 'obj-c': 'objectivec',
    '.net': 'dotnet', 'dot net': 'dotnet', '.net core': 'dotnet',
    'asp.net': 'aspnet',
    'scikit-learn': 'sklearn', 'scikit learn': 'sklearn',
    'postgresql': 'postgres', 'postgre': 'postgres',
    'mongo': 'mongodb', 'mongo db': 'mongodb',
    'sql server': 'mssql', 'microsoft sql server': 'mssql',
    'amazon web services': 'aws',
    'google cloud platform': 'gcp', 'google cloud': 'gcp',
    'microsoft azure': 'azure',
    'machine learning': 'ml',
    'deep learning': 'dl',
    'natural language processing': 'nlp',
    'artificial intelligence': 'ai',
    'ci/cd': 'cicd', 'ci cd': 'cicd',
    'devops': 'devops', 'dev ops': 'devops',
    'html5': 'html', 'html 5': 'html',
    'css3': 'css', 'css 3': 'css',
    'javascript': 'js', 'java script': 'js',
    'typescript': 'ts', 'type script': 'ts',
    'ruby on rails': 'rails', 'ror': 'rails',
    'spring boot': 'springboot',
    'power bi': 'powerbi',
    'tailwind css': 'tailwind', 'tailwindcss': 'tailwind',
    'material ui': 'materialui', 'mui': 'materialui',
    # Added modern tech synonyms
    'langchain': 'langchain', 'lang chain': 'langchain',
    'react 18': 'react', 'react 19': 'react',
    'next.js 14': 'nextjs', 'next.js 15': 'nextjs',
    'python3': 'python', 'python 3': 'python',
    'go lang': 'golang', 'go language': 'golang',
}


def _normalize_skill(skill_name):
    """Normalize a skill name: lowercase, strip, and apply synonym mapping."""
    if not isinstance(skill_name, str):
        skill_name = skill_name.get('name', '') if isinstance(skill_name, dict) else str(skill_name)
    normalized = skill_name.lower().strip()
    return SKILL_SYNONYMS.get(normalized, normalized)


class RecommendationEngine:
    """
    Core recommendation engine for job-resume matching.

    Uses embedding similarity (from sentence-transformers) combined with
    structured signals (skills, education, experience) for scoring.

    Usage:
        engine = RecommendationEngine()
        recommendations = engine.recommend_jobs(resume_data, jobs_data, top_n=10)
        rankings = engine.rank_candidates(job_data, candidates_data, top_n=20)
    """

    def __init__(self):
        # Lazy imports to avoid circular dependencies and cold-start blocking
        self._embedding_service = None

    def _get_embedding_service(self):
        """Lazy-load the embedding service."""
        if self._embedding_service is None:
            from app.ml.embedding_service import embedding_service
            self._embedding_service = embedding_service
        return self._embedding_service

    def recommend_jobs(self, resume_data, jobs_data, top_n=10):
        """
        Recommend jobs for a candidate based on their resume.

        Args:
            resume_data: Dict with keys:
                - cleaned_text: Preprocessed resume text
                - embedding: Pre-computed embedding vector (optional)
                - skills: List of skill names
                - education: List of education entries
                - experience: List of experience entries
                - category: Classified resume category

            jobs_data: List of dicts, each with:
                - id: Job ID
                - combined_text: Job title + description + requirements
                - required_skills: List of skill names
                - experience_level: Required experience level
                - category: Classified job category

            top_n: Number of recommendations to return.

        Returns:
            List of recommendation dicts sorted by overall_score (descending).
        """
        if not jobs_data or not resume_data.get('cleaned_text'):
            return []

        try:
            embedding_svc = self._get_embedding_service()

            # Step 1: Get or compute resume embedding
            resume_embedding = resume_data.get('embedding')
            if resume_embedding is None:
                resume_embedding = embedding_svc.encode(resume_data['cleaned_text'])

            # Step 2: Use FAISS vector search index (Directive 3 Fix)
            from app.ml.job_index import job_index
            job_map = {j['id']: j for j in jobs_data}

            if not job_index.is_built or job_index.size == 0:
                job_texts = [j['combined_text'] for j in jobs_data]
                job_embeddings = embedding_svc.encode_batch(job_texts)
                job_index.build([j['id'] for j in jobs_data], job_embeddings)

            # Query FAISS nearest neighbors
            faiss_scores, faiss_job_ids = job_index.search(resume_embedding, top_k=min(top_n * 3, len(jobs_data)))
            embedding_scores = dict(zip(faiss_job_ids, faiss_scores))

        except Exception as e:
            logger.error(f'Embedding computation failed, falling back to zero scores: {e}')
            embedding_scores = {}

        recommendations = []
        for i, job in enumerate(jobs_data):
            # Structured signal scoring (kept from v1, these are reliable)
            skill_score = self._compute_skill_score(
                resume_data.get('skills', []),
                job.get('required_skills', [])
            )
            education_score = self._compute_education_score(
                resume_data.get('education', []),
                job.get('experience_level', '')
            )
            experience_score = self._compute_experience_score(
                resume_data.get('experience', []),
                job.get('experience_level', '')
            )

            # Category bonus
            category_bonus = 0.0
            if (resume_data.get('category') and job.get('category') and
                    resume_data['category'] == job['category']):
                category_bonus = CATEGORY_BONUS

            # Overall score
            emb_val = max(0.0, float(embedding_scores.get(job['id'], 0.0)))

            overall_score = (
                WEIGHT_EMBEDDING * emb_val +
                WEIGHT_SKILL * skill_score +
                WEIGHT_EDUCATION * education_score +
                WEIGHT_EXPERIENCE * experience_score +
                category_bonus
            )

            # Clamp to [0, 1]
            overall_score = min(1.0, max(0.0, overall_score))

            # Generate feedback
            strengths, improvements = self._generate_feedback(
                resume_data, job, skill_score, education_score, experience_score
            )

            recommendations.append({
                'job_id': job['id'],
                'overall_score': round(overall_score, 4),
                'skill_score': round(skill_score, 4),
                'experience_score': round(experience_score, 4),
                'education_score': round(education_score, 4),
                'score_breakdown': {
                    'embedding_similarity': round(emb_val, 4),
                    'skill_score': round(skill_score, 4),
                    'education_score': round(education_score, 4),
                    'experience_score': round(experience_score, 4),
                    'category_bonus': round(category_bonus, 4),
                },
                'strengths': strengths,
                'improvements': improvements,
            })

        # Sort by overall score (descending)
        recommendations.sort(key=lambda x: x['overall_score'], reverse=True)
        return recommendations[:top_n]

    def rank_candidates(self, job_data, candidates_data, top_n=20):
        """
        Rank candidates for a specific job posting.

        Args:
            job_data: Dict with job details (combined_text, required_skills, etc.)
            candidates_data: List of candidate dicts with resume data.
            top_n: Number of top candidates to return.

        Returns:
            List of ranked candidate dicts.
        """
        if not candidates_data or not job_data.get('combined_text'):
            return []

        try:
            embedding_svc = self._get_embedding_service()

            # Compute job embedding
            job_embedding = embedding_svc.encode(job_data['combined_text'])

            # Compute candidate embeddings
            candidate_texts = [c.get('cleaned_text', '') for c in candidates_data]
            candidate_embeddings = embedding_svc.encode_batch(candidate_texts)

            # Embedding similarities
            embedding_scores = np.dot(candidate_embeddings, job_embedding)

        except Exception as e:
            logger.error(f'Embedding computation failed for ranking: {e}')
            embedding_scores = np.zeros(len(candidates_data))

        rankings = []
        for i, candidate in enumerate(candidates_data):
            skill_score = self._compute_skill_score(
                candidate.get('skills', []),
                job_data.get('required_skills', [])
            )
            education_score = self._compute_education_score(
                candidate.get('education', []),
                job_data.get('experience_level', '')
            )
            experience_score = self._compute_experience_score(
                candidate.get('experience', []),
                job_data.get('experience_level', '')
            )

            emb_val = max(0.0, float(embedding_scores[i]))

            # Category bonus
            category_bonus = 0.0
            if (candidate.get('category') and job_data.get('category') and
                    candidate['category'] == job_data['category']):
                category_bonus = CATEGORY_BONUS

            overall_score = (
                WEIGHT_EMBEDDING * emb_val +
                WEIGHT_SKILL * skill_score +
                WEIGHT_EDUCATION * education_score +
                WEIGHT_EXPERIENCE * experience_score +
                category_bonus
            )
            overall_score = min(1.0, max(0.0, overall_score))

            rankings.append({
                'candidate_id': candidate.get('id'),
                'overall_score': round(overall_score, 4),
                'skill_score': round(skill_score, 4),
                'experience_score': round(experience_score, 4),
                'education_score': round(education_score, 4),
                'cosine_similarity': round(emb_val, 4),
            })

        rankings.sort(key=lambda x: x['overall_score'], reverse=True)
        return rankings[:top_n]

    @staticmethod
    def _compute_skill_score(resume_skills, job_skills):
        """
        Compute continuous soft-margin skill match score (Directive 5 Fix).

        Uses synonym mapping and partial matching followed by a smooth power-scaling curve
        (coverage^0.75) instead of hardcoded step functions.
        """
        if not job_skills:
            return 0.5  # No requirements specified → neutral score

        # Normalize all skills through synonym mapping
        resume_set = set(_normalize_skill(s) for s in resume_skills)
        job_set = set(_normalize_skill(s) for s in job_skills)

        # Remove empty strings
        resume_set.discard('')
        job_set.discard('')

        if not job_set:
            return 0.5

        # Direct match
        matched = resume_set & job_set

        # Partial/substring matching for remaining unmatched job skills
        unmatched_job = job_set - matched
        for job_skill in list(unmatched_job):
            for resume_skill in resume_set:
                if (job_skill in resume_skill or resume_skill in job_skill) and \
                   len(min(job_skill, resume_skill, key=len)) >= 3:
                    matched.add(job_skill)
                    unmatched_job.discard(job_skill)
                    break

        # Coverage: percentage of required skills matched
        coverage = len(matched) / len(job_set)

        # Directive 5 Fix: Smooth continuous soft-margin curve without hardcoded step jumps
        return min(1.0, max(0.0, round(float(coverage ** 0.75), 4)))

    @staticmethod
    def _compute_education_score(resume_education, job_experience_level):
        """
        Compute education match score.

        Scoring:
        - Certificate/Diploma → 0.5
        - Bachelor's → 0.7
        - Master's → 0.9
        - Doctorate → 1.0
        - No education data → 0.4

        Args:
            resume_education: List of education entry dicts.
            job_experience_level: String describing required experience level.

        Returns:
            Float between 0.0 and 1.0
        """
        if not resume_education:
            return 0.4  # No education data → moderate score

        # Find the highest education level
        max_level = 0
        for edu in resume_education:
            degree = edu.get('degree', 'Other') or 'Other'
            degree_lower = degree.lower().strip()
            level = EDUCATION_LEVELS.get(degree_lower, 2)
            max_level = max(max_level, level)

        # Map levels to scores
        level_scores = {
            0: 0.4,   # No data
            1: 0.5,   # Certificate
            2: 0.55,  # Diploma/Associate
            3: 0.7,   # Bachelor's
            4: 0.9,   # Master's
            5: 1.0,   # Doctorate/PhD
        }
        return level_scores.get(max_level, 0.5)

    @staticmethod
    def _compute_experience_score(resume_experience, job_experience_level):
        """
        Compute experience match score.

        Parses actual date ranges from experience entries.

        Args:
            resume_experience: List of experience entry dicts.
            job_experience_level: String describing required experience level.

        Returns:
            Float between 0.0 and 1.0
        """
        if not resume_experience:
            return 0.4  # No experience data → moderate score

        # Estimate total years from experience entries
        estimated_years = 0
        for exp in resume_experience:
            start = exp.get('start_date', '')
            end = exp.get('end_date', '')

            years = _estimate_duration_years(start, end)
            if years > 0:
                estimated_years += years
            else:
                # Fallback: assume ~2 years per position if dates can't be parsed
                estimated_years += 2

        # If job specifies experience level, match against it
        if job_experience_level:
            level_key = job_experience_level.lower().strip()

            # Try to extract years from strings like "2-5 years", "3+ years"
            year_match = re.search(r'(\d+)\s*[-–to]+\s*(\d+)', level_key)
            year_match_plus = re.search(r'(\d+)\s*\+?\s*years?', level_key)

            if year_match:
                min_years = int(year_match.group(1))
                max_years = int(year_match.group(2))
            elif year_match_plus:
                min_years = int(year_match_plus.group(1))
                max_years = min_years + 5
            elif level_key in EXPERIENCE_LEVELS:
                min_years, max_years = EXPERIENCE_LEVELS[level_key]
            else:
                # Default: more experience → higher score
                return min(1.0, 0.4 + estimated_years * 0.08)

            if estimated_years >= min_years:
                ratio = min(1.0, (estimated_years - min_years) / max(1, max_years - min_years))
                return min(1.0, 0.7 + ratio * 0.3)
            else:
                ratio = estimated_years / max(1, min_years)
                return max(0.2, 0.3 + ratio * 0.4)

        # Default: more experience → higher score (capped at 1.0)
        return min(1.0, 0.4 + estimated_years * 0.08)

    @staticmethod
    def _generate_feedback(resume_data, job_data, skill_score, edu_score, exp_score):
        """
        Generate human-readable strengths and improvement suggestions.

        Returns:
            Tuple of (strengths: list, improvements: list)
        """
        strengths = []
        improvements = []

        # Skill feedback
        resume_skills = set(_normalize_skill(s) for s in resume_data.get('skills', []))
        job_skills = set(_normalize_skill(s) for s in job_data.get('required_skills', []))

        matched_skills = resume_skills & job_skills
        missing_skills = job_skills - resume_skills

        if matched_skills:
            top_matched = list(matched_skills)[:5]
            strengths.append(f"Strong match in: {', '.join(top_matched)}")

        if missing_skills:
            top_missing = list(missing_skills)[:5]
            improvements.append(f"Consider learning: {', '.join(top_missing)}")

        if skill_score >= 0.8:
            strengths.append("Excellent skill coverage for this role")
        elif skill_score < 0.3:
            improvements.append("Skills gap detected — focus on key technical requirements")

        # Education feedback
        if edu_score >= 0.8:
            strengths.append("Strong educational background")
        elif edu_score < 0.5:
            improvements.append("Consider pursuing additional qualifications or certifications")

        # Experience feedback
        if exp_score >= 0.8:
            strengths.append("Relevant experience level for this position")
        elif exp_score < 0.4:
            improvements.append("More industry experience would strengthen your application")

        # Category match
        if (resume_data.get('category') and job_data.get('category') and
                resume_data['category'] == job_data['category']):
            strengths.append(f"Resume domain ({resume_data['category']}) matches job category")

        # Default fallbacks
        if not strengths:
            strengths.append("Profile has potential for this role")
        if not improvements:
            improvements.append("Continue building your portfolio and skills")

        return strengths[:5], improvements[:5]


def _estimate_duration_years(start_str, end_str):
    """
    Estimate duration in years from start/end date strings.

    Handles formats like:
    - "Jan 2020" / "January 2020"
    - "2020" (just year)
    - "Present" / "Current"

    Returns estimated years (float), or 0 if parsing fails.
    """
    if not start_str:
        return 0

    from datetime import datetime

    def parse_year(date_str):
        if not date_str:
            return None
        date_str = date_str.strip().lower()
        if date_str in ('present', 'current', 'now', 'ongoing'):
            return datetime.now().year

        # Try to extract a 4-digit year
        year_match = re.search(r'(\d{4})', date_str)
        if year_match:
            return int(year_match.group(1))
        return None

    start_year = parse_year(start_str)
    end_year = parse_year(end_str)

    if start_year and end_year:
        return max(0, end_year - start_year)
    elif start_year:
        # No end date, assume current
        return max(0, datetime.now().year - start_year)

    return 0


# Singleton instance
recommendation_engine = RecommendationEngine()
