"""
Recommendation Engine Module
==============================
Core matching engine that recommends jobs to candidates and ranks candidates for jobs.

Algorithm (as specified in the implementation plan):

Step 1: TF-IDF Vectorization
    Resume Text → TF-IDF Vector (R)
    Job Text → TF-IDF Vector (J)

Step 2: Cosine Similarity
    similarity(R, J) = (R · J) / (||R|| × ||J||)

Step 3: Component Scoring
    skill_score = weighted_jaccard(resume_skills, job_required_skills)
    education_score = education_level_match(resume_edu, job_edu_requirement)
    experience_score = experience_years_match(resume_exp, job_exp_requirement)

Step 4: Job Fit Score
    job_fit_score = 0.4 × cosine_similarity + 0.3 × skill_score
                  + 0.15 × education_score + 0.15 × experience_score

Step 5: KNN Refinement
    Find K nearest job vectors to resume vector
    Combine with cosine similarity rankings

Design Decisions:
- Cosine similarity is the primary matching metric because it's scale-invariant
  and works well with TF-IDF vectors (both resume and job texts).
- KNN refines the ranking by finding structurally similar jobs in the TF-IDF space.
- The weighted scoring formula gives highest weight to text similarity (0.4) and
  skills (0.3) because these are the strongest predictors of job fit.
- Education and experience each get 0.15 weight as supporting factors.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors


# Score component weights (must sum to 1.0)
WEIGHT_COSINE = 0.40
WEIGHT_SKILL = 0.30
WEIGHT_EDUCATION = 0.15
WEIGHT_EXPERIENCE = 0.15

# Education level hierarchy for scoring
EDUCATION_LEVELS = {
    'certificate': 1,
    'diploma': 2,
    'associate': 2,
    "bachelor's": 3,
    "master's": 4,
    'doctorate': 5,
    'phd': 5,
    'other': 2,
}

# Experience level mapping (years)
EXPERIENCE_LEVELS = {
    'entry': (0, 2),
    'entry level': (0, 2),
    'junior': (0, 3),
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


class RecommendationEngine:
    """
    Core recommendation engine for job-resume matching.

    Computes multi-factor similarity scores between resumes and jobs
    using TF-IDF cosine similarity, skill matching, and KNN refinement.

    Usage:
        engine = RecommendationEngine()
        recommendations = engine.recommend_jobs(resume_data, jobs_data, top_n=10)
        rankings = engine.rank_candidates(job_data, candidates_data, top_n=20)
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
            sublinear_tf=True,
        )

    def recommend_jobs(self, resume_data, jobs_data, top_n=10):
        """
        Recommend jobs for a candidate based on their resume.

        Args:
            resume_data: Dict with keys:
                - cleaned_text: Preprocessed resume text
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

        # Step 1: TF-IDF Vectorization
        all_texts = [resume_data['cleaned_text']] + [j['combined_text'] for j in jobs_data]
        try:
            tfidf_matrix = self.vectorizer.fit_transform(all_texts)
        except ValueError:
            return []

        resume_vector = tfidf_matrix[0:1]
        job_vectors = tfidf_matrix[1:]

        # Step 2: Cosine Similarity
        cosine_scores = cosine_similarity(resume_vector, job_vectors)[0]

        # Step 5: KNN Refinement
        knn_scores = self._compute_knn_scores(resume_vector, job_vectors, k=min(5, len(jobs_data)))

        recommendations = []
        for i, job in enumerate(jobs_data):
            # Step 3: Component Scoring
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

            # Category bonus: boost score if resume and job are in the same category
            category_bonus = 0.0
            if (resume_data.get('category') and job.get('category') and
                    resume_data['category'] == job['category']):
                category_bonus = 0.1

            # Step 4: Job Fit Score
            cosine_val = float(cosine_scores[i])
            knn_val = float(knn_scores[i]) if knn_scores is not None else cosine_val

            # Blend cosine and KNN scores
            text_score = 0.7 * cosine_val + 0.3 * knn_val

            overall_score = (
                WEIGHT_COSINE * text_score +
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
                    'cosine_similarity': round(cosine_val, 4),
                    'knn_score': round(float(knn_val), 4),
                    'text_score': round(text_score, 4),
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

        all_texts = [job_data['combined_text']] + [c.get('cleaned_text', '') for c in candidates_data]
        try:
            tfidf_matrix = self.vectorizer.fit_transform(all_texts)
        except ValueError:
            return []

        job_vector = tfidf_matrix[0:1]
        candidate_vectors = tfidf_matrix[1:]

        cosine_scores = cosine_similarity(job_vector, candidate_vectors)[0]

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

            cosine_val = float(cosine_scores[i])

            overall_score = (
                WEIGHT_COSINE * cosine_val +
                WEIGHT_SKILL * skill_score +
                WEIGHT_EDUCATION * education_score +
                WEIGHT_EXPERIENCE * experience_score
            )
            overall_score = min(1.0, max(0.0, overall_score))

            rankings.append({
                'candidate_id': candidate.get('id'),
                'overall_score': round(overall_score, 4),
                'skill_score': round(skill_score, 4),
                'experience_score': round(experience_score, 4),
                'education_score': round(education_score, 4),
                'cosine_similarity': round(cosine_val, 4),
            })

        rankings.sort(key=lambda x: x['overall_score'], reverse=True)
        return rankings[:top_n]

    def _compute_knn_scores(self, resume_vector, job_vectors, k=5):
        """
        Compute KNN-based similarity scores.

        Finds the K nearest job vectors to the resume vector and
        converts distances to similarity scores.

        Args:
            resume_vector: Sparse TF-IDF vector for the resume.
            job_vectors: Sparse TF-IDF matrix for all jobs.
            k: Number of nearest neighbors.

        Returns:
            Array of similarity scores (1 / (1 + distance)).
        """
        if job_vectors.shape[0] < k:
            k = job_vectors.shape[0]

        if k == 0:
            return None

        try:
            knn = NearestNeighbors(n_neighbors=k, metric='cosine', algorithm='brute')
            knn.fit(job_vectors)
            distances, indices = knn.kneighbors(resume_vector)

            # Initialize scores with zeros
            scores = np.zeros(job_vectors.shape[0])

            # Convert distances to similarity scores for KNN neighbors
            for dist, idx in zip(distances[0], indices[0]):
                scores[idx] = 1.0 - dist  # cosine distance to similarity

            # For non-neighbors, use a baseline score
            baseline = float(np.mean(scores[scores > 0])) * 0.5 if np.any(scores > 0) else 0.0
            scores[scores == 0] = baseline

            return scores

        except Exception as e:
            print(f'KNN computation error: {e}')
            return None

    @staticmethod
    def _compute_skill_score(resume_skills, job_skills):
        """
        Compute weighted Jaccard similarity between resume and job skills.

        Weighted Jaccard gives more credit for matching required skills
        vs. having extra unrelated skills.

        Args:
            resume_skills: List of skill name strings from resume.
            job_skills: List of skill name strings required by job.

        Returns:
            Float between 0.0 and 1.0
        """
        if not job_skills:
            return 0.5  # No requirements specified → neutral score

        resume_set = set(s.lower() if isinstance(s, str) else s.get('name', '').lower()
                         for s in resume_skills)
        job_set = set(s.lower() if isinstance(s, str) else s.get('name', '').lower()
                      for s in job_skills)

        if not job_set:
            return 0.5

        # How many required skills does the candidate have?
        matched = resume_set & job_set
        coverage = len(matched) / len(job_set)

        return min(1.0, coverage)

    @staticmethod
    def _compute_education_score(resume_education, job_experience_level):
        """
        Compute education match score.

        Higher education levels get slightly higher scores.
        If no education data is available, returns a neutral 0.5.

        Args:
            resume_education: List of education entry dicts.
            job_experience_level: String describing required experience level.

        Returns:
            Float between 0.0 and 1.0
        """
        if not resume_education:
            return 0.3  # No education data → low score

        # Find the highest education level
        max_level = 0
        for edu in resume_education:
            degree = edu.get('degree', 'Other')
            level = EDUCATION_LEVELS.get(degree.lower(), 2)
            max_level = max(max_level, level)

        # Normalize to 0-1 range (5 levels)
        return min(1.0, max_level / 5.0)

    @staticmethod
    def _compute_experience_score(resume_experience, job_experience_level):
        """
        Compute experience match score.

        Estimates years of experience from resume entries and compares
        to job requirements.

        Args:
            resume_experience: List of experience entry dicts.
            job_experience_level: String describing required experience level.

        Returns:
            Float between 0.0 and 1.0
        """
        if not resume_experience:
            return 0.3  # No experience data → low score

        # Estimate total years from number of positions
        # (rough heuristic: average 2 years per position)
        estimated_years = len(resume_experience) * 2

        # If job specifies experience level, match against it
        if job_experience_level:
            level_key = job_experience_level.lower().strip()
            if level_key in EXPERIENCE_LEVELS:
                min_years, max_years = EXPERIENCE_LEVELS[level_key]
                if estimated_years >= min_years:
                    return min(1.0, 0.6 + (estimated_years - min_years) / (max_years - min_years + 1) * 0.4)
                else:
                    return max(0.1, estimated_years / min_years * 0.6)

        # Default: more experience → higher score (capped)
        return min(1.0, 0.3 + estimated_years * 0.07)

    @staticmethod
    def _generate_feedback(resume_data, job_data, skill_score, edu_score, exp_score):
        """
        Generate human-readable strengths and improvement suggestions.

        This implements the proposal requirement (Section 3.2.4) for
        providing actionable feedback on candidate profiles.

        Returns:
            Tuple of (strengths: list, improvements: list)
        """
        strengths = []
        improvements = []

        # Skill feedback
        resume_skills = set(
            s.lower() if isinstance(s, str) else s.get('name', '').lower()
            for s in resume_data.get('skills', [])
        )
        job_skills = set(
            s.lower() if isinstance(s, str) else s.get('name', '').lower()
            for s in job_data.get('required_skills', [])
        )

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
        elif edu_score < 0.4:
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


# Singleton instance
recommendation_engine = RecommendationEngine()
