"""
Recommendation Service
========================
Business logic for generating, storing, and retrieving job recommendations.

Orchestrates:
1. Loading resume data and all active jobs
2. Running the recommendation engine
3. Storing results in the database
4. Retrieving recommendations for display
"""

from app.extensions import db
from app.models.job import Job
from app.models.recommendation import Recommendation
from app.models.resume import Resume
from app.models.candidate import Candidate
from app.services.resume_service import get_resume_data_for_matching
from app.services.job_service import get_job_data_for_matching
from app.ml.recommendation_engine import recommendation_engine
from app.ml.text_preprocessor import text_preprocessor


def generate_recommendations(candidate, resume=None, top_n=20):
    """
    Generate job recommendations for a candidate based on their active resume.

    Process:
    1. Get the candidate's active resume data
    2. Get all active job postings
    3. Run the recommendation engine
    4. Store recommendations in the database

    Args:
        candidate: Candidate model instance.
        resume: Optional specific resume to use (defaults to active resume).
        top_n: Number of recommendations to generate.

    Returns:
        Tuple of (success: bool, count_or_error: int | str)
    """
    try:
        # Get resume
        if resume is None:
            resume = candidate.active_resume
        if not resume:
            return False, 'No active resume found. Please upload a resume first.'

        # Prepare resume data
        resume_data = get_resume_data_for_matching(resume)
        if not resume_data['cleaned_text']:
            return False, 'Resume text is empty. Please upload a valid resume.'

        # Get all active jobs
        active_jobs = Job.query.filter_by(is_active=True).all()
        if not active_jobs:
            return False, 'No active job postings available.'

        # Prepare job data
        jobs_data = [get_job_data_for_matching(job) for job in active_jobs]

        # Run recommendation engine
        recommendations = recommendation_engine.recommend_jobs(
            resume_data, jobs_data, top_n=top_n
        )

        # Clear existing recommendations for this candidate
        Recommendation.query.filter_by(candidate_id=candidate.id).delete()
        db.session.flush()

        # Store new recommendations
        count = 0
        for rec in recommendations:
            if rec['overall_score'] > 0.05:  # Filter very low scores
                db_rec = Recommendation(
                    candidate_id=candidate.id,
                    job_id=rec['job_id'],
                    resume_id=resume.id,
                    overall_score=rec['overall_score'],
                    skill_score=rec['skill_score'],
                    experience_score=rec['experience_score'],
                    education_score=rec['education_score'],
                    score_breakdown=rec['score_breakdown'],
                    strengths=rec['strengths'],
                    improvements=rec['improvements'],
                )
                db.session.add(db_rec)
                count += 1

        db.session.commit()
        return True, count

    except Exception as e:
        db.session.rollback()
        return False, f'Recommendation generation failed: {str(e)}'


def get_candidate_recommendations(candidate, min_score=0.0):
    """
    Retrieve stored recommendations for a candidate.

    Args:
        candidate: Candidate model instance.
        min_score: Minimum score threshold.

    Returns:
        List of Recommendation model instances with associated Job data.
    """
    return (
        Recommendation.query
        .filter_by(candidate_id=candidate.id)
        .filter(Recommendation.overall_score >= min_score)
        .order_by(Recommendation.overall_score.desc())
        .all()
    )


def get_ranked_candidates_for_job(job, top_n=50):
    """
    Rank all candidates for a specific job posting.

    Args:
        job: Job model instance.
        top_n: Maximum candidates to rank.

    Returns:
        List of dicts with candidate info and scores.
    """
    try:
        job_data = get_job_data_for_matching(job)

        # Get all candidates with active resumes
        candidates = Candidate.query.all()
        candidates_data = []

        for candidate in candidates:
            resume = candidate.active_resume
            if resume and resume.cleaned_text:
                resume_data = get_resume_data_for_matching(resume)
                resume_data['id'] = candidate.id
                candidates_data.append(resume_data)

        if not candidates_data:
            return []

        # Run ranking
        rankings = recommendation_engine.rank_candidates(
            job_data, candidates_data, top_n=top_n
        )

        # Enrich with candidate info
        enriched = []
        for ranking in rankings:
            candidate = db.session.get(Candidate, ranking['candidate_id'])
            if candidate and candidate.user:
                enriched.append({
                    'candidate': candidate,
                    'user': candidate.user,
                    'overall_score': ranking['overall_score'],
                    'skill_score': ranking['skill_score'],
                    'experience_score': ranking['experience_score'],
                    'education_score': ranking['education_score'],
                    'cosine_similarity': ranking['cosine_similarity'],
                })

        return enriched

    except Exception as e:
        print(f'Candidate ranking error: {e}')
        return []
