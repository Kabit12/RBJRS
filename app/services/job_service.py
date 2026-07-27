"""
Job Service
=============
Business logic for job posting CRUD operations and job data management.
"""

from app.extensions import db
from app.models.job import Job, JobSkill
from app.models.skill import Skill
from app.ml.text_preprocessor import text_preprocessor
from app.ml.job_classifier import job_classifier
from flask import current_app


def create_job(recruiter, form_data):
    """
    Create a new job posting.

    Args:
        recruiter: Recruiter model instance.
        form_data: Dict with job details.

    Returns:
        Tuple of (success: bool, job_or_error: Job | str)
    """
    try:
        job = Job(
            recruiter_id=recruiter.id,
            title=form_data['title'].strip(),
            description=form_data['description'].strip(),
            location=form_data.get('location', '').strip() or None,
            job_type=form_data.get('job_type', 'full-time'),
            experience_level=form_data.get('experience_level', '').strip() or None,
            salary_min=int(form_data['salary_min']) if form_data.get('salary_min') else None,
            salary_max=int(form_data['salary_max']) if form_data.get('salary_max') else None,
            requirements=form_data.get('requirements', '').strip() or None,
            responsibilities=form_data.get('responsibilities', '').strip() or None,
            is_active=True,
        )

        # Classify the job into a category
        model_dir = current_app.config.get('MODEL_DIR', 'ml_models')
        if not job_classifier.is_loaded:
            job_classifier.load_model(model_dir)

        combined = job.combined_text
        if combined:
            category, _ = job_classifier.predict(
                text_preprocessor.preprocess(combined)
            )
            job.classified_category = category

        db.session.add(job)
        db.session.flush()

        # Add skills
        skills_text = form_data.get('skills', '')
        if skills_text:
            skill_names = [s.strip() for s in skills_text.split(',') if s.strip()]
            for skill_name in skill_names:
                skill = Skill.get_or_create(skill_name)
                db.session.flush()
                job_skill = JobSkill(
                    job_id=job.id,
                    skill_id=skill.id,
                    is_required=True,
                )
                db.session.add(job_skill)

        db.session.commit()

        # Directive 3 Fix: Upsert job embedding into FAISS index
        try:
            from app.ml.embedding_service import embedding_service
            from app.ml.job_index import job_index
            if job.combined_text:
                job_emb = embedding_service.encode(job.combined_text)
                job_index.upsert_job(job.id, job_emb)
        except Exception as e:
            current_app.logger.warning(f'FAISS index update failed for job {job.id}: {e}')

        return True, job

    except Exception as e:
        db.session.rollback()
        return False, f'Failed to create job: {str(e)}'


def update_job(job, form_data):
    """
    Update an existing job posting.

    Args:
        job: Job model instance to update.
        form_data: Dict with updated job details.

    Returns:
        Tuple of (success: bool, job_or_error: Job | str)
    """
    try:
        job.title = form_data['title'].strip()
        job.description = form_data['description'].strip()
        job.location = form_data.get('location', '').strip() or None
        job.job_type = form_data.get('job_type', 'full-time')
        job.experience_level = form_data.get('experience_level', '').strip() or None
        job.salary_min = int(form_data['salary_min']) if form_data.get('salary_min') else None
        job.salary_max = int(form_data['salary_max']) if form_data.get('salary_max') else None
        job.requirements = form_data.get('requirements', '').strip() or None
        job.responsibilities = form_data.get('responsibilities', '').strip() or None

        # Re-classify
        model_dir = current_app.config.get('MODEL_DIR', 'ml_models')
        if not job_classifier.is_loaded:
            job_classifier.load_model(model_dir)

        combined = job.combined_text
        if combined:
            category, _ = job_classifier.predict(
                text_preprocessor.preprocess(combined)
            )
            job.classified_category = category

        # Update skills
        skills_text = form_data.get('skills', '')
        if skills_text:
            # Remove existing skills
            JobSkill.query.filter_by(job_id=job.id).delete()
            db.session.flush()

            skill_names = [s.strip() for s in skills_text.split(',') if s.strip()]
            for skill_name in skill_names:
                skill = Skill.get_or_create(skill_name)
                db.session.flush()
                job_skill = JobSkill(
                    job_id=job.id,
                    skill_id=skill.id,
                    is_required=True,
                )
                db.session.add(job_skill)

        db.session.commit()

        # Directive 3 Fix: Re-upsert updated job embedding into FAISS index
        try:
            from app.ml.embedding_service import embedding_service
            from app.ml.job_index import job_index
            if job.combined_text:
                job_emb = embedding_service.encode(job.combined_text)
                job_index.upsert_job(job.id, job_emb)
        except Exception as e:
            current_app.logger.warning(f'FAISS index update failed for job {job.id}: {e}')

        return True, job

    except Exception as e:
        db.session.rollback()
        return False, f'Failed to update job: {str(e)}'


def delete_job(job):
    """Delete a job posting."""
    try:
        job_id = job.id
        db.session.delete(job)
        db.session.commit()

        # Directive 3 Fix: Remove deleted job from FAISS index
        try:
            from app.ml.job_index import job_index
            job_index.remove_job(job_id)
        except Exception as e:
            current_app.logger.warning(f'FAISS index removal failed for job {job_id}: {e}')

        return True, 'Job deleted successfully.'
    except Exception as e:
        db.session.rollback()
        return False, f'Failed to delete job: {str(e)}'


def get_job_data_for_matching(job):
    """
    Prepare job data in the format expected by the recommendation engine.

    Args:
        job: Job model instance.

    Returns:
        Dict with combined_text, required_skills, experience_level, category.
    """
    required_skills = [
        js.skill.name for js in job.required_skills.all()
        if js.skill and js.is_required
    ]

    return {
        'id': job.id,
        'combined_text': job.combined_text or '',
        'required_skills': required_skills,
        'experience_level': job.experience_level or '',
        'category': job.classified_category,
    }
