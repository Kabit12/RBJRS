"""
Application Service
=====================
Business logic for job applications, extracted from candidate routes.

Design Decision:
- Separates HTTP concerns (routes) from business logic (this service).
- The route handler only deals with request/response/redirect.
- All DB operations, file handling, and notifications happen here.
"""

import os
from flask import current_app
from app.extensions import db
from app.models.application import Application
from app.models.notification import Notification
from app.utils.validators import secure_filename_custom, validate_resume_file


def create_application(candidate, job, resume, cover_letter_text=None,
                       cover_letter_file=None, match_score=None, score_breakdown=None):
    """
    Create a new job application for a candidate.

    Args:
        candidate: Candidate model instance.
        job: Job model instance.
        resume: Resume model instance to attach.
        cover_letter_text: Optional cover letter text.
        cover_letter_file: Optional Flask FileStorage for cover letter.
        match_score: Optional match score from recommendations.
        score_breakdown: Optional score breakdown dict.

    Returns:
        Tuple of (success: bool, application_or_error: Application | str)
    """
    try:
        # Check for existing application
        existing = Application.query.filter_by(
            candidate_id=candidate.id, job_id=job.id
        ).first()
        if existing:
            return False, 'You have already applied for this job.'

        # Handle cover letter file upload
        cover_letter_file_path = None
        if cover_letter_file and cover_letter_file.filename:
            filename = secure_filename_custom(cover_letter_file.filename)
            upload_dir = os.path.join(
                current_app.config['UPLOAD_FOLDER'], 'cover_letters'
            )
            os.makedirs(upload_dir, exist_ok=True)
            file_path = os.path.join(upload_dir, filename)
            cover_letter_file.save(file_path)
            cover_letter_file_path = file_path

        # Create application
        application = Application(
            candidate_id=candidate.id,
            job_id=job.id,
            resume_id=resume.id if resume else None,
            status='pending',
            match_score=match_score,
            score_breakdown=score_breakdown,
            cover_letter=cover_letter_text.strip() if cover_letter_text else None,
            cover_letter_file=cover_letter_file_path,
        )
        db.session.add(application)
        db.session.flush()

        # Notify the recruiter
        _notify_recruiter(job, candidate)

        db.session.commit()
        return True, application

    except Exception as e:
        db.session.rollback()
        return False, f'Application submission failed: {str(e)}'


def _notify_recruiter(job, candidate):
    """Create a notification for the recruiter about a new application."""
    try:
        from flask import url_for
        recruiter_user_id = job.recruiter.user_id
        notification = Notification(
            user_id=recruiter_user_id,
            message=f'{candidate.user.full_name} applied for "{job.title}"',
            link=url_for('recruiter.view_applicants', job_id=job.id),
        )
        db.session.add(notification)
    except Exception:
        # Don't fail the application if notification fails
        pass
