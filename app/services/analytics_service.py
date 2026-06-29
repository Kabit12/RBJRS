"""
Analytics Service
===================
Business logic for platform analytics and reporting.
Provides aggregated statistics for the admin dashboard.
"""

from app.extensions import db
from app.models.user import User
from app.models.candidate import Candidate
from app.models.recruiter import Recruiter
from app.models.job import Job
from app.models.application import Application
from app.models.recommendation import Recommendation
from app.models.resume import Resume
from sqlalchemy import func


def get_platform_stats():
    """
    Get overall platform statistics.

    Returns:
        Dictionary with aggregated platform metrics.
    """
    return {
        'total_users': User.query.count(),
        'total_candidates': User.query.filter_by(role='candidate').count(),
        'total_recruiters': User.query.filter_by(role='recruiter').count(),
        'total_jobs': Job.query.count(),
        'active_jobs': Job.query.filter_by(is_active=True).count(),
        'total_applications': Application.query.count(),
        'total_resumes': Resume.query.count(),
        'total_recommendations': Recommendation.query.count(),
    }


def get_application_stats():
    """Get application status breakdown."""
    statuses = db.session.query(
        Application.status,
        func.count(Application.id)
    ).group_by(Application.status).all()

    return {status: count for status, count in statuses}


def get_job_category_stats():
    """Get job distribution by category."""
    categories = db.session.query(
        Job.classified_category,
        func.count(Job.id)
    ).group_by(Job.classified_category).all()

    return {cat or 'Uncategorized': count for cat, count in categories}


def get_recent_users(limit=10):
    """Get most recently registered users."""
    return User.query.order_by(User.created_at.desc()).limit(limit).all()


def get_recent_jobs(limit=10):
    """Get most recently posted jobs."""
    return Job.query.order_by(Job.posted_at.desc()).limit(limit).all()


def get_recent_applications(limit=10):
    """Get most recent applications."""
    return Application.query.order_by(Application.applied_at.desc()).limit(limit).all()
