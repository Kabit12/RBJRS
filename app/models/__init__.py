"""
Models Package
===============
Imports all models so SQLAlchemy can discover them during table creation
and migration generation. This file also makes imports cleaner elsewhere:

    from app.models import User, Candidate, Job  # instead of separate imports

Every model imported here will be registered with SQLAlchemy's metadata,
ensuring create_all() and Alembic migrations work correctly.
"""

from app.models.user import User
from app.models.candidate import Candidate
from app.models.recruiter import Recruiter
from app.models.skill import Skill
from app.models.resume import (
    Resume,
    ResumeSkill,
    ResumeEducation,
    ResumeExperience,
    ResumeCertification,
    ResumeProject,
)
from app.models.job import Job, JobSkill
from app.models.application import Application
from app.models.recommendation import Recommendation

# This list makes it convenient to reference all models programmatically
__all__ = [
    'User',
    'Candidate',
    'Recruiter',
    'Skill',
    'Resume',
    'ResumeSkill',
    'ResumeEducation',
    'ResumeExperience',
    'ResumeCertification',
    'ResumeProject',
    'Job',
    'JobSkill',
    'Application',
    'Recommendation',
]
