"""
Job Models
===========
Represents job postings created by recruiters and their required skills.

Maps to: JOBS and JOB_SKILLS tables in the ER diagram.

Design Decisions:
- Job type is an enum (full-time, part-time, contract, internship) matching
  standard industry classifications.
- experience_level is a free-text field (e.g., "2-5 years", "Senior") rather
  than an enum because job requirements vary widely.
- requirements and responsibilities are stored as text fields, which will be
  parsed by the NLP pipeline for matching, similar to how resumes are processed.
- classified_category mirrors the resume classifier — both resumes and jobs
  are classified into the same category taxonomy so the recommendation engine
  can prioritize same-category matches.
- The is_active flag and deadline allow jobs to be deactivated or expire.
"""

from datetime import datetime, timezone
from app.extensions import db


class Job(db.Model):
    """
    Job posting created by a recruiter.

    The NLP pipeline processes the description, requirements, and responsibilities
    fields to extract keywords and compute similarity with candidate resumes.
    """

    __tablename__ = 'jobs'

    id = db.Column(db.Integer, primary_key=True)
    recruiter_id = db.Column(
        db.Integer,
        db.ForeignKey('recruiters.id', ondelete='CASCADE'),
        nullable=False,
        index=True
    )

    # Job Details
    title = db.Column(db.String(255), nullable=False)
    description = db.Column(db.Text, nullable=False)
    location = db.Column(db.String(255))
    job_type = db.Column(
        db.Enum('full-time', 'part-time', 'contract', 'internship', name='job_type_enum'),
        default='full-time'
    )
    experience_level = db.Column(db.String(100))  # e.g., "Entry Level", "2-5 years", "Senior"
    salary_range = db.Column(db.String(100))       # e.g., "50000-70000" or "Negotiable"

    # Detailed requirements (parsed by NLP for matching)
    requirements = db.Column(db.Text)        # What the employer needs
    responsibilities = db.Column(db.Text)    # What the role involves

    # ML Classification (same taxonomy as resume classifier)
    classified_category = db.Column(db.String(100))

    # Status
    is_active = db.Column(db.Boolean, default=True, nullable=False)
    posted_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    deadline = db.Column(db.DateTime)

    # Relationships
    required_skills = db.relationship(
        'JobSkill', backref='job', lazy='dynamic', cascade='all, delete-orphan'
    )
    applications = db.relationship(
        'Application', backref='job', lazy='dynamic', cascade='all, delete-orphan'
    )
    recommendations = db.relationship(
        'Recommendation', backref='job', lazy='dynamic', cascade='all, delete-orphan'
    )

    @property
    def is_expired(self):
        """Check if the job posting has passed its deadline."""
        if self.deadline:
            return datetime.now(timezone.utc) > self.deadline
        return False

    @property
    def applicant_count(self):
        """Returns the number of applications received."""
        return self.applications.count()

    @property
    def combined_text(self):
        """
        Concatenates all text fields for TF-IDF vectorization.
        Used by the recommendation engine for cosine similarity computation.
        """
        parts = [
            self.title or '',
            self.description or '',
            self.requirements or '',
            self.responsibilities or '',
        ]
        return ' '.join(parts)

    def __repr__(self):
        return f'<Job {self.title}>'


class JobSkill(db.Model):
    """
    Junction table linking a job to its required/preferred skills.
    The is_required flag distinguishes must-have from nice-to-have skills,
    allowing the scoring engine to weight them differently.
    """

    __tablename__ = 'job_skills'

    id = db.Column(db.Integer, primary_key=True)
    job_id = db.Column(
        db.Integer,
        db.ForeignKey('jobs.id', ondelete='CASCADE'),
        nullable=False,
        index=True
    )
    skill_id = db.Column(
        db.Integer,
        db.ForeignKey('skills.id', ondelete='CASCADE'),
        nullable=False
    )
    is_required = db.Column(db.Boolean, default=True)  # True = required, False = preferred

    # Relationship to Skill model (needed for template access: js.skill.name)
    skill = db.relationship('Skill', backref='job_skills')

    # Prevent duplicate skill entries per job
    __table_args__ = (
        db.UniqueConstraint('job_id', 'skill_id', name='uq_job_skill'),
    )

    def __repr__(self):
        req = 'required' if self.is_required else 'preferred'
        return f'<JobSkill job={self.job_id} skill={self.skill_id} ({req})>'
