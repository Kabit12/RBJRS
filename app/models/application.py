"""
Application Model
==================
Tracks job applications submitted by candidates.

Maps to: APPLICATIONS table in the ER diagram.

Design Decisions:
- The application links a candidate to a job with a specific resume version.
  This means if a candidate updates their resume later, the original match
  score is preserved (important for audit trail and recruiter review).
- Status workflow: pending → reviewed → shortlisted/rejected → accepted
- match_score is the Job Fit Score computed by the recommendation engine
  at the time of application.
- score_breakdown stores the detailed component scores as JSON
  (skill_score, education_score, experience_score, cosine_similarity).
"""

from datetime import datetime, timezone
from app.extensions import db


class Application(db.Model):
    """
    Job application linking a candidate (with a specific resume) to a job.

    The match_score and score_breakdown are computed at application time
    and stored for historical reference even if the resume is later updated.
    """

    __tablename__ = 'applications'

    id = db.Column(db.Integer, primary_key=True)
    candidate_id = db.Column(
        db.Integer,
        db.ForeignKey('candidates.id', ondelete='CASCADE'),
        nullable=False,
        index=True
    )
    job_id = db.Column(
        db.Integer,
        db.ForeignKey('jobs.id', ondelete='CASCADE'),
        nullable=False,
        index=True
    )
    resume_id = db.Column(
        db.Integer,
        db.ForeignKey('resumes.id', ondelete='SET NULL'),
        nullable=True
    )

    # Application Status
    status = db.Column(
        db.Enum('pending', 'reviewed', 'shortlisted', 'rejected', 'accepted',
                name='application_status'),
        default='pending',
        nullable=False
    )

    # Match Score (computed at application time)
    match_score = db.Column(db.Float)  # Overall Job Fit Score (0.0 - 1.0)
    score_breakdown = db.Column(db.JSON)  # Detailed component scores

    # Optional cover letter
    cover_letter = db.Column(db.Text)
    cover_letter_file = db.Column(db.String(500))  # Path to uploaded cover letter file

    # Timestamps
    applied_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = db.Column(
        db.DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc)
    )

    # Prevent duplicate applications (one application per candidate per job)
    __table_args__ = (
        db.UniqueConstraint('candidate_id', 'job_id', name='uq_candidate_job_application'),
    )

    @property
    def status_display(self):
        """Returns a human-readable status string."""
        return self.status.replace('_', ' ').title() if self.status else 'Unknown'

    @property
    def match_percentage(self):
        """Returns match score as a percentage string."""
        if self.match_score is not None:
            return f'{self.match_score * 100:.1f}%'
        return 'N/A'

    def __repr__(self):
        return f'<Application candidate={self.candidate_id} job={self.job_id} ({self.status})>'
