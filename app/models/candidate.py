"""
Candidate Model
================
Profile data specific to job seekers. Extends the base User model
through a one-to-one foreign key relationship.

Maps to: CANDIDATES table in the ER diagram.

Design Decisions:
- Separated from User table because candidate-specific fields (headline, bio,
  linkedin, portfolio) don't apply to recruiters or admins.
- The user_id foreign key enforces the one-to-one relationship.
- Contains personal/professional metadata used in recommendation context
  but NOT for the ML matching itself (that comes from parsed resume data).
"""

from app.extensions import db


class Candidate(db.Model):
    """
    Job seeker profile containing personal and professional metadata.

    This model stores the candidate's self-reported profile information.
    The actual data used for job matching (skills, experience, education)
    comes from the parsed Resume model, not from here.
    """

    __tablename__ = 'candidates'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(
        db.Integer,
        db.ForeignKey('users.id', ondelete='CASCADE'),
        unique=True,
        nullable=False
    )

    # Contact & Location
    phone = db.Column(db.String(20))
    location = db.Column(db.String(255))

    # Professional Summary
    headline = db.Column(db.String(255))  # e.g., "Full Stack Developer | Python Expert"
    bio = db.Column(db.Text)

    # Online Presence
    linkedin_url = db.Column(db.String(500))
    portfolio_url = db.Column(db.String(500))

    # Profile Image
    profile_image = db.Column(db.String(500))

    # Relationships
    resumes = db.relationship(
        'Resume', backref='candidate', lazy='dynamic', cascade='all, delete-orphan'
    )
    applications = db.relationship(
        'Application', backref='candidate', lazy='dynamic', cascade='all, delete-orphan'
    )
    recommendations = db.relationship(
        'Recommendation', backref='candidate', lazy='dynamic', cascade='all, delete-orphan'
    )

    @property
    def active_resume(self):
        """Returns the candidate's currently active resume, if any."""
        return self.resumes.filter_by(is_active=True).first()

    def __repr__(self):
        return f'<Candidate {self.user.email if self.user else self.id}>'
