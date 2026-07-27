"""
Recruiter Model
================
Company/employer profile for users who post jobs and review candidates.

Maps to: RECRUITERS table in the ER diagram.

Design Decisions:
- Company information is stored directly on the recruiter profile rather
  than in a separate Company table. This simplifies the schema for the
  project scope (one recruiter per company profile).
- If future scaling requires multiple recruiters per company, this can be
  refactored to a separate Company model with a many-to-one relationship.
"""

from app.extensions import db


class Recruiter(db.Model):
    """
    Recruiter/employer profile containing company information.

    Recruiters can:
    - Create, edit, and delete job postings
    - View applicants for their jobs
    - See candidate match scores and rankings
    """

    __tablename__ = 'recruiters'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(
        db.Integer,
        db.ForeignKey('users.id', ondelete='CASCADE'),
        unique=True,
        nullable=False
    )

    # Company Information
    company_name = db.Column(db.String(255), nullable=False)
    company_website = db.Column(db.String(500))
    company_logo = db.Column(db.String(500))
    industry = db.Column(db.String(100))
    company_size = db.Column(db.String(50))  # e.g., "1-10", "11-50", "51-200", "201-500", "500+"
    company_description = db.Column(db.Text)

    # Approval Status (admin must approve before recruiter can use platform)
    is_approved = db.Column(db.Boolean, default=False, nullable=False)

    # Recruiter Contact
    phone = db.Column(db.String(20))
    location = db.Column(db.String(255))

    # Relationships
    jobs = db.relationship(
        'Job', backref='recruiter', lazy='dynamic', cascade='all, delete-orphan'
    )

    @property
    def active_jobs_count(self):
        """Returns the number of currently active job postings."""
        return self.jobs.filter_by(is_active=True).count()

    def __repr__(self):
        return f'<Recruiter {self.company_name}>'
