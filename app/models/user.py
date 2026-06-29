"""
User Model
===========
Central authentication entity for the system. Every person who interacts
with the platform (candidate, recruiter, or admin) has a User record.

Maps to: USERS table in the ER diagram.

Design Decisions:
- Single users table with a 'role' enum discriminator rather than separate tables
  per role type. This simplifies authentication (one login flow) while allowing
  role-specific data via one-to-one relationships (Candidate, Recruiter profiles).
- Password is stored as a bcrypt hash (never plaintext).
- Email is unique and serves as the login identifier.
- is_active flag allows soft-disabling accounts without deletion.
"""

from datetime import datetime, timezone
from app.extensions import db, login_manager
from flask_login import UserMixin


class User(UserMixin, db.Model):
    """
    Core user entity for authentication and authorization.

    Roles:
        - 'candidate': Job seekers who upload resumes and receive recommendations
        - 'recruiter': Employers who post jobs and review candidates
        - 'admin': System administrators with full platform access

    UserMixin provides:
        - is_authenticated: True if user has valid credentials
        - is_active: True if account is not disabled
        - is_anonymous: False for real users
        - get_id(): Returns user ID as string for session management
    """

    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    first_name = db.Column(db.String(100), nullable=False)
    last_name = db.Column(db.String(100), nullable=False)
    role = db.Column(
        db.Enum('candidate', 'recruiter', 'admin', name='user_role'),
        nullable=False,
        default='candidate'
    )
    is_active = db.Column(db.Boolean, default=True, nullable=False)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = db.Column(
        db.DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc)
    )

    # Relationships — one-to-one with role-specific profiles
    candidate_profile = db.relationship(
        'Candidate', backref='user', uselist=False, cascade='all, delete-orphan'
    )
    recruiter_profile = db.relationship(
        'Recruiter', backref='user', uselist=False, cascade='all, delete-orphan'
    )

    @property
    def full_name(self):
        """Returns the user's full name."""
        return f'{self.first_name} {self.last_name}'

    @property
    def is_candidate(self):
        return self.role == 'candidate'

    @property
    def is_recruiter(self):
        return self.role == 'recruiter'

    @property
    def is_admin(self):
        return self.role == 'admin'

    def __repr__(self):
        return f'<User {self.email} ({self.role})>'


@login_manager.user_loader
def load_user(user_id):
    """
    Flask-Login callback to reload user from session.
    Called on every request to populate current_user.
    """
    return db.session.get(User, int(user_id))
