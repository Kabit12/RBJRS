"""
Notification Model
===================
Stores in-app notifications for users (e.g., new application alerts for recruiters).

Design Decisions:
- Notifications are stored in the database for persistence across sessions.
- Each notification has a link to the relevant page (e.g., the applicants view).
- is_read flag allows showing unread count badges.
- Notifications are linked to the User model (not Recruiter/Candidate)
  so the system can be extended to notify any user type.
"""

from datetime import datetime, timezone
from app.extensions import db


class Notification(db.Model):
    """
    In-app notification for a user.

    Created automatically when events occur (e.g., candidate applies for a job).
    Displayed in the recruiter dashboard with an unread count badge.
    """

    __tablename__ = 'notifications'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(
        db.Integer,
        db.ForeignKey('users.id', ondelete='CASCADE'),
        nullable=False,
        index=True
    )
    message = db.Column(db.String(500), nullable=False)
    link = db.Column(db.String(500))  # URL to navigate to when clicked
    is_read = db.Column(db.Boolean, default=False, nullable=False)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

    # Relationship to User
    user = db.relationship('User', backref=db.backref('notifications', lazy='dynamic'))

    def __repr__(self):
        status = 'read' if self.is_read else 'unread'
        return f'<Notification user={self.user_id} ({status})>'
