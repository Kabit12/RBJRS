"""
Skill Model
=============
Master table of recognized skills used for structured skill matching.

Maps to: SKILLS table in the ER diagram.

Design Decisions:
- A normalized skills table (rather than free-text skill fields) enables:
  1. Consistent skill matching between resumes and jobs
  2. Accurate Jaccard similarity computation
  3. Skill analytics across the platform
  4. Auto-suggestion during job posting
- Skills are categorized (e.g., "Programming Language", "Framework", "Soft Skill")
  to support weighted matching in the recommendation engine.
- The name field is unique and indexed for fast lookups during parsing.
"""

from app.extensions import db


class Skill(db.Model):
    """
    Master skill entity for structured skill matching.

    Skills are extracted from resumes and job descriptions during parsing,
    then linked via junction tables (ResumeSkill, JobSkill) to enable
    set-based similarity computations.
    """

    __tablename__ = 'skills'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), unique=True, nullable=False, index=True)
    category = db.Column(db.String(50))  # e.g., "programming", "framework", "database", "soft_skill"

    def __repr__(self):
        return f'<Skill {self.name}>'

    @staticmethod
    def get_or_create(name, category=None):
        """
        Returns an existing skill or creates a new one.
        Ensures skill names are always stored in lowercase for consistency.
        """
        normalized_name = name.strip().lower()
        skill = Skill.query.filter_by(name=normalized_name).first()
        if not skill:
            skill = Skill(name=normalized_name, category=category)
            db.session.add(skill)
        return skill
