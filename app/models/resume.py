"""
Resume Models
==============
Represents uploaded resumes and all structured data extracted from them.

Maps to: RESUMES, RESUME_SKILLS, RESUME_EDUCATION, RESUME_EXPERIENCE,
         RESUME_CERTIFICATIONS, RESUME_PROJECTS tables in the ER diagram.

Design Decisions:
- The Resume model stores both the raw uploaded file reference AND the
  extracted/parsed data. This allows re-processing without re-uploading.
- raw_text: Full text extracted from PDF/DOCX (used for TF-IDF vectorization)
- cleaned_text: Preprocessed text after NLP pipeline (tokenized, lemmatized, stopwords removed)
- classified_category: The job domain predicted by the ML classifier (e.g., "Data Science")
- parsed_data: JSON blob with the complete structured extraction for flexibility
- Child tables (ResumeSkill, ResumeEducation, etc.) store normalized structured
  data for precise matching. These are separate tables (not just JSON) because
  the recommendation engine needs to do set operations and comparisons on them.
- is_active flag: A candidate may upload multiple resumes over time, but only
  one is "active" for recommendations at any given time.
"""

from datetime import datetime, timezone
from app.extensions import db


class Resume(db.Model):
    """
    Uploaded resume with extracted structured information.

    Lifecycle:
    1. Candidate uploads PDF/DOCX → file saved, raw_text extracted
    2. NLP pipeline cleans text → cleaned_text stored
    3. Feature extractor populates child tables (skills, education, etc.)
    4. ML classifier categorizes resume → classified_category set
    5. Recommendation engine uses all above for job matching
    """

    __tablename__ = 'resumes'

    id = db.Column(db.Integer, primary_key=True)
    candidate_id = db.Column(
        db.Integer,
        db.ForeignKey('candidates.id', ondelete='CASCADE'),
        nullable=False,
        index=True
    )

    # File metadata
    file_path = db.Column(db.String(500), nullable=False)
    file_name = db.Column(db.String(255), nullable=False)
    file_type = db.Column(db.String(10), nullable=False)  # 'pdf' or 'docx'

    # Extracted text
    raw_text = db.Column(db.Text)       # Full text as extracted from document
    cleaned_text = db.Column(db.Text)   # After NLP preprocessing

    # ML Classification
    classified_category = db.Column(db.String(100))  # Predicted job domain
    classification_confidence = db.Column(db.Float)   # Model confidence score

    # Complete parsed data as JSON (backup/flexibility)
    parsed_data = db.Column(db.JSON)

    # Status
    is_active = db.Column(db.Boolean, default=True, nullable=False)
    uploaded_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

    # Relationships to structured extraction tables
    skills = db.relationship(
        'ResumeSkill', backref='resume', lazy='dynamic', cascade='all, delete-orphan'
    )
    education = db.relationship(
        'ResumeEducation', backref='resume', lazy='dynamic', cascade='all, delete-orphan'
    )
    experience = db.relationship(
        'ResumeExperience', backref='resume', lazy='dynamic', cascade='all, delete-orphan'
    )
    certifications = db.relationship(
        'ResumeCertification', backref='resume', lazy='dynamic', cascade='all, delete-orphan'
    )
    projects = db.relationship(
        'ResumeProject', backref='resume', lazy='dynamic', cascade='all, delete-orphan'
    )

    def __repr__(self):
        return f'<Resume {self.file_name} ({self.classified_category or "unclassified"})>'


class ResumeSkill(db.Model):
    """
    Junction table linking a resume to its extracted skills.
    Includes proficiency level when detectable from resume context.
    """

    __tablename__ = 'resume_skills'

    id = db.Column(db.Integer, primary_key=True)
    resume_id = db.Column(
        db.Integer,
        db.ForeignKey('resumes.id', ondelete='CASCADE'),
        nullable=False,
        index=True
    )
    skill_id = db.Column(
        db.Integer,
        db.ForeignKey('skills.id', ondelete='CASCADE'),
        nullable=False
    )
    proficiency_level = db.Column(db.String(20))  # "beginner", "intermediate", "expert"

    # Relationship to Skill model (needed for template access: rs.skill.name)
    skill = db.relationship('Skill', backref='resume_skills')

    # Ensure no duplicate skill entries per resume
    __table_args__ = (
        db.UniqueConstraint('resume_id', 'skill_id', name='uq_resume_skill'),
    )

    def __repr__(self):
        return f'<ResumeSkill resume={self.resume_id} skill={self.skill_id}>'


class ResumeEducation(db.Model):
    """
    Educational qualifications extracted from a resume.
    Each record represents one degree/qualification.
    """

    __tablename__ = 'resume_education'

    id = db.Column(db.Integer, primary_key=True)
    resume_id = db.Column(
        db.Integer,
        db.ForeignKey('resumes.id', ondelete='CASCADE'),
        nullable=False,
        index=True
    )
    institution = db.Column(db.String(255))
    degree = db.Column(db.String(255))       # e.g., "Bachelor of Science"
    field_of_study = db.Column(db.String(255))  # e.g., "Computer Science"
    start_date = db.Column(db.String(50))
    end_date = db.Column(db.String(50))
    gpa = db.Column(db.Float)

    def __repr__(self):
        return f'<ResumeEducation {self.degree} in {self.field_of_study}>'


class ResumeExperience(db.Model):
    """
    Work experience entries extracted from a resume.
    Each record represents one job position.
    """

    __tablename__ = 'resume_experience'

    id = db.Column(db.Integer, primary_key=True)
    resume_id = db.Column(
        db.Integer,
        db.ForeignKey('resumes.id', ondelete='CASCADE'),
        nullable=False,
        index=True
    )
    company = db.Column(db.String(255))
    title = db.Column(db.String(255))        # Job title
    start_date = db.Column(db.String(50))
    end_date = db.Column(db.String(50))      # None = "Present"
    description = db.Column(db.Text)

    def __repr__(self):
        return f'<ResumeExperience {self.title} at {self.company}>'


class ResumeCertification(db.Model):
    """
    Professional certifications extracted from a resume.
    """

    __tablename__ = 'resume_certifications'

    id = db.Column(db.Integer, primary_key=True)
    resume_id = db.Column(
        db.Integer,
        db.ForeignKey('resumes.id', ondelete='CASCADE'),
        nullable=False,
        index=True
    )
    name = db.Column(db.String(255), nullable=False)
    issuing_org = db.Column(db.String(255))
    issue_date = db.Column(db.String(50))

    def __repr__(self):
        return f'<ResumeCertification {self.name}>'


class ResumeProject(db.Model):
    """
    Projects extracted from a resume.
    """

    __tablename__ = 'resume_projects'

    id = db.Column(db.Integer, primary_key=True)
    resume_id = db.Column(
        db.Integer,
        db.ForeignKey('resumes.id', ondelete='CASCADE'),
        nullable=False,
        index=True
    )
    name = db.Column(db.String(255), nullable=False)
    description = db.Column(db.Text)
    technologies = db.Column(db.String(500))  # Comma-separated tech stack

    def __repr__(self):
        return f'<ResumeProject {self.name}>'
