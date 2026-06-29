"""
Resume Service
================
Business logic for resume upload, processing, and management.

Orchestrates the full resume processing pipeline:
1. Save uploaded file
2. Extract text (PDF/DOCX parser)
3. Clean text (NLP preprocessor)
4. Extract features (skills, education, experience)
5. Classify resume category (ML classifier)
6. Store structured data in database
7. Trigger recommendation generation
"""

import os
from flask import current_app
from app.extensions import db
from app.models.resume import (
    Resume, ResumeSkill, ResumeEducation, ResumeExperience,
    ResumeCertification, ResumeProject
)
from app.models.skill import Skill
from app.ml.resume_parser import resume_parser
from app.ml.text_preprocessor import text_preprocessor
from app.ml.feature_extractor import feature_extractor
from app.ml.resume_classifier import resume_classifier
from app.utils.validators import validate_resume_file, secure_filename_custom


def process_resume_upload(candidate, file):
    """
    Full resume upload and processing pipeline.

    Args:
        candidate: Candidate model instance.
        file: Flask FileStorage object from upload form.

    Returns:
        Tuple of (success: bool, resume_or_error: Resume | str)
    """
    # Step 0: Validate file
    is_valid, error = validate_resume_file(file)
    if not is_valid:
        return False, error

    try:
        # Step 1: Save file to disk
        filename = secure_filename_custom(file.filename)
        upload_dir = os.path.join(
            current_app.config['UPLOAD_FOLDER'], 'resumes'
        )
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, filename)
        file.save(file_path)

        file_ext = filename.rsplit('.', 1)[1].lower()

        # Step 2: Extract raw text
        raw_text = resume_parser.parse(file_path)
        if not raw_text or len(raw_text.strip()) < 20:
            return False, 'Could not extract text from the uploaded file. Please try a different file.'

        # Step 3: Clean and preprocess text
        cleaned_text = text_preprocessor.preprocess(raw_text)

        # Step 4: Extract features
        features = feature_extractor.extract_all(raw_text)

        # Step 5: Classify resume category
        # Load the model if not already loaded
        model_dir = current_app.config.get('MODEL_DIR', 'ml_models')
        if not resume_classifier.is_loaded:
            resume_classifier.load_model(model_dir)

        category, confidence = resume_classifier.predict(cleaned_text)

        # Step 6: Deactivate previous resumes
        Resume.query.filter_by(
            candidate_id=candidate.id,
            is_active=True
        ).update({'is_active': False})

        # Step 7: Create Resume record
        resume = Resume(
            candidate_id=candidate.id,
            file_path=file_path,
            file_name=filename,
            file_type=file_ext,
            raw_text=raw_text,
            cleaned_text=cleaned_text,
            classified_category=category,
            classification_confidence=confidence,
            parsed_data=features,
            is_active=True,
        )
        db.session.add(resume)
        db.session.flush()  # Get resume.id

        # Step 8: Store structured features in related tables
        _store_skills(resume, features.get('skills', []))
        _store_education(resume, features.get('education', []))
        _store_experience(resume, features.get('experience', []))
        _store_certifications(resume, features.get('certifications', []))
        _store_projects(resume, features.get('projects', []))

        db.session.commit()

        return True, resume

    except Exception as e:
        db.session.rollback()
        return False, f'Resume processing failed: {str(e)}'


def get_resume_data_for_matching(resume):
    """
    Prepare resume data in the format expected by the recommendation engine.

    Args:
        resume: Resume model instance.

    Returns:
        Dict with cleaned_text, skills, education, experience, category.
    """
    skills = [rs.skill.name for rs in resume.skills.all() if rs.skill]
    education = [
        {
            'degree': edu.degree,
            'field_of_study': edu.field_of_study,
            'institution': edu.institution,
            'gpa': edu.gpa,
        }
        for edu in resume.education.all()
    ]
    experience = [
        {
            'title': exp.title,
            'company': exp.company,
            'start_date': exp.start_date,
            'end_date': exp.end_date,
        }
        for exp in resume.experience.all()
    ]

    return {
        'cleaned_text': resume.cleaned_text or '',
        'skills': skills,
        'education': education,
        'experience': experience,
        'category': resume.classified_category,
    }


def _store_skills(resume, skills_list):
    """Store extracted skills in the database."""
    for skill_data in skills_list:
        skill_name = skill_data.get('name', '').strip()
        skill_category = skill_data.get('category', None)
        if skill_name:
            skill = Skill.get_or_create(skill_name, skill_category)
            db.session.flush()
            resume_skill = ResumeSkill(
                resume_id=resume.id,
                skill_id=skill.id,
                proficiency_level=None,
            )
            db.session.add(resume_skill)


def _store_education(resume, education_list):
    """Store extracted education entries."""
    for edu_data in education_list:
        edu = ResumeEducation(
            resume_id=resume.id,
            institution=edu_data.get('institution'),
            degree=edu_data.get('degree'),
            field_of_study=edu_data.get('field_of_study'),
            start_date=edu_data.get('start_date'),
            end_date=edu_data.get('end_date'),
            gpa=edu_data.get('gpa'),
        )
        db.session.add(edu)


def _store_experience(resume, experience_list):
    """Store extracted work experience entries."""
    for exp_data in experience_list:
        exp = ResumeExperience(
            resume_id=resume.id,
            company=exp_data.get('company'),
            title=exp_data.get('title'),
            start_date=exp_data.get('start_date'),
            end_date=exp_data.get('end_date'),
            description=exp_data.get('description'),
        )
        db.session.add(exp)


def _store_certifications(resume, cert_list):
    """Store extracted certifications."""
    for cert_data in cert_list:
        cert = ResumeCertification(
            resume_id=resume.id,
            name=cert_data.get('name', 'Unknown'),
            issuing_org=cert_data.get('issuing_org'),
            issue_date=cert_data.get('issue_date'),
        )
        db.session.add(cert)


def _store_projects(resume, project_list):
    """Store extracted projects."""
    for proj_data in project_list:
        proj = ResumeProject(
            resume_id=resume.id,
            name=proj_data.get('name', 'Unknown'),
            description=proj_data.get('description'),
            technologies=proj_data.get('technologies'),
        )
        db.session.add(proj)
