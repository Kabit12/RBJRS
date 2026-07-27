"""
Candidate Routes
==================
Handles all candidate-facing features:
- Dashboard overview
- Resume upload and management (multiple resumes, delete, switch active)
- Job recommendations
- Job applications (with cover letter file upload)
- Application history
- Profile management

All routes require authenticated candidate role access.
"""

import os
from flask import render_template, redirect, url_for, flash, request, abort, current_app
from flask_login import login_required, current_user
from app.blueprints.candidate import candidate_bp
from app.utils.decorators import candidate_required
from app.extensions import db
from app.models.resume import Resume
from app.models.job import Job
from app.models.application import Application
from app.models.recommendation import Recommendation
from app.models.notification import Notification
from app.services.resume_service import process_resume_upload
from app.services.recommendation_service import (
    generate_recommendations,
    get_candidate_recommendations
)
from app.extensions import limiter


@candidate_bp.route('/dashboard')
@login_required
@candidate_required
def dashboard():
    """Candidate dashboard — overview of resume status, recommendations, and applications."""
    candidate = current_user.candidate_profile
    active_resume = candidate.active_resume if candidate else None
    recommendations = []
    recent_applications = []

    if candidate:
        recommendations = get_candidate_recommendations(candidate)[:5]
        recent_applications = (
            Application.query
            .filter_by(candidate_id=candidate.id)
            .order_by(Application.applied_at.desc())
            .limit(5)
            .all()
        )

    return render_template(
        'candidate/dashboard.html',
        candidate=candidate,
        active_resume=active_resume,
        recommendations=recommendations,
        recent_applications=recent_applications,
    )


@candidate_bp.route('/resume', methods=['GET', 'POST'])
@login_required
@candidate_required
@limiter.limit('10 per hour', methods=['POST'])
def resume():
    """Resume upload and management page."""
    candidate = current_user.candidate_profile

    if request.method == 'POST':
        file = request.files.get('resume')
        if not file:
            flash('Please select a file to upload.', 'danger')
            return redirect(url_for('candidate.resume'))

        target_role = request.form.get('target_role', '').strip() or None

        success, result = process_resume_upload(candidate, file, target_role=target_role)
        if success:
            flash('Resume uploaded and processed successfully!', 'success')
            # Auto-generate recommendations
            rec_success, rec_result = generate_recommendations(candidate, result)
            if rec_success:
                flash(f'{rec_result} job recommendations generated!', 'info')
            return redirect(url_for('candidate.resume'))
        else:
            flash(result, 'danger')

    active_resume = candidate.active_resume
    all_resumes = candidate.resumes.order_by(Resume.uploaded_at.desc()).all()

    return render_template(
        'candidate/resume.html',
        candidate=candidate,
        active_resume=active_resume,
        all_resumes=all_resumes,
    )


@candidate_bp.route('/resume/<int:resume_id>/set-active', methods=['POST'])
@login_required
@candidate_required
def set_active_resume(resume_id):
    """Set a specific resume as the active one."""
    candidate = current_user.candidate_profile
    target_resume = db.session.get(Resume, resume_id)

    if not target_resume or target_resume.candidate_id != candidate.id:
        abort(404)

    # Deactivate all other resumes
    Resume.query.filter_by(
        candidate_id=candidate.id, is_active=True
    ).update({'is_active': False})

    # Activate the selected resume
    target_resume.is_active = True
    db.session.commit()

    flash(f'Resume "{target_resume.file_name}" is now active.', 'success')

    # Regenerate recommendations with the new active resume
    rec_success, rec_result = generate_recommendations(candidate, target_resume)
    if rec_success:
        flash(f'Recommendations refreshed! {rec_result} jobs matched.', 'info')

    return redirect(url_for('candidate.resume'))


@candidate_bp.route('/resume/<int:resume_id>/delete', methods=['POST'])
@login_required
@candidate_required
def delete_resume(resume_id):
    """Delete a resume from history."""
    candidate = current_user.candidate_profile
    target_resume = db.session.get(Resume, resume_id)

    if not target_resume or target_resume.candidate_id != candidate.id:
        abort(404)

    was_active = target_resume.is_active
    file_name = target_resume.file_name

    # Delete the file from disk
    if target_resume.file_path and os.path.exists(target_resume.file_path):
        try:
            os.remove(target_resume.file_path)
        except OSError:
            pass

    # Delete from database
    db.session.delete(target_resume)
    db.session.commit()

    flash(f'Resume "{file_name}" has been deleted.', 'success')

    # If deleted resume was active, activate the most recent remaining one
    if was_active:
        latest = candidate.resumes.order_by(Resume.uploaded_at.desc()).first()
        if latest:
            latest.is_active = True
            db.session.commit()
            flash(f'Resume "{latest.file_name}" is now active.', 'info')

    return redirect(url_for('candidate.resume'))


@candidate_bp.route('/recommendations')
@login_required
@candidate_required
def recommendations():
    """View all job recommendations."""
    candidate = current_user.candidate_profile
    recs = get_candidate_recommendations(candidate)

    return render_template(
        'candidate/recommendations.html',
        recommendations=recs,
    )


@candidate_bp.route('/recommendations/refresh')
@login_required
@candidate_required
def refresh_recommendations():
    """Regenerate recommendations."""
    candidate = current_user.candidate_profile
    success, result = generate_recommendations(candidate)
    if success:
        flash(f'Recommendations refreshed! {result} jobs matched.', 'success')
    else:
        flash(result, 'warning')
    return redirect(url_for('candidate.recommendations'))


@candidate_bp.route('/jobs')
@login_required
@candidate_required
def browse_jobs():
    """Browse all active job listings."""
    page = request.args.get('page', 1, type=int)
    per_page = 12

    jobs = (
        Job.query
        .filter_by(is_active=True)
        .order_by(Job.posted_at.desc())
        .paginate(page=page, per_page=per_page, error_out=False)
    )

    return render_template('candidate/jobs.html', jobs=jobs)


@candidate_bp.route('/jobs/<int:job_id>')
@login_required
@candidate_required
def job_detail(job_id):
    """View job details with match score."""
    job = db.session.get(Job, job_id)
    if not job:
        abort(404)

    candidate = current_user.candidate_profile
    recommendation = Recommendation.query.filter_by(
        candidate_id=candidate.id, job_id=job_id
    ).first()
    existing_application = Application.query.filter_by(
        candidate_id=candidate.id, job_id=job_id
    ).first()

    # Get all resumes for the resume selector in the apply form
    all_resumes = candidate.resumes.order_by(Resume.uploaded_at.desc()).all()

    return render_template(
        'candidate/job_detail.html',
        job=job,
        recommendation=recommendation,
        existing_application=existing_application,
        all_resumes=all_resumes,
    )


@candidate_bp.route('/jobs/<int:job_id>/apply', methods=['POST'])
@login_required
@candidate_required
@limiter.limit('20 per hour')
def apply_job(job_id):
    """Apply for a job with optional cover letter text + file upload."""
    candidate = current_user.candidate_profile
    job = db.session.get(Job, job_id)
    if not job:
        abort(404)

    # Determine which resume to use
    selected_resume_id = request.form.get('resume_id', type=int)
    if selected_resume_id:
        selected_resume = db.session.get(Resume, selected_resume_id)
        if not selected_resume or selected_resume.candidate_id != candidate.id:
            selected_resume = candidate.active_resume
    else:
        selected_resume = candidate.active_resume

    # Get match score from recommendations
    recommendation = Recommendation.query.filter_by(
        candidate_id=candidate.id, job_id=job_id
    ).first()

    # Delegate to application service (handles DB, files, notifications)
    from app.services.application_service import create_application
    success, result = create_application(
        candidate=candidate,
        job=job,
        resume=selected_resume,
        cover_letter_text=request.form.get('cover_letter', ''),
        cover_letter_file=request.files.get('cover_letter_file'),
        match_score=recommendation.overall_score if recommendation else None,
        score_breakdown=recommendation.score_breakdown if recommendation else None,
    )

    if success:
        flash('Application submitted successfully!', 'success')
    else:
        flash(result, 'warning')

    return redirect(url_for('candidate.applications'))


@candidate_bp.route('/applications')
@login_required
@candidate_required
def applications():
    """View application history."""
    candidate = current_user.candidate_profile
    apps = (
        Application.query
        .filter_by(candidate_id=candidate.id)
        .order_by(Application.applied_at.desc())
        .all()
    )

    return render_template('candidate/applications.html', applications=apps)


@candidate_bp.route('/profile', methods=['GET', 'POST'])
@login_required
@candidate_required
def profile():
    """Edit candidate profile."""
    candidate = current_user.candidate_profile

    if request.method == 'POST':
        current_user.first_name = request.form.get('first_name', '').strip()
        current_user.last_name = request.form.get('last_name', '').strip()
        candidate.phone = request.form.get('phone', '').strip() or None
        candidate.location = request.form.get('location', '').strip() or None
        candidate.headline = request.form.get('headline', '').strip() or None
        candidate.bio = request.form.get('bio', '').strip() or None
        candidate.linkedin_url = request.form.get('linkedin_url', '').strip() or None
        candidate.portfolio_url = request.form.get('portfolio_url', '').strip() or None

        db.session.commit()
        flash('Profile updated successfully!', 'success')
        return redirect(url_for('candidate.profile'))

    return render_template('candidate/profile.html', candidate=candidate)
