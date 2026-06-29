"""
Recruiter Routes
==================
Handles all recruiter-facing features:
- Dashboard overview
- Job posting CRUD (create, read, update, delete)
- Applicant viewing
- Candidate ranking/scores
- Company profile management
"""

from flask import render_template, redirect, url_for, flash, request, abort
from flask_login import login_required, current_user
from app.blueprints.recruiter import recruiter_bp
from app.utils.decorators import recruiter_required
from app.extensions import db
from app.models.job import Job
from app.models.application import Application
from app.services.job_service import create_job, update_job, delete_job
from app.services.recommendation_service import get_ranked_candidates_for_job


@recruiter_bp.route('/dashboard')
@login_required
@recruiter_required
def dashboard():
    """Recruiter dashboard — overview of jobs, applicants, and metrics."""
    recruiter = current_user.recruiter_profile
    jobs = recruiter.jobs.order_by(Job.posted_at.desc()).limit(5).all()

    total_jobs = recruiter.jobs.count()
    active_jobs = recruiter.jobs.filter_by(is_active=True).count()
    total_applications = sum(job.applicant_count for job in recruiter.jobs.all())

    return render_template(
        'recruiter/dashboard.html',
        recruiter=recruiter,
        jobs=jobs,
        total_jobs=total_jobs,
        active_jobs=active_jobs,
        total_applications=total_applications,
    )


@recruiter_bp.route('/jobs')
@login_required
@recruiter_required
def manage_jobs():
    """View and manage all job postings."""
    recruiter = current_user.recruiter_profile
    jobs = recruiter.jobs.order_by(Job.posted_at.desc()).all()

    return render_template('recruiter/jobs.html', jobs=jobs)


@recruiter_bp.route('/jobs/create', methods=['GET', 'POST'])
@login_required
@recruiter_required
def create_job_view():
    """Create a new job posting."""
    if request.method == 'POST':
        form_data = {
            'title': request.form.get('title', ''),
            'description': request.form.get('description', ''),
            'location': request.form.get('location', ''),
            'job_type': request.form.get('job_type', 'full-time'),
            'experience_level': request.form.get('experience_level', ''),
            'salary_range': request.form.get('salary_range', ''),
            'requirements': request.form.get('requirements', ''),
            'responsibilities': request.form.get('responsibilities', ''),
            'skills': request.form.get('skills', ''),
        }

        if not form_data['title'] or not form_data['description']:
            flash('Title and description are required.', 'danger')
            return render_template('recruiter/job_form.html', form_data=form_data, is_edit=False)

        recruiter = current_user.recruiter_profile
        success, result = create_job(recruiter, form_data)

        if success:
            flash('Job posted successfully!', 'success')
            return redirect(url_for('recruiter.manage_jobs'))
        else:
            flash(result, 'danger')

    return render_template('recruiter/job_form.html', form_data={}, is_edit=False)


@recruiter_bp.route('/jobs/<int:job_id>/edit', methods=['GET', 'POST'])
@login_required
@recruiter_required
def edit_job(job_id):
    """Edit an existing job posting."""
    job = db.session.get(Job, job_id)
    if not job or job.recruiter_id != current_user.recruiter_profile.id:
        abort(404)

    if request.method == 'POST':
        form_data = {
            'title': request.form.get('title', ''),
            'description': request.form.get('description', ''),
            'location': request.form.get('location', ''),
            'job_type': request.form.get('job_type', 'full-time'),
            'experience_level': request.form.get('experience_level', ''),
            'salary_range': request.form.get('salary_range', ''),
            'requirements': request.form.get('requirements', ''),
            'responsibilities': request.form.get('responsibilities', ''),
            'skills': request.form.get('skills', ''),
        }

        success, result = update_job(job, form_data)
        if success:
            flash('Job updated successfully!', 'success')
            return redirect(url_for('recruiter.manage_jobs'))
        else:
            flash(result, 'danger')

    # Populate form data from existing job
    existing_skills = ', '.join([
        js.skill.name for js in job.required_skills.all() if js.skill
    ])
    form_data = {
        'title': job.title,
        'description': job.description,
        'location': job.location or '',
        'job_type': job.job_type,
        'experience_level': job.experience_level or '',
        'salary_range': job.salary_range or '',
        'requirements': job.requirements or '',
        'responsibilities': job.responsibilities or '',
        'skills': existing_skills,
    }

    return render_template('recruiter/job_form.html', form_data=form_data, is_edit=True, job=job)


@recruiter_bp.route('/jobs/<int:job_id>/delete', methods=['POST'])
@login_required
@recruiter_required
def delete_job_view(job_id):
    """Delete a job posting."""
    job = db.session.get(Job, job_id)
    if not job or job.recruiter_id != current_user.recruiter_profile.id:
        abort(404)

    success, msg = delete_job(job)
    flash(msg, 'success' if success else 'danger')
    return redirect(url_for('recruiter.manage_jobs'))


@recruiter_bp.route('/jobs/<int:job_id>/applicants')
@login_required
@recruiter_required
def view_applicants(job_id):
    """View applicants for a specific job."""
    job = db.session.get(Job, job_id)
    if not job or job.recruiter_id != current_user.recruiter_profile.id:
        abort(404)

    applications = (
        Application.query
        .filter_by(job_id=job_id)
        .order_by(Application.match_score.desc().nullslast())
        .all()
    )

    # Get ranked candidates
    ranked_candidates = get_ranked_candidates_for_job(job, top_n=50)

    return render_template(
        'recruiter/applicants.html',
        job=job,
        applications=applications,
        ranked_candidates=ranked_candidates,
    )


@recruiter_bp.route('/applications/<int:app_id>/status', methods=['POST'])
@login_required
@recruiter_required
def update_application_status(app_id):
    """Update application status."""
    application = db.session.get(Application, app_id)
    if not application:
        abort(404)

    # Verify the job belongs to this recruiter
    job = db.session.get(Job, application.job_id)
    if not job or job.recruiter_id != current_user.recruiter_profile.id:
        abort(403)

    new_status = request.form.get('status')
    if new_status in ['pending', 'reviewed', 'shortlisted', 'rejected', 'accepted']:
        application.status = new_status
        db.session.commit()
        flash(f'Application status updated to {new_status}.', 'success')
    else:
        flash('Invalid status.', 'danger')

    return redirect(url_for('recruiter.view_applicants', job_id=application.job_id))


@recruiter_bp.route('/profile', methods=['GET', 'POST'])
@login_required
@recruiter_required
def profile():
    """Edit recruiter/company profile."""
    recruiter = current_user.recruiter_profile

    if request.method == 'POST':
        current_user.first_name = request.form.get('first_name', '').strip()
        current_user.last_name = request.form.get('last_name', '').strip()
        recruiter.company_name = request.form.get('company_name', '').strip()
        recruiter.company_website = request.form.get('company_website', '').strip() or None
        recruiter.industry = request.form.get('industry', '').strip() or None
        recruiter.company_size = request.form.get('company_size', '').strip() or None
        recruiter.company_description = request.form.get('company_description', '').strip() or None
        recruiter.phone = request.form.get('phone', '').strip() or None
        recruiter.location = request.form.get('location', '').strip() or None

        db.session.commit()
        flash('Profile updated successfully!', 'success')
        return redirect(url_for('recruiter.profile'))

    return render_template('recruiter/profile.html', recruiter=recruiter)
