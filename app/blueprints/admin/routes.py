"""
Admin Routes
==============
Handles admin dashboard, user management, job management, and analytics.
"""

from flask import render_template, redirect, url_for, flash, request, abort
from flask_login import login_required, current_user
from app.blueprints.admin import admin_bp
from app.utils.decorators import admin_required
from app.extensions import db
from app.models.user import User
from app.models.job import Job
from app.models.application import Application
from app.services.analytics_service import (
    get_platform_stats,
    get_application_stats,
    get_job_category_stats,
    get_recent_users,
    get_recent_jobs,
    get_recent_applications,
)


@admin_bp.route('/dashboard')
@login_required
@admin_required
def dashboard():
    """Admin dashboard with platform overview."""
    stats = get_platform_stats()
    app_stats = get_application_stats()
    category_stats = get_job_category_stats()
    recent_users = get_recent_users(5)
    recent_jobs = get_recent_jobs(5)

    return render_template(
        'admin/dashboard.html',
        stats=stats,
        app_stats=app_stats,
        category_stats=category_stats,
        recent_users=recent_users,
        recent_jobs=recent_jobs,
    )


@admin_bp.route('/users')
@login_required
@admin_required
def manage_users():
    """Manage all users."""
    page = request.args.get('page', 1, type=int)
    role_filter = request.args.get('role', '')

    query = User.query
    if role_filter:
        query = query.filter_by(role=role_filter)

    users = query.order_by(User.created_at.desc()).paginate(
        page=page, per_page=20, error_out=False
    )

    return render_template('admin/users.html', users=users, role_filter=role_filter)


@admin_bp.route('/users/<int:user_id>/toggle', methods=['POST'])
@login_required
@admin_required
def toggle_user(user_id):
    """Activate/deactivate a user account."""
    user = db.session.get(User, user_id)
    if not user:
        abort(404)
    if user.id == current_user.id:
        flash('You cannot deactivate your own account.', 'danger')
        return redirect(url_for('admin.manage_users'))

    user.is_active = not user.is_active
    db.session.commit()
    status = 'activated' if user.is_active else 'deactivated'
    flash(f'User {user.email} has been {status}.', 'success')
    return redirect(url_for('admin.manage_users'))


@admin_bp.route('/users/<int:user_id>/delete', methods=['POST'])
@login_required
@admin_required
def delete_user(user_id):
    """Delete a user account."""
    user = db.session.get(User, user_id)
    if not user:
        abort(404)
    if user.id == current_user.id:
        flash('You cannot delete your own account.', 'danger')
        return redirect(url_for('admin.manage_users'))

    email = user.email
    db.session.delete(user)
    db.session.commit()
    flash(f'User {email} has been deleted.', 'success')
    return redirect(url_for('admin.manage_users'))


@admin_bp.route('/jobs')
@login_required
@admin_required
def manage_jobs():
    """Manage all job postings."""
    page = request.args.get('page', 1, type=int)
    jobs = (
        Job.query
        .order_by(Job.posted_at.desc())
        .paginate(page=page, per_page=20, error_out=False)
    )

    return render_template('admin/jobs.html', jobs=jobs)


@admin_bp.route('/jobs/<int:job_id>/toggle', methods=['POST'])
@login_required
@admin_required
def toggle_job(job_id):
    """Activate/deactivate a job posting."""
    job = db.session.get(Job, job_id)
    if not job:
        abort(404)

    job.is_active = not job.is_active
    db.session.commit()
    status = 'activated' if job.is_active else 'deactivated'
    flash(f'Job "{job.title}" has been {status}.', 'success')
    return redirect(url_for('admin.manage_jobs'))


@admin_bp.route('/recruiters/<int:user_id>/approve', methods=['POST'])
@login_required
@admin_required
def approve_recruiter(user_id):
    """Approve a recruiter account."""
    user = db.session.get(User, user_id)
    if not user or user.role != 'recruiter':
        abort(404)

    recruiter = user.recruiter_profile
    if recruiter:
        recruiter.is_approved = True
        db.session.commit()
        flash(f'Recruiter {user.full_name} ({recruiter.company_name}) has been approved.', 'success')
    else:
        flash('Recruiter profile not found.', 'danger')

    return redirect(url_for('admin.manage_users', role='recruiter'))


@admin_bp.route('/recruiters/<int:user_id>/disapprove', methods=['POST'])
@login_required
@admin_required
def disapprove_recruiter(user_id):
    """Disapprove (revoke approval) a recruiter account."""
    user = db.session.get(User, user_id)
    if not user or user.role != 'recruiter':
        abort(404)

    recruiter = user.recruiter_profile
    if recruiter:
        recruiter.is_approved = False
        db.session.commit()
        flash(f'Recruiter {user.full_name} ({recruiter.company_name}) has been disapproved.', 'warning')
    else:
        flash('Recruiter profile not found.', 'danger')

    return redirect(url_for('admin.manage_users', role='recruiter'))


@admin_bp.route('/analytics')
@login_required
@admin_required
def analytics():
    """Analytics and reports page."""
    stats = get_platform_stats()
    app_stats = get_application_stats()
    category_stats = get_job_category_stats()

    return render_template(
        'admin/analytics.html',
        stats=stats,
        app_stats=app_stats,
        category_stats=category_stats,
    )
