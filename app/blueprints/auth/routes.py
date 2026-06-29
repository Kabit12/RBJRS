"""
Authentication Routes
======================
Handles registration, login, logout, and authentication redirects.

URL Structure:
    /auth/login           — Login page (GET/POST)
    /auth/register        — Registration type selection (GET)
    /auth/register/candidate — Candidate registration (GET/POST)
    /auth/register/recruiter — Recruiter registration (GET/POST)
    /auth/logout          — Logout (GET)
"""

from flask import render_template, redirect, url_for, flash, request
from flask_login import login_user, logout_user, login_required, current_user
from app.blueprints.auth import auth_bp
from app.blueprints.auth.forms import (
    LoginForm,
    CandidateRegistrationForm,
    RecruiterRegistrationForm
)
from app.services.auth_service import (
    authenticate_user,
    register_candidate,
    register_recruiter
)


@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    """Handle user login."""
    # Redirect if already logged in
    if current_user.is_authenticated:
        return _redirect_to_dashboard(current_user)

    form = LoginForm()

    if form.validate_on_submit():
        success, result = authenticate_user(form.email.data, form.password.data)

        if success:
            user = result
            login_user(user, remember=form.remember_me.data)
            flash(f'Welcome back, {user.first_name}!', 'success')

            # Redirect to the page they were trying to access, or dashboard
            next_page = request.args.get('next')
            if next_page:
                return redirect(next_page)
            return _redirect_to_dashboard(user)
        else:
            flash(result, 'danger')

    return render_template('auth/login.html', form=form)


@auth_bp.route('/register')
def register():
    """Registration type selection page."""
    if current_user.is_authenticated:
        return _redirect_to_dashboard(current_user)
    return render_template('auth/register_choice.html')


@auth_bp.route('/register/candidate', methods=['GET', 'POST'])
def register_candidate_view():
    """Handle candidate registration."""
    if current_user.is_authenticated:
        return _redirect_to_dashboard(current_user)

    form = CandidateRegistrationForm()

    if form.validate_on_submit():
        form_data = {
            'email': form.email.data,
            'password': form.password.data,
            'first_name': form.first_name.data,
            'last_name': form.last_name.data,
            'phone': form.phone.data,
            'location': form.location.data,
        }
        success, result = register_candidate(form_data)

        if success:
            user = result
            login_user(user)
            flash('Account created successfully! Welcome to RBJRS.', 'success')
            return redirect(url_for('candidate.dashboard'))
        else:
            flash(result, 'danger')

    return render_template('auth/register_candidate.html', form=form)


@auth_bp.route('/register/recruiter', methods=['GET', 'POST'])
def register_recruiter_view():
    """Handle recruiter registration."""
    if current_user.is_authenticated:
        return _redirect_to_dashboard(current_user)

    form = RecruiterRegistrationForm()

    if form.validate_on_submit():
        form_data = {
            'email': form.email.data,
            'password': form.password.data,
            'first_name': form.first_name.data,
            'last_name': form.last_name.data,
            'company_name': form.company_name.data,
            'industry': form.industry.data,
            'company_size': form.company_size.data,
            'phone': form.phone.data,
            'location': form.location.data,
        }
        success, result = register_recruiter(form_data)

        if success:
            user = result
            login_user(user)
            flash('Account created successfully! Welcome to RBJRS.', 'success')
            return redirect(url_for('recruiter.dashboard'))
        else:
            flash(result, 'danger')

    return render_template('auth/register_recruiter.html', form=form)


@auth_bp.route('/logout')
@login_required
def logout():
    """Log out the current user."""
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('main.index'))


def _redirect_to_dashboard(user):
    """Redirect user to their role-specific dashboard."""
    if user.role == 'candidate':
        return redirect(url_for('candidate.dashboard'))
    elif user.role == 'recruiter':
        return redirect(url_for('recruiter.dashboard'))
    elif user.role == 'admin':
        return redirect(url_for('admin.dashboard'))
    return redirect(url_for('main.index'))
