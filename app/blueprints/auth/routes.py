from flask import render_template, redirect, url_for, flash, request, jsonify, current_app
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
    register_recruiter,
    register_or_login_google_user
)
from app.extensions import limiter


@auth_bp.route('/login', methods=['GET', 'POST'])
@limiter.limit('5 per minute')
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

    google_client_id = current_app.config.get('GOOGLE_CLIENT_ID', '')
    return render_template('auth/login.html', form=form, google_client_id=google_client_id)


@auth_bp.route('/register')
def register():
    """Registration type selection page."""
    if current_user.is_authenticated:
        return _redirect_to_dashboard(current_user)
    return render_template('auth/register_choice.html')


@auth_bp.route('/register/candidate', methods=['GET', 'POST'])
@limiter.limit('3 per minute')
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

    google_client_id = current_app.config.get('GOOGLE_CLIENT_ID', '')
    return render_template('auth/register_candidate.html', form=form, google_client_id=google_client_id)


@auth_bp.route('/register/recruiter', methods=['GET', 'POST'])
@limiter.limit('3 per minute')
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
            flash('Account created successfully! Your account is pending admin approval.', 'info')
            return redirect(url_for('recruiter.dashboard'))
        else:
            flash(result, 'danger')

    return render_template('auth/register_recruiter.html', form=form)


@auth_bp.route('/google-callback', methods=['POST'])
@limiter.limit('10 per minute')
def google_callback():
    """
    Handle Google Sign-In callback.

    Receives the Google ID token from the client-side GIS library,
    verifies it server-side, and logs in or registers the user.
    """
    token = request.form.get('credential', '')

    if not token:
        flash('Google sign-in failed: no token received.', 'danger')
        return redirect(url_for('auth.login'))

    try:
        from google.oauth2 import id_token
        from google.auth.transport import requests as google_requests

        client_id = current_app.config.get('GOOGLE_CLIENT_ID', '')
        if not client_id:
            flash('Google sign-in is not configured. Please contact the administrator.', 'danger')
            return redirect(url_for('auth.login'))

        # Verify the ID token
        idinfo = id_token.verify_oauth2_token(
            token,
            google_requests.Request(),
            client_id
        )

        google_user_info = {
            'email': idinfo['email'],
            'given_name': idinfo.get('given_name', 'User'),
            'family_name': idinfo.get('family_name', ''),
        }

        success, result = register_or_login_google_user(google_user_info)
        if success:
            user = result
            login_user(user, remember=True)
            flash(f'Welcome, {user.first_name}!', 'success')
            return _redirect_to_dashboard(user)
        else:
            flash(result, 'danger')

    except ValueError as e:
        flash(f'Google sign-in failed: invalid token.', 'danger')
    except Exception as e:
        flash(f'Google sign-in error: {str(e)}', 'danger')

    return redirect(url_for('auth.login'))


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
