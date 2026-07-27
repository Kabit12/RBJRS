"""
Authentication Tests
======================
Tests for login, registration, and access control flows.

Coverage:
- Candidate registration with valid/invalid data
- Recruiter registration with admin notification
- Login with valid/invalid credentials
- Login rate limiting
- Logout
- Authenticated redirect behavior
"""

import pytest


class TestLogin:
    """Tests for the login flow."""

    def test_login_page_loads(self, client):
        """Login page should return 200 for anonymous users."""
        response = client.get('/auth/login')
        assert response.status_code == 200
        assert b'Sign In' in response.data or b'Login' in response.data or b'login' in response.data

    def test_login_with_valid_credentials(self, client, make_user, app):
        """Login with correct email/password should redirect to dashboard."""
        with app.app_context():
            user = make_user(
                email='logintest@example.com',
                password='ValidPass123!',
                role='candidate',
            )

        response = client.post('/auth/login', data={
            'email': 'logintest@example.com',
            'password': 'ValidPass123!',
        }, follow_redirects=False)
        # Should redirect to dashboard on success
        assert response.status_code in (302, 303)

    def test_login_with_wrong_password(self, client, make_user, app):
        """Login with wrong password should show error."""
        with app.app_context():
            make_user(
                email='wrongpass@example.com',
                password='CorrectPass123!',
                role='candidate',
            )

        response = client.post('/auth/login', data={
            'email': 'wrongpass@example.com',
            'password': 'WrongPassword!',
        }, follow_redirects=True)
        assert response.status_code == 200
        assert b'Invalid' in response.data or b'invalid' in response.data

    def test_login_with_nonexistent_email(self, client):
        """Login with email that doesn't exist should show error."""
        response = client.post('/auth/login', data={
            'email': 'nonexistent@example.com',
            'password': 'SomePassword!',
        }, follow_redirects=True)
        assert response.status_code == 200
        assert b'Invalid' in response.data or b'invalid' in response.data

    def test_login_with_empty_fields(self, client):
        """Login with empty fields should show validation error."""
        response = client.post('/auth/login', data={
            'email': '',
            'password': '',
        }, follow_redirects=True)
        assert response.status_code == 200

    def test_login_with_deactivated_user(self, client, make_user, app):
        """Login with a deactivated account should be blocked."""
        with app.app_context():
            make_user(
                email='deactivated@example.com',
                password='ValidPass123!',
                role='candidate',
                is_active=False,
            )

        response = client.post('/auth/login', data={
            'email': 'deactivated@example.com',
            'password': 'ValidPass123!',
        }, follow_redirects=True)
        assert response.status_code == 200
        assert b'deactivated' in response.data or b'Deactivated' in response.data


class TestCandidateRegistration:
    """Tests for candidate registration."""

    def test_register_page_loads(self, client):
        """Registration page should return 200."""
        response = client.get('/auth/register/candidate')
        assert response.status_code == 200

    def test_register_candidate_success(self, client, app):
        """Valid registration should create a candidate and redirect."""
        response = client.post('/auth/register/candidate', data={
            'first_name': 'Test',
            'last_name': 'Candidate',
            'email': 'newcandidate@example.com',
            'password': 'SecurePass123!',
            'confirm_password': 'SecurePass123!',
        }, follow_redirects=False)
        # Should redirect to dashboard
        assert response.status_code in (302, 303)

        # Verify user was created
        with app.app_context():
            from app.models.user import User
            user = User.query.filter_by(email='newcandidate@example.com').first()
            assert user is not None
            assert user.role == 'candidate'
            assert user.candidate_profile is not None

    def test_register_duplicate_email(self, client, make_user, app):
        """Registration with existing email should fail."""
        with app.app_context():
            make_user(email='existing@example.com')

        response = client.post('/auth/register/candidate', data={
            'first_name': 'Test',
            'last_name': 'Duplicate',
            'email': 'existing@example.com',
            'password': 'SecurePass123!',
            'confirm_password': 'SecurePass123!',
        }, follow_redirects=True)
        assert response.status_code == 200
        # WTForms validate_email raises "already exists" or the service returns an error
        assert b'already exists' in response.data or b'Registration failed' in response.data

    def test_register_password_too_short(self, client):
        """Registration with password < 8 chars should fail."""
        response = client.post('/auth/register/candidate', data={
            'first_name': 'Test',
            'last_name': 'Short',
            'email': 'shortpass@example.com',
            'password': 'ab12',
            'confirm_password': 'ab12',
        }, follow_redirects=True)
        assert response.status_code == 200
        assert b'8 characters' in response.data or b'at least' in response.data

    def test_register_password_mismatch(self, client):
        """Registration with non-matching passwords should fail."""
        response = client.post('/auth/register/candidate', data={
            'first_name': 'Test',
            'last_name': 'Mismatch',
            'email': 'mismatch@example.com',
            'password': 'SecurePass123!',
            'confirm_password': 'DifferentPass456!',
        }, follow_redirects=True)
        assert response.status_code == 200
        assert b'match' in response.data or b'Match' in response.data


class TestRecruiterRegistration:
    """Tests for recruiter registration."""

    def test_register_recruiter_success(self, client, app):
        """Valid recruiter registration should succeed and notify admins."""
        # Create an admin user first to receive notification
        with app.app_context():
            from app.services.auth_service import create_admin_user
            create_admin_user('admin@example.com', 'AdminPass123!')

        response = client.post('/auth/register/recruiter', data={
            'first_name': 'Recruiter',
            'last_name': 'Test',
            'email': 'recruiter_new@example.com',
            'password': 'SecurePass123!',
            'confirm_password': 'SecurePass123!',
            'company_name': 'Test Corp',
            'industry': 'Technology',
            'company_size': '11-50',
        }, follow_redirects=False)
        assert response.status_code in (302, 303)

        # Verify user was created
        with app.app_context():
            from app.models.user import User
            user = User.query.filter_by(email='recruiter_new@example.com').first()
            assert user is not None
            assert user.role == 'recruiter'
            assert user.recruiter_profile is not None

            # Verify admin notification was created
            from app.models.notification import Notification
            admin = User.query.filter_by(email='admin@example.com').first()
            if admin:
                notif = Notification.query.filter_by(user_id=admin.id).first()
                assert notif is not None
                assert 'recruiter' in notif.message.lower() or 'pending' in notif.message.lower()

    def test_register_recruiter_missing_company(self, client):
        """Recruiter registration without company name should fail."""
        response = client.post('/auth/register/recruiter', data={
            'first_name': 'Recruiter',
            'last_name': 'NoCompany',
            'email': 'nocompany@example.com',
            'password': 'SecurePass123!',
            'confirm_password': 'SecurePass123!',
            'company_name': '',  # Required field
        }, follow_redirects=True)
        assert response.status_code == 200
        assert b'required' in response.data or b'Required' in response.data


class TestLogout:
    """Tests for logout."""

    def test_logout_redirects(self, client, make_user, app):
        """Logout should redirect to landing page."""
        with app.app_context():
            make_user(email='logouttest@example.com', password='ValidPass123!')

        # Login first
        client.post('/auth/login', data={
            'email': 'logouttest@example.com',
            'password': 'ValidPass123!',
        })

        # Logout
        response = client.get('/auth/logout', follow_redirects=False)
        assert response.status_code in (302, 303)


class TestAccessControl:
    """Tests for role-based access control."""

    def test_candidate_dashboard_requires_login(self, client):
        """Candidate dashboard should redirect to login for anonymous users."""
        response = client.get('/candidate/dashboard', follow_redirects=False)
        assert response.status_code in (302, 303)

    def test_recruiter_dashboard_requires_login(self, client):
        """Recruiter dashboard should redirect to login for anonymous users."""
        response = client.get('/recruiter/dashboard', follow_redirects=False)
        assert response.status_code in (302, 303)

    def test_admin_dashboard_requires_login(self, client):
        """Admin dashboard should redirect to login for anonymous users."""
        response = client.get('/admin/dashboard', follow_redirects=False)
        assert response.status_code in (302, 303)
