"""
Authentication Forms
=====================
WTForms definitions for user registration and login.

Design Decision:
- WTForms provides built-in CSRF protection, server-side validation,
  and clean error handling through Flask-WTF integration.
- Separate registration forms for candidates and recruiters because
  recruiters need company information at signup.
- Email validation uses the email-validator package for RFC compliance.
"""

from flask_wtf import FlaskForm
from wtforms import (
    StringField, PasswordField, BooleanField, SelectField,
    TextAreaField, SubmitField
)
from wtforms.validators import (
    DataRequired, Email, EqualTo, Length, ValidationError, Optional
)
from app.models.user import User


class LoginForm(FlaskForm):
    """Login form with email and password."""

    email = StringField('Email Address', validators=[
        DataRequired(message='Email is required.'),
        Email(message='Please enter a valid email address.')
    ])
    password = PasswordField('Password', validators=[
        DataRequired(message='Password is required.')
    ])
    remember_me = BooleanField('Remember Me')
    submit = SubmitField('Sign In')


class CandidateRegistrationForm(FlaskForm):
    """Registration form for job seekers (candidates)."""

    first_name = StringField('First Name', validators=[
        DataRequired(message='First name is required.'),
        Length(min=2, max=100, message='First name must be 2-100 characters.')
    ])
    last_name = StringField('Last Name', validators=[
        DataRequired(message='Last name is required.'),
        Length(min=2, max=100, message='Last name must be 2-100 characters.')
    ])
    email = StringField('Email Address', validators=[
        DataRequired(message='Email is required.'),
        Email(message='Please enter a valid email address.'),
        Length(max=255)
    ])
    password = PasswordField('Password', validators=[
        DataRequired(message='Password is required.'),
        Length(min=8, message='Password must be at least 8 characters.')
    ])
    confirm_password = PasswordField('Confirm Password', validators=[
        DataRequired(message='Please confirm your password.'),
        EqualTo('password', message='Passwords do not match.')
    ])
    phone = StringField('Phone Number', validators=[Optional(), Length(max=20)])
    location = StringField('Location', validators=[Optional(), Length(max=255)])
    submit = SubmitField('Create Account')

    def validate_email(self, field):
        """Check that email is not already registered."""
        user = User.query.filter_by(email=field.data.lower().strip()).first()
        if user:
            raise ValidationError('An account with this email already exists.')


class RecruiterRegistrationForm(FlaskForm):
    """Registration form for recruiters/employers."""

    first_name = StringField('First Name', validators=[
        DataRequired(message='First name is required.'),
        Length(min=2, max=100)
    ])
    last_name = StringField('Last Name', validators=[
        DataRequired(message='Last name is required.'),
        Length(min=2, max=100)
    ])
    email = StringField('Email Address', validators=[
        DataRequired(message='Email is required.'),
        Email(message='Please enter a valid email address.'),
        Length(max=255)
    ])
    password = PasswordField('Password', validators=[
        DataRequired(message='Password is required.'),
        Length(min=8, message='Password must be at least 8 characters.')
    ])
    confirm_password = PasswordField('Confirm Password', validators=[
        DataRequired(message='Please confirm your password.'),
        EqualTo('password', message='Passwords do not match.')
    ])
    company_name = StringField('Company Name', validators=[
        DataRequired(message='Company name is required.'),
        Length(min=2, max=255)
    ])
    industry = StringField('Industry', validators=[Optional(), Length(max=100)])
    company_size = SelectField('Company Size', choices=[
        ('', 'Select company size'),
        ('1-10', '1-10 employees'),
        ('11-50', '11-50 employees'),
        ('51-200', '51-200 employees'),
        ('201-500', '201-500 employees'),
        ('500+', '500+ employees')
    ], validators=[Optional()])
    phone = StringField('Phone Number', validators=[Optional(), Length(max=20)])
    location = StringField('Location', validators=[Optional(), Length(max=255)])
    submit = SubmitField('Create Account')

    def validate_email(self, field):
        """Check that email is not already registered."""
        user = User.query.filter_by(email=field.data.lower().strip()).first()
        if user:
            raise ValidationError('An account with this email already exists.')
