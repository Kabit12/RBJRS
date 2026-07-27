"""
Candidate File Routes
=======================
Authenticated file serving for candidate-owned files.

After uploads were moved out of app/static/ for security, files need to be
served via authenticated routes that verify ownership.

Security Model:
- Candidates can only download their own resumes
- Files are served via flask.send_file (not static serving)
- @login_required + ownership check on every request
"""

import os
from flask import abort, current_app, send_file
from flask_login import login_required, current_user
from app.blueprints.candidate import candidate_bp
from app.utils.decorators import candidate_required
from app.extensions import db
from app.models.resume import Resume


@candidate_bp.route('/resume/<int:resume_id>/download')
@login_required
@candidate_required
def download_resume(resume_id):
    """
    Download a candidate's own resume file.

    Security:
    - Only the owning candidate can download their resume.
    - File is served via send_file(), not static URL.
    """
    candidate = current_user.candidate_profile
    resume = db.session.get(Resume, resume_id)

    if not resume or resume.candidate_id != candidate.id:
        abort(404)

    # Security Fix: Path traversal prevention using strict commonpath directory boundary check
    upload_folder = os.path.abspath(current_app.config['UPLOAD_FOLDER'])
    file_path = resume.file_path
    if not os.path.isabs(file_path):
        file_path = os.path.abspath(os.path.join(current_app.root_path, '..', file_path))
    else:
        file_path = os.path.abspath(file_path)

    try:
        if os.path.commonpath([upload_folder, file_path]) != upload_folder or not os.path.exists(file_path):
            abort(404)
    except Exception:
        abort(404)

    return send_file(
        file_path,
        as_attachment=True,
        download_name=resume.file_name,
    )
