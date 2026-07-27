# D-Project — Resume-Based Job Recommendation Platform

## Project Overview
A Flask-based web application for job matching and recruitment with ML-powered resume parsing and job recommendations. Built for a 5th semester academic project (Project RBJRS).

## Tech Stack
- **Framework**: Flask with Blueprints (auth, candidate, recruiter, admin)
- **Database**: SQLAlchemy + Flask-Migrate (SQLite/PostgreSQL)
- **Auth**: Flask-Login + Flask-Bcrypt + Google OAuth (google-auth)
- **Forms**: Flask-WTF + WTForms
- **ML/MLOps**: scikit-learn, NLTK, joblib, matplotlib for visualizations
- **PDF Processing**: PyMuPDF (fitz), pdfplumber, FPDF2 for report generation
- **Frontend**: Server-rendered Jinja2 templates + vanilla JS/CSS

## Project Structure
```
D:\Project\
├── app/
│   ├── __init__.py              # Flask app factory (create_app)
│   ├── config.py                # Config classes (Config, Dev/Prod/Test)
│   ├── models/                  # SQLAlchemy models
│   │   ├── __init__.py
│   │   ├── user.py              # User, Candidate, Recruiter, Admin
│   │   ├── job.py               # Job, JobCategory, Application
│   │   ├── resume.py            # Resume, Skill, Education, Experience
│   │   ├── application.py       # Application, ApplicationStatus
│   │   ├── recruiter.py         # RecruiterProfile, Company
│   │   └── notification.py      # Notification model
│   ├── blueprints/
│   │   ├── auth/                # Auth routes (login, register, OAuth)
│   │   ├── candidate/           # Candidate dashboard, jobs, applications
│   │   ├── recruiter/           # Recruiter dashboard, jobs, applicants
│   │   ├── admin/               # Admin dashboard, user management
│   │   └── main/                # Landing page, public routes
│   ├── services/                # Business logic layer
│   │   ├── auth_service.py      # Auth logic (register, login, OAuth)
│   │   ├── job_service.py       # Job CRUD, search, filtering
│   │   ├── resume_service.py    # Resume parsing, skill extraction
│   │   └── notification_service.py
│   ├── ml/
│   │   ├── recommendation_engine.py    # Job recommendation engine
│   │   ├── text_preprocessor.py        # NLP text preprocessing (NLTK)
│   │   └── training/
│   │       ├── train_classifier.py     # Model training script
│   │       └── visualizations.py       # Training visualizations (matplotlib)
│   ├── templates/               # Jinja2 templates
│   │   ├── auth/
│   │   ├── candidate/
│   │   ├── recruiter/
│   │   ├── admin/
│   │   └── base.html
│   └── static/
│       ├── css/, js/, uploads/
├── ml_models/                   # Trained models & artifacts
│   ├── training_metrics.json
│   └── visualizations/
├── instance/                    # SQLite DB (dev)
├── convert_report_to_pdf.py     # PDF report generator
├── seed.py                      # Database seeding script
├── run.py                       # App entry point
├── requirements.txt
├── .env / .env.example
└── README.md
```

## Key Components

### Models (SQLAlchemy)
- **User** (abstract base) → **Candidate**, **Recruiter**, **Admin** (joined table inheritance)
- **Job** ↔ **JobCategory** (many-to-one), **Job** ↔ **Application** (one-to-many)
- **Resume** → **Skill**, **Education**, **Experience** (one-to-many each)
- **Application** with **ApplicationStatus** enum (pending, reviewed, accepted, rejected)
- **Notification** for user notifications

### Blueprints & Routes
| Blueprint | Prefix | Key Routes |
|-----------|--------|------------|
| `auth` | `/auth` | `/login`, `/register/candidate`, `/register/recruiter`, `/google-login` |
| `candidate` | `/candidate` | `/dashboard`, `/jobs`, `/jobs/<id>`, `/jobs/<id>/apply`, `/resume`, `/applications`, `/recommendations` |
| `recruiter` | `/recruiter` | `/dashboard`, `/jobs`, `/jobs/create`, `/jobs/<id>/edit`, `/applicants/<job_id>`, `/applicants/<id>/update-status`, `/pending-approval` |
| `admin` | `/admin` | `/dashboard`, `/users`, `/users/<id>/toggle`, `/users/<id>/delete`, `/jobs/<id>/toggle`, `/analytics` |

### Services Layer
- **AuthService**: User registration, login, Google OAuth, password hashing
- **JobService**: Job CRUD, search/filter, category management, recruiter approval workflow
- **ResumeService**: PDF parsing (pdfplumber), skill extraction (NLTK), section identification
- **RecommendationEngine**: TF-IDF + cosine similarity for job-candidate matching

### ML Pipeline
- **Training**: `ml/training/train_classifier.py` → trains classifier, saves to `ml_models/`
- **Visualizations**: `ml/training/visualizations.py` → generates training charts in `ml_models/visualizations/`
- **Inference**: `recommendation_engine.py` → loads model, computes TF-IDF similarity for recommendations
- **Text Preprocessing**: NLTK-based cleaning, tokenization, lemmatization, skill extraction

## Development Setup
```bash
# 1. Create venv & install
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env with SECRET_KEY, DATABASE_URL, GOOGLE_OAUTH creds, etc.

# 3. Initialize DB
flask db upgrade  # or: python run.py (auto-creates tables)

# 4. Seed database (optional)
python seed.py

# 5. Train ML models (optional, pre-trained in ml_models/)
python -m app.ml.training.train_classifier

# 6. Run dev server
python run.py
# Or: flask run
```

## Environment Variables (`.env`)
```env
SECRET_KEY=your-secret-key
DATABASE_URL=sqlite:///instance/app.db  # or postgresql://...
FLASK_ENV=development
GOOGLE_CLIENT_ID=xxx
GOOGLE_CLIENT_SECRET=xxx
MAIL_SERVER=smtp.gmail.com
MAIL_PORT=587
MAIL_USERNAME=xxx
MAIL_PASSWORD=xxx
```

## Key Workflows

### Candidate Flow
1. Register → Upload resume (PDF) → Auto-parsed for skills/education/experience
2. Browse jobs → Apply → Track applications
3. Get ML-powered job recommendations based on resume skills

### Recruiter Flow
1. Register → **Pending admin approval** → Approved recruiter can post jobs
2. Create/edit jobs → Manage applications → Update application status
3. View ranked candidates for each job (ML ranking)

### Admin Flow
1. Approve/disapprove recruiters
2. Manage all users (toggle active, delete)
3. Toggle job visibility
4. Platform analytics dashboard

## ML Model Details
- **Vectorizer**: TF-IDF (scikit-learn) on combined job description + skills
- **Similarity**: Cosine similarity between candidate profile vector and job vectors
- **Training Data**: Generated from job descriptions + synthetic candidate profiles
- **Artifacts**: `ml_models/` contains `vectorizer.joblib`, `classifier.joblib`, `training_metrics.json`

## Common Commands
```bash
# Run tests (if any)
pytest

# DB migrations
flask db migrate -m "message"
flask db upgrade

# Generate PDF report
python convert_report_to_pdf.py

# Train ML models from scratch
python -m app.ml.training.train_classifier

# Generate training visualizations
python -m app.ml.training.visualizations
```

## Architecture Notes
- **Layered**: Blueprints (HTTP) → Services (business logic) → Models (data) → ML (intelligence)
- **Blueprints** are thin HTTP handlers; business logic lives in `services/`
- **ML** is decoupled via `RecommendationEngine` class (loads models at startup)
- **File uploads**: Stored in `app/static/uploads/` (resumes, logos, profiles)
- **Admin approval required** for recruiters before they can post jobs

## Common Patterns in Codebase
- **Service pattern**: All DB operations in `services/*.py`, not in routes
- **Decorator auth**: `@login_required`, `@candidate_required`, `@recruiter_required`, `@admin_required`
- **Form validation**: WTForms classes in each blueprint's `forms.py` (or inline)
- **Template inheritance**: All extend `base.html` with blocks for content, scripts, styles
- **Flash messages**: Used for user feedback across redirects

## Known TODOs / Future Work
- [ ] Add automated tests (pytest)
- [ ] Dockerize for production deployment
- [ ] Add Redis caching for recommendations
- [ ] Implement real-time notifications (WebSockets)
- [ ] Add resume parsing for DOCX format
- [ ] Improve ML model with transformer embeddings