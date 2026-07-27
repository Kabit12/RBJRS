"""
Database Seed Script
=====================
Creates initial data for the application:
1. Admin user
2. Sample skills
3. (Optional) Sample recruiter + jobs for testing

Usage:
    python seed.py
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import create_app
from app.services.auth_service import create_admin_user, register_recruiter
from app.extensions import db
from app.models.skill import Skill
from app.services.job_service import create_job


def seed_database():
    """Seed the database with initial data."""
    app = create_app()

    with app.app_context():
        print('=' * 50)
        print('Database Seeding')
        print('=' * 50)

        # 1. Create admin user
        print('\n[1] Creating admin user...')
        success, result = create_admin_user(
            email='admin@rbjrs.com',
            password='admin123',
            first_name='Admin',
            last_name='User'
        )
        if success:
            print(f'  [OK] Admin created: admin@rbjrs.com / admin123')
        else:
            print(f'  -> {result}')

        # 2. Seed common skills
        print('\n[2] Seeding skills database...')
        skills = [
            ('python', 'programming'), ('java', 'programming'), ('javascript', 'programming'),
            ('c++', 'programming'), ('c#', 'programming'), ('sql', 'programming'),
            ('html', 'programming'), ('css', 'programming'), ('typescript', 'programming'),
            ('react', 'framework'), ('angular', 'framework'), ('vue', 'framework'),
            ('django', 'framework'), ('flask', 'framework'), ('spring boot', 'framework'),
            ('node.js', 'framework'), ('express', 'framework'), ('tensorflow', 'framework'),
            ('pytorch', 'framework'), ('scikit-learn', 'framework'),
            ('mysql', 'database'), ('postgresql', 'database'), ('mongodb', 'database'),
            ('redis', 'database'), ('sqlite', 'database'),
            ('aws', 'cloud'), ('azure', 'cloud'), ('gcp', 'cloud'),
            ('docker', 'devops'), ('kubernetes', 'devops'), ('git', 'tools'),
            ('machine learning', 'data_science'), ('deep learning', 'data_science'),
            ('data analysis', 'data_science'), ('nlp', 'data_science'),
            ('leadership', 'soft_skill'), ('communication', 'soft_skill'),
            ('teamwork', 'soft_skill'), ('problem solving', 'soft_skill'),
        ]

        count = 0
        for name, category in skills:
            existing = Skill.query.filter_by(name=name).first()
            if not existing:
                skill = Skill(name=name, category=category)
                db.session.add(skill)
                count += 1
        db.session.commit()
        print(f'  [OK] {count} new skills added')

        # 3. Create sample recruiter with jobs
        print('\n[3] Creating sample recruiter and jobs...')
        success, result = register_recruiter({
            'email': 'recruiter@techcorp.com',
            'password': 'recruiter123',
            'first_name': 'Sarah',
            'last_name': 'Johnson',
            'company_name': 'TechCorp Solutions',
            'industry': 'Technology',
            'company_size': '51-200',
            'location': 'San Francisco, CA',
        })

        if success:
            print(f'  [OK] Recruiter created: recruiter@techcorp.com / recruiter123')
            recruiter = result.recruiter_profile
            # Auto-approve the sample recruiter
            recruiter.is_approved = True
            db.session.commit()
            print(f'  [OK] Recruiter auto-approved')

            # Create sample jobs
            sample_jobs = [
                {
                    'title': 'Senior Python Developer',
                    'description': 'We are looking for an experienced Python developer to join our backend team. You will work on building scalable APIs, microservices, and data pipelines. The ideal candidate has strong experience with Django/Flask, PostgreSQL, Docker, and cloud platforms.',
                    'location': 'Kathmandu, Nepal (Hybrid)',
                    'job_type': 'full-time',
                    'experience_level': 'Senior',
                    'salary_min': '80000',
                    'salary_max': '150000',
                    'requirements': 'Bachelor\'s degree in Computer Science or related field.\n5+ years of Python development experience.\nStrong knowledge of Django or Flask frameworks.\nExperience with PostgreSQL, Redis, Docker.\nFamiliarity with AWS or GCP cloud services.',
                    'responsibilities': 'Design and implement RESTful APIs.\nBuild and maintain microservices architecture.\nOptimize database queries and application performance.\nParticipate in code reviews and mentor junior developers.\nCollaborate with product and design teams.',
                    'skills': 'python, django, flask, postgresql, docker, aws, redis, git',
                },
                {
                    'title': 'Data Scientist',
                    'description': 'Join our data science team to build machine learning models that drive product decisions. You will work on recommendation systems, NLP, and predictive analytics.',
                    'location': 'Remote',
                    'job_type': 'full-time',
                    'experience_level': 'Mid Level',
                    'salary_min': '60000',
                    'salary_max': '120000',
                    'requirements': 'Master\'s degree in Data Science, Statistics, or related field.\n3+ years of experience in ML/data science.\nStrong Python skills (NumPy, Pandas, scikit-learn, TensorFlow).\nExperience with NLP and recommendation systems.',
                    'responsibilities': 'Build and deploy machine learning models.\nConduct statistical analysis and A/B testing.\nCreate data visualizations and dashboards.\nCollaborate with engineering teams on model deployment.',
                    'skills': 'python, machine learning, deep learning, nlp, tensorflow, scikit-learn, sql, data analysis',
                },
                {
                    'title': 'Frontend Developer (React)',
                    'description': 'We need a talented frontend developer to create beautiful, responsive web applications using React. You will work closely with designers and backend engineers.',
                    'location': 'Lalitpur, Nepal',
                    'job_type': 'full-time',
                    'experience_level': 'Mid Level',
                    'salary_min': '50000',
                    'salary_max': '100000',
                    'requirements': '3+ years of React development experience.\nStrong HTML, CSS, JavaScript/TypeScript skills.\nExperience with state management (Redux, Zustand).\nFamiliarity with RESTful APIs and GraphQL.',
                    'responsibilities': 'Build responsive UI components using React.\nImplement pixel-perfect designs from Figma mockups.\nOptimize frontend performance.\nWrite unit and integration tests.',
                    'skills': 'javascript, typescript, react, html, css, git',
                },
                {
                    'title': 'DevOps Engineer',
                    'description': 'We are seeking a DevOps engineer to manage our cloud infrastructure and CI/CD pipelines. You will ensure high availability and reliability of our production systems.',
                    'location': 'Kathmandu, Nepal (Remote OK)',
                    'job_type': 'full-time',
                    'experience_level': 'Senior',
                    'salary_min': '90000',
                    'salary_max': '160000',
                    'requirements': '5+ years of DevOps/SRE experience.\nStrong knowledge of AWS services.\nExperience with Docker, Kubernetes, Terraform.\nCI/CD pipeline expertise (Jenkins, GitHub Actions).',
                    'responsibilities': 'Design and manage cloud infrastructure on AWS.\nBuild and maintain CI/CD pipelines.\nMonitor system performance and reliability.\nImplement security best practices.',
                    'skills': 'aws, docker, kubernetes, terraform, git, python, linux',
                },
                {
                    'title': 'Java Backend Developer',
                    'description': 'Looking for a Java developer to build enterprise-grade backend services using Spring Boot. Strong understanding of microservices architecture required.',
                    'location': 'Pokhara, Nepal',
                    'job_type': 'full-time',
                    'experience_level': 'Mid Level',
                    'salary_min': '45000',
                    'salary_max': '85000',
                    'requirements': '3-5 years of Java development.\nSpring Boot and Hibernate experience.\nSQL database knowledge (MySQL, PostgreSQL).\nExperience with microservices and REST APIs.',
                    'responsibilities': 'Develop microservices using Spring Boot.\nDesign database schemas and write optimized queries.\nParticipate in architecture decisions.\nWrite comprehensive unit and integration tests.',
                    'skills': 'java, spring boot, mysql, postgresql, docker, git',
                },
            ]

            for job_data in sample_jobs:
                success_j, result_j = create_job(recruiter, job_data)
                if success_j:
                    print(f'  [OK] Job created: {job_data["title"]}')
                else:
                    print(f'  [FAIL] Failed: {result_j}')
        else:
            print(f'  -> {result}')

        print(f'\n{"=" * 50}')
        print('Seeding complete!')
        print(f'{"=" * 50}')
        print('\nTest accounts:')
        print('  Admin:     admin@rbjrs.com / admin123')
        print('  Recruiter: recruiter@techcorp.com / recruiter123')
        print('  (Register a candidate account via the UI)')


if __name__ == '__main__':
    seed_database()
