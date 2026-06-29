"""
Feature Extractor Module
==========================
Extracts structured information from resume text:
- Skills (programming languages, frameworks, tools, soft skills)
- Education (degrees, institutions, fields of study)
- Experience (companies, titles, durations)
- Certifications (names, issuers)
- Projects (names, technologies)

Design Decisions:
- Uses regex-based pattern matching combined with a curated skill dictionary
  for reliable extraction without requiring a trained NER model.
- Skill categories are defined as dictionaries for weighted matching.
- Education and experience extraction use section-based parsing:
  the text is split into sections (Education, Experience, etc.) and
  each section is parsed with domain-specific patterns.
- This approach is more reliable than pure NER for structured resumes
  because resumes follow predictable formatting conventions.

This is the third step in the processing pipeline:
    Upload → Parse → Preprocess → [Extract Features] → Classify → Recommend
"""

import re
from collections import defaultdict


# Comprehensive skill dictionary organized by category
SKILL_DATABASE = {
    'programming': [
        'python', 'java', 'javascript', 'typescript', 'c', 'c++', 'c#',
        'ruby', 'php', 'swift', 'kotlin', 'go', 'rust', 'scala', 'r',
        'matlab', 'perl', 'dart', 'lua', 'haskell', 'objective-c',
        'visual basic', 'assembly', 'groovy', 'julia', 'fortran',
        'cobol', 'bash', 'shell', 'powershell', 'sql', 'plsql',
        'html', 'css', 'sass', 'less', 'xml', 'json', 'yaml',
    ],
    'framework': [
        'react', 'angular', 'vue', 'svelte', 'next.js', 'nuxt',
        'django', 'flask', 'fastapi', 'spring', 'spring boot',
        'express', 'node.js', 'rails', 'ruby on rails', 'laravel',
        'asp.net', '.net', '.net core', 'symfony', 'codeigniter',
        'bootstrap', 'tailwind', 'material ui', 'jquery',
        'react native', 'flutter', 'ionic', 'xamarin', 'electron',
        'tensorflow', 'pytorch', 'keras', 'scikit-learn', 'sklearn',
        'opencv', 'pandas', 'numpy', 'scipy', 'matplotlib',
        'streamlit', 'gradio', 'huggingface',
    ],
    'database': [
        'mysql', 'postgresql', 'mongodb', 'sqlite', 'oracle',
        'sql server', 'redis', 'cassandra', 'dynamodb', 'firebase',
        'elasticsearch', 'neo4j', 'mariadb', 'couchdb', 'influxdb',
        'supabase', 'prisma', 'sequelize', 'sqlalchemy',
    ],
    'cloud': [
        'aws', 'azure', 'gcp', 'google cloud', 'heroku', 'digitalocean',
        'vercel', 'netlify', 'cloudflare', 'ec2', 's3', 'lambda',
        'ecs', 'eks', 'fargate', 'rds', 'cloudfront', 'sqs', 'sns',
        'kubernetes', 'docker', 'terraform', 'ansible', 'jenkins',
        'github actions', 'gitlab ci', 'circleci', 'travis ci',
    ],
    'data_science': [
        'machine learning', 'deep learning', 'natural language processing',
        'nlp', 'computer vision', 'data mining', 'data analysis',
        'statistics', 'regression', 'classification', 'clustering',
        'neural network', 'cnn', 'rnn', 'lstm', 'transformer',
        'bert', 'gpt', 'reinforcement learning', 'a/b testing',
        'feature engineering', 'model training', 'etl', 'data pipeline',
        'big data', 'hadoop', 'spark', 'kafka', 'airflow',
        'tableau', 'power bi', 'looker', 'data visualization',
    ],
    'devops': [
        'ci/cd', 'devops', 'sre', 'monitoring', 'logging',
        'prometheus', 'grafana', 'datadog', 'new relic', 'splunk',
        'nginx', 'apache', 'load balancing', 'microservices',
        'serverless', 'infrastructure as code', 'iac',
        'linux', 'unix', 'windows server', 'networking',
    ],
    'tools': [
        'git', 'github', 'gitlab', 'bitbucket', 'jira', 'confluence',
        'slack', 'trello', 'asana', 'notion', 'figma', 'sketch',
        'adobe photoshop', 'adobe illustrator', 'postman', 'swagger',
        'vs code', 'intellij', 'eclipse', 'vim', 'emacs',
        'agile', 'scrum', 'kanban', 'waterfall',
    ],
    'soft_skill': [
        'leadership', 'communication', 'teamwork', 'problem solving',
        'critical thinking', 'time management', 'project management',
        'presentation', 'negotiation', 'mentoring', 'collaboration',
        'adaptability', 'creativity', 'analytical', 'detail oriented',
        'self motivated', 'organizational', 'decision making',
        'conflict resolution', 'customer service', 'public speaking',
    ],
    'security': [
        'cybersecurity', 'network security', 'penetration testing',
        'ethical hacking', 'siem', 'firewall', 'ids', 'ips',
        'encryption', 'ssl', 'tls', 'oauth', 'jwt', 'sso',
        'vulnerability assessment', 'compliance', 'gdpr', 'hipaa',
        'soc2', 'iso 27001',
    ],
}

# Education keywords
DEGREE_PATTERNS = [
    r'(?:bachelor|b\.?s\.?|b\.?a\.?|b\.?e\.?|b\.?tech|b\.?sc)',
    r'(?:master|m\.?s\.?|m\.?a\.?|m\.?e\.?|m\.?tech|m\.?sc|mba)',
    r'(?:doctor|ph\.?d|d\.?phil)',
    r'(?:diploma|associate|certificate)',
]

EDUCATION_KEYWORDS = [
    'university', 'college', 'institute', 'school', 'academy',
    'bachelor', 'master', 'phd', 'doctorate', 'diploma',
    'degree', 'engineering', 'science', 'arts', 'technology',
    'computer science', 'information technology', 'mathematics',
    'physics', 'chemistry', 'biology', 'business', 'management',
    'economics', 'finance', 'accounting', 'marketing',
    'mechanical', 'electrical', 'civil', 'electronics',
]

# Section headers to identify resume sections
SECTION_HEADERS = {
    'education': ['education', 'academic', 'qualification', 'degree'],
    'experience': ['experience', 'employment', 'work history', 'professional',
                   'career', 'work experience', 'professional experience'],
    'skills': ['skills', 'technical skills', 'competencies', 'expertise',
               'proficiencies', 'technologies', 'tools'],
    'certifications': ['certification', 'certificate', 'licenses', 'accreditation'],
    'projects': ['project', 'personal project', 'academic project', 'portfolio'],
}


class FeatureExtractor:
    """
    Extracts structured features from resume text.

    Usage:
        extractor = FeatureExtractor()
        features = extractor.extract_all(resume_text)
    """

    def extract_all(self, text):
        """
        Extract all features from resume text.

        Args:
            text: Raw or cleaned resume text.

        Returns:
            Dictionary with keys: skills, education, experience,
            certifications, projects
        """
        if not text:
            return {
                'skills': [],
                'education': [],
                'experience': [],
                'certifications': [],
                'projects': [],
            }

        text_lower = text.lower()
        sections = self._identify_sections(text)

        return {
            'skills': self.extract_skills(text_lower),
            'education': self.extract_education(text, sections.get('education', '')),
            'experience': self.extract_experience(text, sections.get('experience', '')),
            'certifications': self.extract_certifications(text, sections.get('certifications', '')),
            'projects': self.extract_projects(text, sections.get('projects', '')),
        }

    def extract_skills(self, text):
        """
        Extract skills by matching against the skill database.

        Uses word-boundary matching to avoid false positives
        (e.g., "class" inside "classification").

        Args:
            text: Lowercase resume text.

        Returns:
            List of dicts with 'name' and 'category' keys.
        """
        text_lower = text.lower()
        found_skills = []
        seen = set()

        for category, skills in SKILL_DATABASE.items():
            for skill in skills:
                if skill in seen:
                    continue

                # Use word boundary matching for single words,
                # substring matching for multi-word skills
                if ' ' in skill or '.' in skill or '+' in skill or '#' in skill:
                    # Multi-word or special character skills: use simple containment
                    if skill in text_lower:
                        found_skills.append({
                            'name': skill,
                            'category': category,
                        })
                        seen.add(skill)
                else:
                    # Single word skills: use word boundary to avoid partial matches
                    pattern = r'\b' + re.escape(skill) + r'\b'
                    if re.search(pattern, text_lower):
                        found_skills.append({
                            'name': skill,
                            'category': category,
                        })
                        seen.add(skill)

        return found_skills

    def extract_education(self, text, education_section=''):
        """
        Extract education entries from resume text.

        Looks for degree patterns, institution names, and fields of study.

        Args:
            text: Full resume text.
            education_section: Text from the education section specifically.

        Returns:
            List of education entry dicts.
        """
        search_text = education_section if education_section else text
        entries = []

        # Look for degree patterns
        for pattern in DEGREE_PATTERNS:
            matches = re.finditer(
                pattern + r'[^\n]*(?:in|of)?\s*([^\n,]*)',
                search_text,
                re.IGNORECASE
            )
            for match in matches:
                full_match = match.group(0).strip()
                field = match.group(1).strip() if match.group(1) else ''

                # Try to identify the degree level
                degree = self._classify_degree(full_match)

                entries.append({
                    'degree': degree,
                    'field_of_study': field[:255] if field else None,
                    'institution': None,  # Hard to extract reliably
                    'start_date': None,
                    'end_date': None,
                    'gpa': self._extract_gpa(search_text),
                })

        # Deduplicate by degree
        seen_degrees = set()
        unique_entries = []
        for entry in entries:
            key = (entry['degree'], entry.get('field_of_study', ''))
            if key not in seen_degrees:
                seen_degrees.add(key)
                unique_entries.append(entry)

        return unique_entries[:5]  # Limit to 5 entries

    def extract_experience(self, text, experience_section=''):
        """
        Extract work experience entries.

        Args:
            text: Full resume text.
            experience_section: Text from the experience section.

        Returns:
            List of experience entry dicts.
        """
        search_text = experience_section if experience_section else text
        entries = []

        # Look for date range patterns (common in experience sections)
        date_pattern = r'(\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[\s,]*\d{4})\s*[-–—to]+\s*((?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[\s,]*\d{4}|present|current)'

        matches = re.finditer(date_pattern, search_text, re.IGNORECASE)
        for match in matches:
            start_date = match.group(1).strip()
            end_date = match.group(2).strip()

            # Get surrounding context for company/title
            start_pos = max(0, match.start() - 200)
            context = search_text[start_pos:match.start()].strip()
            lines = [l.strip() for l in context.split('\n') if l.strip()]

            title = lines[-1] if lines else None
            company = lines[-2] if len(lines) > 1 else None

            entries.append({
                'title': title[:255] if title else None,
                'company': company[:255] if company else None,
                'start_date': start_date,
                'end_date': end_date,
                'description': None,
            })

        return entries[:10]  # Limit to 10 entries

    def extract_certifications(self, text, cert_section=''):
        """
        Extract certifications from resume text.

        Args:
            text: Full resume text.
            cert_section: Text from the certifications section.

        Returns:
            List of certification dicts.
        """
        search_text = cert_section if cert_section else text
        certifications = []

        # Common certification patterns
        cert_patterns = [
            r'(?:certified|certification)\s+(?:in\s+)?([^\n,]+)',
            r'(aws\s+certified\s+[^\n,]+)',
            r'(google\s+cloud\s+(?:certified|professional)\s+[^\n,]+)',
            r'(microsoft\s+certified\s+[^\n,]+)',
            r'(cisco\s+certified\s+[^\n,]+)',
            r'(pmp|prince2|scrum\s+master|itil|cissp|ceh|comptia\s+[a-z]+)',
        ]

        for pattern in cert_patterns:
            matches = re.finditer(pattern, search_text, re.IGNORECASE)
            for match in matches:
                name = match.group(1).strip() if match.group(1) else match.group(0).strip()
                if len(name) > 3:  # Filter noise
                    certifications.append({
                        'name': name[:255],
                        'issuing_org': None,
                        'issue_date': None,
                    })

        return certifications[:10]

    def extract_projects(self, text, project_section=''):
        """
        Extract project entries from resume text.

        Args:
            text: Full resume text.
            project_section: Text from the projects section.

        Returns:
            List of project dicts.
        """
        search_text = project_section if project_section else text
        projects = []

        # Split project section by common separators
        if project_section:
            # Try to split by bullet points or numbered items
            items = re.split(r'\n(?=[•●▪\-\d])', project_section)
            for item in items:
                item = item.strip()
                if len(item) > 10:
                    # First line is usually the project name
                    lines = item.split('\n')
                    name = re.sub(r'^[•●▪\-\d\.\)\s]+', '', lines[0]).strip()
                    description = ' '.join(lines[1:]).strip() if len(lines) > 1 else None

                    if name:
                        projects.append({
                            'name': name[:255],
                            'description': description[:500] if description else None,
                            'technologies': None,
                        })

        return projects[:10]

    def _identify_sections(self, text):
        """
        Split resume text into identified sections.

        Uses section headers to identify boundaries between sections.

        Args:
            text: Full resume text.

        Returns:
            Dictionary mapping section names to their text content.
        """
        sections = {}
        lines = text.split('\n')
        current_section = None
        current_content = []

        for line in lines:
            line_lower = line.strip().lower()

            # Check if this line is a section header
            detected_section = None
            for section_name, keywords in SECTION_HEADERS.items():
                for keyword in keywords:
                    if (line_lower == keyword or
                        line_lower.startswith(keyword + ':') or
                        line_lower.startswith(keyword + ' ') and len(line_lower) < len(keyword) + 5):
                        detected_section = section_name
                        break
                if detected_section:
                    break

            if detected_section:
                # Save previous section
                if current_section:
                    sections[current_section] = '\n'.join(current_content)
                current_section = detected_section
                current_content = []
            else:
                current_content.append(line)

        # Save last section
        if current_section:
            sections[current_section] = '\n'.join(current_content)

        return sections

    @staticmethod
    def _classify_degree(text):
        """Classify degree level from text."""
        text_lower = text.lower()
        if any(kw in text_lower for kw in ['ph.d', 'phd', 'doctor', 'doctorate']):
            return 'Doctorate'
        elif any(kw in text_lower for kw in ['master', 'm.s', 'ms', 'm.a', 'ma', 'm.e', 'mba', 'm.tech', 'mtech']):
            return "Master's"
        elif any(kw in text_lower for kw in ['bachelor', 'b.s', 'bs', 'b.a', 'ba', 'b.e', 'b.tech', 'btech']):
            return "Bachelor's"
        elif any(kw in text_lower for kw in ['diploma', 'associate']):
            return 'Diploma'
        elif 'certificate' in text_lower:
            return 'Certificate'
        return 'Other'

    @staticmethod
    def _extract_gpa(text):
        """Extract GPA if present in text."""
        gpa_match = re.search(r'(?:gpa|cgpa|grade)[:\s]*(\d+\.?\d*)\s*(?:/\s*(\d+\.?\d*))?',
                              text, re.IGNORECASE)
        if gpa_match:
            gpa = float(gpa_match.group(1))
            scale = float(gpa_match.group(2)) if gpa_match.group(2) else 4.0
            if gpa <= scale:
                return gpa
        return None


# Singleton instance
feature_extractor = FeatureExtractor()
