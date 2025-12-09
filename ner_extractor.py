"""
Enhanced Named Entity Recognition for Resumes
==============================================
Custom NER for extracting resume-specific entities:
- SKILL: Technical and soft skills
- EDUCATION: Degrees, institutions, fields of study
- EXPERIENCE: Job titles, companies, durations
- CERTIFICATION: Professional certifications
- PROJECT: Project names and descriptions

Uses spaCy NER + rule-based patterns + custom entity recognition.

Author: Noel John
Course: IE 7500 Applied NLP, Northeastern University
"""

import re
import json
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
from pathlib import Path

import spacy
from spacy.tokens import Doc, Span
from spacy.language import Language
from spacy.matcher import Matcher, PhraseMatcher

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import os
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


# =============================================================================
# Custom Entity Labels for Resumes
# =============================================================================

RESUME_ENTITY_LABELS = {
    'SKILL': 'Technical or soft skill',
    'DEGREE': 'Academic degree (BS, MS, PhD, etc.)',
    'INSTITUTION': 'Educational institution',
    'FIELD_OF_STUDY': 'Major or field of study',
    'JOB_TITLE': 'Job title or role',
    'COMPANY': 'Company or organization',
    'DURATION': 'Employment duration',
    'CERTIFICATION': 'Professional certification',
    'PROJECT': 'Project name',
    'LOCATION': 'Geographic location',
    'DATE': 'Date or time period'
}


# =============================================================================
# Pattern Definitions
# =============================================================================

# Degree patterns
DEGREE_PATTERNS = [
    # Full degree names with field
    r'\b(Bachelor\'?s?|Master\'?s?|Doctorate|Doctor)\s+(?:of\s+)?(Science|Arts|Engineering|Business|Administration|Philosophy|Education|Fine Arts|Laws|Medicine)(?:\s+in\s+[\w\s]+)?\b',
    # PhD variations
    r'\b(Ph\.?D\.?|Doctor\s+of\s+Philosophy)\b',
    # Professional doctorates
    r'\b(M\.?D\.?|J\.?D\.?|Ed\.?D\.?|D\.?B\.?A\.?|Pharm\.?D\.?)\b',
    # MBA and other professional masters
    r'\b(MBA|MFA|MPA|MPH|MSW|LLM|LLB|MEng|MCS|MIS)\b',
    # B.S./M.S. with optional periods - be more specific
    r'\b(B\.S\.?|B\.?S\.)\s*(?:in\s+[\w\s]+)?',
    r'\b(M\.S\.?|M\.?S\.)\s*(?:in\s+[\w\s]+)?',
    r'\b(B\.A\.?|B\.?A\.)\s*(?:in\s+[\w\s]+)?',
    r'\b(M\.A\.?|M\.?A\.)\s*(?:in\s+[\w\s]+)?',
    r'\b(B\.E\.?|B\.?E\.)\s*(?:in\s+[\w\s]+)?',
    r'\b(M\.E\.?|M\.?E\.)\s*(?:in\s+[\w\s]+)?',
    # Bachelor/Master of Science/Arts spelled out
    r'\bBachelor\s+of\s+Science\b',
    r'\bBachelor\s+of\s+Arts\b',
    r'\bMaster\s+of\s+Science\b',
    r'\bMaster\s+of\s+Arts\b',
    r'\bMaster\s+of\s+Business\s+Administration\b',
    # B.Tech/M.Tech
    r'\b(B\.?Tech\.?|M\.?Tech\.?)\b',
    # Associate degrees
    r'\b(A\.?A\.?S?|A\.?S\.?|Associate\'?s?\s+(?:of\s+)?(?:Science|Arts|Applied\s+Science)?)\b',
    # High school
    r'\b(High\s+School\s+Diploma|GED|Secondary\s+Education)\b',
]

# Certification patterns
CERTIFICATION_PATTERNS = [
    # AWS Certifications
    r'\bAWS\s+Certified[\w\s-]*(?:Solutions\s+Architect|Developer|SysOps|DevOps|Machine\s+Learning|Data\s+Analytics|Cloud\s+Practitioner)?[\w\s-]*',
    r'\bAmazon\s+Web\s+Services\s+Certified[\w\s-]*',
    # Azure Certifications
    r'\bAzure\s+(?:Certified|Administrator|Developer|Solutions\s+Architect|Data\s+Engineer|AI\s+Engineer)[\w\s-]*',
    r'\bMicrosoft\s+Certified[\w\s-]*',
    # Google Cloud Certifications
    r'\bGoogle\s+Cloud\s+(?:Certified|Professional)[\w\s-]*',
    r'\bGCP\s+(?:Certified|Professional)[\w\s-]*',
    # Project Management
    r'\b(PMP|Project\s+Management\s+Professional)\b',
    r'\b(CAPM|Certified\s+Associate\s+in\s+Project\s+Management)\b',
    r'\b(CSM|Certified\s+Scrum\s+Master)\b',
    r'\b(PSM\s*[I1]?|Professional\s+Scrum\s+Master)\b',
    r'\bSAFe[\s\d]*(?:Agilist|Practitioner|Architect)?\b',
    # IT Certifications
    r'\b(ITIL[\s\w]*(?:Foundation|Practitioner|Expert)?)\b',
    r'\b(CISSP|Certified\s+Information\s+Systems\s+Security\s+Professional)\b',
    r'\b(CISM|Certified\s+Information\s+Security\s+Manager)\b',
    r'\b(CEH|Certified\s+Ethical\s+Hacker)\b',
    r'\bCompTIA[\s\w+-]*(?:A\+|Network\+|Security\+|Cloud\+|Data\+|Linux\+)?\b',
    # Data Certifications
    r'\bTableau[\s\w]*(?:Desktop|Server)?\s*Certified[\w\s]*',
    r'\bCertified\s+(?:Data\s+)?(?:Analyst|Scientist|Engineer)[\w\s]*',
    # Finance Certifications
    r'\b(CPA|Certified\s+Public\s+Accountant)\b',
    r'\b(CFA|Chartered\s+Financial\s+Analyst)\b',
    r'\b(CFP|Certified\s+Financial\s+Planner)\b',
    r'\bSeries\s+\d+\b',
    # Six Sigma
    r'\b(?:Lean\s+)?Six\s+Sigma[\s\w]*(?:Green|Black|Yellow|White)\s*Belt\b',
    r'\b(?:Green|Black|Yellow|White)\s*Belt(?:\s+Certified)?\b',
    # Generic certification pattern
    r'\bCertified\s+[\w\s]+(?:Professional|Specialist|Expert|Administrator|Developer|Engineer|Analyst)\b',
]

# Job title patterns
JOB_TITLE_PATTERNS = [
    # Data roles
    r'\b(Senior|Junior|Lead|Principal|Staff|Associate)?\s*(Data)\s+(Scientist|Analyst|Engineer|Architect)\b',
    r'\b(Business\s+Intelligence|BI)\s+(Analyst|Developer|Engineer)\b',
    r'\b(Machine\s+Learning|ML|AI)\s+(Engineer|Scientist|Researcher)\b',
    r'\b(Data)\s+(Science|Analytics)\s+(Manager|Director|Lead)\b',
    # Software roles
    r'\b(Senior|Junior|Lead|Principal|Staff)?\s*(Software|Backend|Frontend|Full[\s-]?Stack)\s+(Engineer|Developer)\b',
    r'\b(DevOps|SRE|Platform|Infrastructure)\s+(Engineer|Architect)\b',
    r'\b(QA|Quality\s+Assurance|Test)\s+(Engineer|Analyst|Lead)\b',
    # Management
    r'\b(Engineering|Product|Project|Program)\s+(Manager|Director|Lead)\b',
    r'\b(VP|Vice\s+President|Director|Head)\s+of\s+[\w\s]+\b',
    r'\b(CTO|CEO|CFO|CIO|COO|Chief[\w\s]+Officer)\b',
    # Other common roles
    r'\b(Research)\s+(Scientist|Engineer|Assistant)\b',
    r'\b(Solutions?)\s+(Architect|Engineer|Consultant)\b',
    r'\b(Technical)\s+(Lead|Architect|Writer|Consultant)\b',
    r'\b(Consultant|Analyst|Specialist|Coordinator|Administrator)\b',
    r'\b(Intern|Internship|Co-op|Trainee)\b',
]

# Duration patterns
DURATION_PATTERNS = [
    r'\b(\d{1,2}/\d{4})\s*[-–]\s*(Present|\d{1,2}/\d{4})\b',
    r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}\s*[-–]\s*(Present|(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4})\b',
    r'\b\d{4}\s*[-–]\s*(Present|\d{4})\b',
    r'\b(\d+)\s*(years?|months?|yrs?|mos?)\s*(of\s+experience)?\b',
]

# Field of study patterns
FIELD_OF_STUDY_PATTERNS = [
    r'\b(Computer\s+Science|Information\s+Technology|Software\s+Engineering)\b',
    r'\b(Data\s+Science|Data\s+Analytics|Business\s+Analytics)\b',
    r'\b(Electrical\s+Engineering|Mechanical\s+Engineering|Civil\s+Engineering)\b',
    r'\b(Mathematics|Statistics|Physics|Chemistry|Biology)\b',
    r'\b(Economics|Finance|Accounting|Business\s+Administration)\b',
    r'\b(Psychology|Sociology|Political\s+Science)\b',
    r'\b(Information\s+Systems|Management\s+Information\s+Systems|MIS)\b',
    r'\b(Artificial\s+Intelligence|Machine\s+Learning)\b',
]


# =============================================================================
# Skill Taxonomy (Extended)
# =============================================================================

SKILL_TAXONOMY = {
    'programming_languages': [
        'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'c',
        'r', 'sql', 'scala', 'go', 'golang', 'rust', 'ruby', 'php',
        'swift', 'kotlin', 'matlab', 'perl', 'bash', 'shell', 'powershell',
        'html', 'css', 'sass', 'less', 'vba', 'sas', 'stata', 'julia',
        'groovy', 'lua', 'haskell', 'erlang', 'elixir', 'clojure', 'f#',
        'objective-c', 'assembly', 'cobol', 'fortran', 'lisp', 'prolog'
    ],
    'ml_ai_frameworks': [
        'tensorflow', 'pytorch', 'keras', 'scikit-learn', 'sklearn',
        'xgboost', 'lightgbm', 'catboost', 'hugging face', 'transformers',
        'spacy', 'nltk', 'gensim', 'opencv', 'yolo', 'detectron',
        'mlflow', 'kubeflow', 'ray', 'dask', 'h2o', 'automl',
        'langchain', 'llama', 'openai', 'anthropic', 'bert', 'gpt',
        'stable diffusion', 'midjourney', 'dall-e', 'whisper',
        'fastai', 'caffe', 'mxnet', 'theano', 'paddlepaddle'
    ],
    'data_science': [
        'machine learning', 'deep learning', 'neural networks', 'nlp',
        'natural language processing', 'computer vision', 'reinforcement learning',
        'statistical analysis', 'data analysis', 'predictive modeling',
        'time series', 'regression', 'classification', 'clustering',
        'a/b testing', 'hypothesis testing', 'feature engineering',
        'model deployment', 'mlops', 'data mining', 'etl',
        'anomaly detection', 'recommendation systems', 'sentiment analysis',
        'named entity recognition', 'ner', 'text classification',
        'image classification', 'object detection', 'segmentation',
        'dimensionality reduction', 'pca', 'tsne', 'umap'
    ],
    'data_tools': [
        'pandas', 'numpy', 'scipy', 'polars', 'vaex', 'modin',
        'jupyter', 'jupyter notebook', 'jupyterlab', 'colab',
        'anaconda', 'conda', 'pip', 'poetry', 'virtualenv'
    ],
    'visualization': [
        'tableau', 'power bi', 'powerbi', 'looker', 'qlik', 'qlikview',
        'matplotlib', 'seaborn', 'plotly', 'd3.js', 'd3', 'ggplot',
        'altair', 'bokeh', 'grafana', 'superset', 'metabase',
        'data visualization', 'dashboards', 'reporting', 'kibana',
        'chartjs', 'highcharts', 'echarts', 'folium', 'kepler.gl'
    ],
    'cloud_platforms': [
        'aws', 'amazon web services', 'azure', 'microsoft azure',
        'gcp', 'google cloud', 'google cloud platform',
        'ec2', 's3', 'lambda', 'sagemaker', 'redshift', 'athena',
        'azure ml', 'azure databricks', 'bigquery', 'dataflow',
        'cloud computing', 'serverless', 'iaas', 'paas', 'saas',
        'heroku', 'digitalocean', 'linode', 'vultr', 'cloudflare',
        'vercel', 'netlify', 'firebase', 'supabase'
    ],
    'databases': [
        'sql', 'mysql', 'postgresql', 'postgres', 'oracle', 'sql server',
        'mssql', 'sqlite', 'mariadb', 'mongodb', 'cassandra', 'redis',
        'elasticsearch', 'neo4j', 'dynamodb', 'cosmosdb', 'firebase',
        'snowflake', 'databricks', 'redshift', 'bigquery',
        'database design', 'data modeling', 'nosql', 'couchdb',
        'influxdb', 'timescaledb', 'clickhouse', 'cockroachdb',
        'supabase', 'planetscale', 'fauna', 'arangodb'
    ],
    'big_data': [
        'spark', 'pyspark', 'hadoop', 'hive', 'kafka', 'flink',
        'airflow', 'luigi', 'prefect', 'dagster', 'nifi',
        'data pipeline', 'data warehouse', 'data lake', 'etl',
        'data engineering', 'batch processing', 'stream processing',
        'presto', 'trino', 'dbt', 'fivetran', 'stitch', 'airbyte',
        'delta lake', 'iceberg', 'hudi', 'lakehouse'
    ],
    'devops_tools': [
        'docker', 'kubernetes', 'k8s', 'git', 'github', 'gitlab',
        'bitbucket', 'jenkins', 'ci/cd', 'terraform', 'ansible',
        'linux', 'unix', 'bash', 'shell scripting',
        'api', 'rest api', 'graphql', 'microservices',
        'prometheus', 'grafana', 'datadog', 'splunk', 'elk',
        'nginx', 'apache', 'haproxy', 'traefik',
        'helm', 'argocd', 'circleci', 'travis ci', 'github actions'
    ],
    'web_frameworks': [
        'react', 'angular', 'vue', 'svelte', 'next.js', 'nuxt',
        'node.js', 'nodejs', 'express', 'nestjs', 'fastify',
        'django', 'flask', 'fastapi', 'spring', 'spring boot',
        'rails', 'ruby on rails', 'laravel', 'symfony',
        'asp.net', '.net core', 'blazor'
    ],
    'frontend': [
        'html', 'css', 'javascript', 'typescript',
        'tailwind', 'bootstrap', 'material ui', 'chakra ui',
        'styled components', 'emotion', 'sass', 'less',
        'webpack', 'vite', 'parcel', 'rollup', 'esbuild',
        'redux', 'mobx', 'zustand', 'recoil', 'jotai',
        'react query', 'swr', 'apollo', 'relay'
    ],
    'mobile': [
        'react native', 'flutter', 'swift', 'kotlin',
        'ios', 'android', 'xamarin', 'ionic', 'cordova',
        'expo', 'swiftui', 'jetpack compose'
    ],
    'testing': [
        'pytest', 'unittest', 'jest', 'mocha', 'jasmine',
        'selenium', 'cypress', 'playwright', 'puppeteer',
        'junit', 'testng', 'cucumber', 'behave',
        'unit testing', 'integration testing', 'e2e testing',
        'tdd', 'bdd', 'test automation'
    ],
    'business_tools': [
        'excel', 'microsoft excel', 'google sheets', 'spreadsheets',
        'alteryx', 'sap', 'salesforce', 'hubspot', 'jira', 'confluence',
        'asana', 'trello', 'slack', 'teams', 'zoom',
        'powerpoint', 'google slides', 'word', 'google docs',
        'notion', 'airtable', 'monday.com', 'clickup', 'linear'
    ],
    'analytics_platforms': [
        'google analytics', 'adobe analytics', 'mixpanel', 'amplitude',
        'segment', 'heap', 'hotjar', 'optimizely', 'vwo',
        'data studio', 'looker studio', 'fullstory', 'pendo'
    ],
    'soft_skills': [
        'communication', 'leadership', 'teamwork', 'problem solving',
        'critical thinking', 'project management', 'agile', 'scrum',
        'stakeholder management', 'presentation', 'collaboration',
        'time management', 'mentoring', 'coaching', 'negotiation',
        'conflict resolution', 'decision making', 'adaptability'
    ],
    'methodologies': [
        'agile', 'scrum', 'kanban', 'lean', 'waterfall',
        'devops', 'devsecops', 'gitops', 'infrastructure as code',
        'continuous integration', 'continuous deployment',
        'test driven development', 'behavior driven development',
        'pair programming', 'code review', 'design patterns'
    ]
}

# Flatten skills for lookup
ALL_SKILLS: Set[str] = set()
SKILL_TO_CATEGORY: Dict[str, str] = {}
for category, skills in SKILL_TAXONOMY.items():
    for skill in skills:
        ALL_SKILLS.add(skill.lower())
        SKILL_TO_CATEGORY[skill.lower()] = category


# =============================================================================
# Data Classes for Extracted Entities
# =============================================================================

@dataclass
class ExtractedEntity:
    """Single extracted entity"""
    text: str
    label: str
    start: int
    end: int
    confidence: float = 1.0
    metadata: Dict = field(default_factory=dict)


@dataclass
class ResumeEntities:
    """All entities extracted from a resume"""
    skills: List[ExtractedEntity] = field(default_factory=list)
    degrees: List[ExtractedEntity] = field(default_factory=list)
    institutions: List[ExtractedEntity] = field(default_factory=list)
    fields_of_study: List[ExtractedEntity] = field(default_factory=list)
    job_titles: List[ExtractedEntity] = field(default_factory=list)
    companies: List[ExtractedEntity] = field(default_factory=list)
    durations: List[ExtractedEntity] = field(default_factory=list)
    certifications: List[ExtractedEntity] = field(default_factory=list)
    locations: List[ExtractedEntity] = field(default_factory=list)
    dates: List[ExtractedEntity] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'skills': [{'text': e.text, 'category': e.metadata.get('category', '')} for e in self.skills],
            'degrees': [e.text for e in self.degrees],
            'institutions': [e.text for e in self.institutions],
            'fields_of_study': [e.text for e in self.fields_of_study],
            'job_titles': [e.text for e in self.job_titles],
            'companies': [e.text for e in self.companies],
            'durations': [e.text for e in self.durations],
            'certifications': [e.text for e in self.certifications],
            'locations': [e.text for e in self.locations],
            'dates': [e.text for e in self.dates]
        }

    def get_skills_flat(self) -> List[str]:
        """Get skills as flat list"""
        return [e.text.lower() for e in self.skills]

    def get_skills_by_category(self) -> Dict[str, List[str]]:
        """Get skills grouped by category"""
        result = defaultdict(list)
        for entity in self.skills:
            category = entity.metadata.get('category', 'other')
            result[category].append(entity.text.lower())
        return dict(result)


# =============================================================================
# Resume NER Extractor
# =============================================================================

class ResumeNERExtractor:
    """
    Named Entity Recognition for Resumes

    Combines:
    1. spaCy's pre-trained NER (ORG, GPE, DATE, PERSON)
    2. Rule-based pattern matching for resume-specific entities
    3. Taxonomy-based skill extraction
    """

    def __init__(self):
        self.nlp = nlp
        self._setup_matchers()
        self._compile_patterns()

    def _setup_matchers(self):
        """Setup spaCy matchers"""
        self.matcher = Matcher(self.nlp.vocab)
        self.phrase_matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")

        # Add skill phrases to phrase matcher
        skill_patterns = [self.nlp.make_doc(skill) for skill in ALL_SKILLS]
        self.phrase_matcher.add("SKILL", skill_patterns)

    def _compile_patterns(self):
        """Compile regex patterns"""
        self.degree_patterns = [re.compile(
            p, re.IGNORECASE) for p in DEGREE_PATTERNS]
        self.cert_patterns = [re.compile(p, re.IGNORECASE)
                              for p in CERTIFICATION_PATTERNS]
        self.job_title_patterns = [re.compile(
            p, re.IGNORECASE) for p in JOB_TITLE_PATTERNS]
        self.duration_patterns = [re.compile(
            p, re.IGNORECASE) for p in DURATION_PATTERNS]
        self.field_patterns = [re.compile(
            p, re.IGNORECASE) for p in FIELD_OF_STUDY_PATTERNS]

        # Skill patterns for word-boundary matching
        self.skill_patterns = {}
        for skill in ALL_SKILLS:
            escaped = re.escape(skill)
            self.skill_patterns[skill] = re.compile(
                r'\b' + escaped + r'\b', re.IGNORECASE)

    def extract(self, text: str) -> ResumeEntities:
        """Extract all entities from resume text"""
        entities = ResumeEntities()

        # Process with spaCy
        doc = self.nlp(text)

        # 1. Extract skills using taxonomy matching
        entities.skills = self._extract_skills(text)

        # 2. Extract degrees
        entities.degrees = self._extract_with_patterns(
            text, self.degree_patterns, 'DEGREE')

        # 3. Extract certifications
        entities.certifications = self._extract_with_patterns(
            text, self.cert_patterns, 'CERTIFICATION')

        # 4. Extract job titles
        entities.job_titles = self._extract_with_patterns(
            text, self.job_title_patterns, 'JOB_TITLE')

        # 5. Extract durations
        entities.durations = self._extract_with_patterns(
            text, self.duration_patterns, 'DURATION')

        # 6. Extract fields of study
        entities.fields_of_study = self._extract_with_patterns(
            text, self.field_patterns, 'FIELD_OF_STUDY')

        # 7. Extract spaCy entities (ORG, GPE, DATE)
        entities.companies = self._extract_spacy_orgs(doc)
        entities.locations = self._extract_spacy_entities(
            doc, 'GPE', 'LOCATION')
        entities.dates = self._extract_spacy_entities(doc, 'DATE', 'DATE')

        # 8. Extract institutions (from ORG entities + patterns)
        entities.institutions = self._extract_institutions(doc, text)

        return entities

    def _extract_skills(self, text: str) -> List[ExtractedEntity]:
        """Extract skills using taxonomy matching"""
        found_skills = []
        text_lower = text.lower()

        for skill, pattern in self.skill_patterns.items():
            match = pattern.search(text_lower)
            if match:
                category = SKILL_TO_CATEGORY.get(skill, 'other')
                found_skills.append(ExtractedEntity(
                    text=skill,
                    label='SKILL',
                    start=match.start(),
                    end=match.end(),
                    confidence=1.0,
                    metadata={'category': category}
                ))

        return found_skills

    def _extract_with_patterns(
        self,
        text: str,
        patterns: List[re.Pattern],
        label: str
    ) -> List[ExtractedEntity]:
        """Extract entities using regex patterns"""
        found = []
        seen_texts = set()

        for pattern in patterns:
            for match in pattern.finditer(text):
                matched_text = match.group(0).strip()
                # Normalize and deduplicate
                normalized = ' '.join(matched_text.split())
                if normalized.lower() not in seen_texts:
                    seen_texts.add(normalized.lower())
                    found.append(ExtractedEntity(
                        text=normalized,
                        label=label,
                        start=match.start(),
                        end=match.end(),
                        confidence=0.9
                    ))

        return found

    def _extract_spacy_orgs(self, doc: Doc) -> List[ExtractedEntity]:
        """Extract organizations using spaCy, filtering out tools/skills and false positives"""
        companies = []
        seen = set()

        # Common false positives in resumes
        false_positive_orgs = ALL_SKILLS | {
            'gpa', 'bc', 'ca', 'ny', 'present', 'current',
            'data analytics engineering', 'computer science', 'data science',
            'machine learning', 'artificial intelligence', 'business administration',
            'b.s.', 'm.s.', 'ph.d.', 'mba', 'b.a.', 'm.a.',
            'bachelor', 'master', 'doctorate', 'degree',
            # Common resume section words that get misclassified
            'experience', 'education', 'skills', 'projects', 'summary',
            'responsibilities', 'achievements', 'certifications'
        }

        # Words that indicate it's likely NOT a company
        exclude_patterns = [
            'university', 'college', 'institute', 'school', 'academy',  # These are institutions
            'b.s', 'm.s', 'ph.d', 'mba', 'bachelor', 'master',
            'created', 'built', 'developed', 'analyzed', 'reduced', 'increased',
            'statistics', 'science', 'engineering', 'analytics'
        ]

        for ent in doc.ents:
            if ent.label_ == 'ORG':
                text_lower = ent.text.lower().strip()

                # Skip if in false positives
                if text_lower in false_positive_orgs:
                    continue

                # Skip if contains exclude patterns
                if any(pattern in text_lower for pattern in exclude_patterns):
                    continue

                # Skip very short or very long entities
                if len(text_lower) < 2 or len(text_lower) > 50:
                    continue

                # Skip if starts with bullet point or special chars
                if ent.text.startswith('•') or ent.text.startswith('-'):
                    continue

                if text_lower not in seen:
                    seen.add(text_lower)
                    companies.append(ExtractedEntity(
                        text=ent.text.strip(),
                        label='COMPANY',
                        start=ent.start_char,
                        end=ent.end_char,
                        confidence=0.8
                    ))

        return companies

    def _extract_spacy_entities(
        self,
        doc: Doc,
        spacy_label: str,
        custom_label: str
    ) -> List[ExtractedEntity]:
        """Extract entities by spaCy label"""
        entities = []
        seen = set()

        # Filter out skills and common false positives
        false_positives = ALL_SKILLS | {
            # State abbreviations often misclassified
            'gpa', 'bc', 'ca', 'ny', 'tx', 'wa', 'ma',
            'm.s.', 'b.s.', 'b.a.', 'm.a.', 'ph.d.', 'mba',  # Degrees
            'present', 'current'
        }

        for ent in doc.ents:
            if ent.label_ == spacy_label:
                text_lower = ent.text.lower().strip()
                # Skip if it's a skill or known false positive
                if text_lower not in false_positives and text_lower not in seen:
                    # Additional filter: skip very short entities for locations
                    if spacy_label == 'GPE' and len(text_lower) < 3:
                        continue
                    seen.add(text_lower)
                    entities.append(ExtractedEntity(
                        text=ent.text,
                        label=custom_label,
                        start=ent.start_char,
                        end=ent.end_char,
                        confidence=0.8
                    ))

        return entities

    def _extract_institutions(self, doc: Doc, text: str) -> List[ExtractedEntity]:
        """Extract educational institutions"""
        institutions = []
        seen = set()

        # Pattern for universities/colleges
        institution_pattern = re.compile(
            r'\b((?:[\w\s]+)?(?:University|College|Institute|School|Academy)(?:\s+of\s+[\w\s]+)?)\b',
            re.IGNORECASE
        )

        for match in institution_pattern.finditer(text):
            inst_text = match.group(0).strip()
            if inst_text.lower() not in seen and len(inst_text) > 5:
                seen.add(inst_text.lower())
                institutions.append(ExtractedEntity(
                    text=inst_text,
                    label='INSTITUTION',
                    start=match.start(),
                    end=match.end(),
                    confidence=0.9
                ))

        # Also check spaCy ORG entities for educational keywords
        edu_keywords = ['university', 'college',
                        'institute', 'school', 'academy']
        for ent in doc.ents:
            if ent.label_ == 'ORG':
                if any(kw in ent.text.lower() for kw in edu_keywords):
                    if ent.text.lower() not in seen:
                        seen.add(ent.text.lower())
                        institutions.append(ExtractedEntity(
                            text=ent.text,
                            label='INSTITUTION',
                            start=ent.start_char,
                            end=ent.end_char,
                            confidence=0.85
                        ))

        return institutions

    def extract_from_section(self, text: str, section_type: str) -> ResumeEntities:
        """
        Extract entities with section-aware context.
        Adjusts extraction strategy based on section type.
        """
        entities = self.extract(text)

        # Boost confidence for entities in relevant sections
        if section_type == 'education':
            for e in entities.degrees:
                e.confidence = min(1.0, e.confidence + 0.1)
            for e in entities.institutions:
                e.confidence = min(1.0, e.confidence + 0.1)
        elif section_type == 'experience':
            for e in entities.job_titles:
                e.confidence = min(1.0, e.confidence + 0.1)
            for e in entities.companies:
                e.confidence = min(1.0, e.confidence + 0.1)
        elif section_type == 'skills':
            for e in entities.skills:
                e.confidence = min(1.0, e.confidence + 0.1)

        return entities


# =============================================================================
# NER Evaluation Metrics
# =============================================================================

class NEREvaluator:
    """Evaluate NER performance against ground truth"""

    @staticmethod
    def evaluate(
        predicted: List[str],
        ground_truth: List[str]
    ) -> Dict[str, float]:
        """Calculate precision, recall, F1"""
        pred_set = set(s.lower() for s in predicted)
        truth_set = set(s.lower() for s in ground_truth)

        if not truth_set:
            return {'precision': 1.0, 'recall': 1.0, 'f1': 1.0}

        true_positives = len(pred_set & truth_set)
        false_positives = len(pred_set - truth_set)
        false_negatives = len(truth_set - pred_set)

        precision = true_positives / \
            (true_positives + false_positives) if (true_positives +
                                                   false_positives) > 0 else 0
        recall = true_positives / \
            (true_positives + false_negatives) if (true_positives +
                                                   false_negatives) > 0 else 0
        f1 = 2 * precision * recall / \
            (precision + recall) if (precision + recall) > 0 else 0

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives
        }

    @staticmethod
    def evaluate_all_entities(
        predicted: ResumeEntities,
        ground_truth: Dict[str, List[str]]
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate all entity types"""
        results = {}

        entity_mappings = {
            'skills': predicted.get_skills_flat(),
            'degrees': [e.text for e in predicted.degrees],
            'institutions': [e.text for e in predicted.institutions],
            'job_titles': [e.text for e in predicted.job_titles],
            'companies': [e.text for e in predicted.companies],
            'certifications': [e.text for e in predicted.certifications]
        }

        for entity_type, predictions in entity_mappings.items():
            if entity_type in ground_truth:
                results[entity_type] = NEREvaluator.evaluate(
                    predictions,
                    ground_truth[entity_type]
                )

        return results


# =============================================================================
# Backward Compatibility
# =============================================================================

class EntityExtractor(ResumeNERExtractor):
    """Alias for backward compatibility"""

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract entities (backward compatible)"""
        entities = self.extract(text)
        return {
            'organizations': [e.text for e in entities.companies],
            'locations': [e.text for e in entities.locations],
            'dates': [e.text for e in entities.dates],
            'persons': [],  # Not extracted in new version
            'degrees': [e.text for e in entities.degrees],
            'job_titles': [e.text for e in entities.job_titles],
            'institutions': [e.text for e in entities.institutions],
            'certifications': [e.text for e in entities.certifications]
        }

    def extract_skills(self, text: str) -> List[str]:
        """Extract skills (backward compatible)"""
        entities = self.extract(text)
        return entities.get_skills_flat()


# =============================================================================
# Main / Demo
# =============================================================================

if __name__ == "__main__":
    # Demo
    sample_resume = """
    JOHN DOE
    Senior Data Scientist | john.doe@email.com | San Francisco, CA
    
    SUMMARY
    Data Scientist with 5+ years of experience in machine learning, deep learning,
    and statistical analysis. Expert in Python, TensorFlow, and cloud platforms.
    
    EXPERIENCE
    
    Senior Data Scientist, Google (2021 - Present)
    • Developed NLP models using PyTorch and Transformers
    • Built recommendation systems serving 100M+ users
    • Led team of 5 data scientists
    
    Data Scientist, Meta (2019 - 2021)
    • Created computer vision pipelines with TensorFlow
    • Deployed models on AWS SageMaker
    • Improved model accuracy by 25%
    
    EDUCATION
    Ph.D. in Computer Science, Stanford University, 2019
    B.S. in Mathematics, MIT, 2014
    
    SKILLS
    Python, PyTorch, TensorFlow, Keras, SQL, AWS, Docker, Kubernetes,
    Machine Learning, Deep Learning, NLP, Computer Vision, A/B Testing
    
    CERTIFICATIONS
    AWS Certified Machine Learning Specialty
    Google Cloud Professional Data Engineer
    """

    extractor = ResumeNERExtractor()
    entities = extractor.extract(sample_resume)

    print("=" * 60)
    print("RESUME NER EXTRACTION RESULTS")
    print("=" * 60)

    print(f"\nSKILLS ({len(entities.skills)}):")
    skills_by_cat = entities.get_skills_by_category()
    for cat, skills in skills_by_cat.items():
        print(f"  {cat}: {', '.join(skills)}")

    print(f"\nDEGREES ({len(entities.degrees)}):")
    for e in entities.degrees:
        print(f"  - {e.text}")

    print(f"\nINSTITUTIONS ({len(entities.institutions)}):")
    for e in entities.institutions:
        print(f"  - {e.text}")

    print(f"\nJOB TITLES ({len(entities.job_titles)}):")
    for e in entities.job_titles:
        print(f"  - {e.text}")

    print(f"\nCOMPANIES ({len(entities.companies)}):")
    for e in entities.companies:
        print(f"  - {e.text}")

    print(f"\nCERTIFICATIONS ({len(entities.certifications)}):")
    for e in entities.certifications:
        print(f"  - {e.text}")

    print(f"\nLOCATIONS ({len(entities.locations)}):")
    for e in entities.locations:
        print(f"  - {e.text}")
