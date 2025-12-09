"""
Enhanced Resume-Job Matching Pipeline v2.0
==========================================
Adds BERT/Sentence Transformer embeddings for semantic matching
alongside TF-IDF baseline for comparison.

Author: Noel John
Course: IE 7500 Applied NLP, Northeastern University
"""

import re
import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict

import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sentence Transformers for BERT embeddings
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMER_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMER_AVAILABLE = False
    print("Warning: sentence-transformers not installed. Using TF-IDF only.")

# Load spaCy
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import os
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class SkillMatch:
    """Represents a skill matching result"""
    matched: List[str] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)
    extra: List[str] = field(default_factory=list)
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0


@dataclass
class MatchResult:
    """Complete matching result between resume and job"""
    overall_score: float
    semantic_score_tfidf: float
    semantic_score_bert: Optional[float]
    skill_match: SkillMatch
    section_scores: Dict[str, float]
    recommendations: List[str]
    fit_level: str  # "Strong", "Moderate", "Weak"


# =============================================================================
# Extended Skill Taxonomy
# =============================================================================

SKILL_TAXONOMY = {
    'programming_languages': [
        'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'c',
        'r', 'sql', 'scala', 'go', 'golang', 'rust', 'ruby', 'php',
        'swift', 'kotlin', 'matlab', 'perl', 'bash', 'shell', 'powershell',
        'html', 'css', 'sass', 'less', 'vba', 'sas', 'stata'
    ],
    'ml_ai_frameworks': [
        'tensorflow', 'pytorch', 'keras', 'scikit-learn', 'sklearn',
        'xgboost', 'lightgbm', 'catboost', 'hugging face', 'transformers',
        'spacy', 'nltk', 'gensim', 'opencv', 'yolo', 'detectron',
        'mlflow', 'kubeflow', 'ray', 'dask', 'h2o', 'automl',
        'langchain', 'llama', 'openai api', 'anthropic api'
    ],
    'data_science': [
        'machine learning', 'deep learning', 'neural networks', 'nlp',
        'natural language processing', 'computer vision', 'reinforcement learning',
        'statistical analysis', 'data analysis', 'predictive modeling',
        'time series', 'regression', 'classification', 'clustering',
        'a/b testing', 'hypothesis testing', 'feature engineering',
        'model deployment', 'mlops', 'data mining', 'etl'
    ],
    'visualization': [
        'tableau', 'power bi', 'powerbi', 'looker', 'qlik', 'qlikview',
        'matplotlib', 'seaborn', 'plotly', 'd3.js', 'd3', 'ggplot',
        'altair', 'bokeh', 'grafana', 'superset', 'metabase',
        'data visualization', 'dashboards', 'dashboard', 'reporting'
    ],
    'cloud_platforms': [
        'aws', 'amazon web services', 'azure', 'microsoft azure',
        'gcp', 'google cloud', 'google cloud platform',
        'ec2', 's3', 'lambda', 'sagemaker', 'redshift', 'athena',
        'azure ml', 'azure databricks', 'bigquery', 'dataflow',
        'cloud computing', 'serverless', 'iaas', 'paas', 'saas'
    ],
    'databases': [
        'sql', 'mysql', 'postgresql', 'postgres', 'oracle', 'sql server',
        'mssql', 'sqlite', 'mariadb', 'mongodb', 'cassandra', 'redis',
        'elasticsearch', 'neo4j', 'dynamodb', 'cosmosdb', 'firebase',
        'snowflake', 'databricks', 'redshift', 'bigquery',
        'database design', 'data modeling', 'nosql'
    ],
    'big_data': [
        'spark', 'pyspark', 'hadoop', 'hive', 'kafka', 'flink',
        'airflow', 'luigi', 'prefect', 'dagster', 'nifi',
        'data pipeline', 'data warehouse', 'data lake', 'etl',
        'data engineering', 'batch processing', 'stream processing'
    ],
    'devops_tools': [
        'docker', 'kubernetes', 'k8s', 'git', 'github', 'gitlab',
        'bitbucket', 'jenkins', 'ci/cd', 'terraform', 'ansible',
        'linux', 'unix', 'bash', 'shell scripting',
        'api', 'rest api', 'graphql', 'microservices'
    ],
    'web_development': [
        'react', 'angular', 'vue', 'node.js', 'nodejs', 'express',
        'django', 'flask', 'fastapi', 'spring', 'spring boot',
        'html', 'css', 'javascript', 'typescript',
        'frontend', 'backend', 'full stack', 'fullstack'
    ],
    'business_tools': [
        'excel', 'microsoft excel', 'google sheets', 'spreadsheets',
        'alteryx', 'sap', 'salesforce', 'hubspot', 'jira', 'confluence',
        'asana', 'trello', 'slack', 'microsoft teams', 'zoom',
        'powerpoint', 'google slides', 'word', 'google docs'
    ],
    'analytics_platforms': [
        'google analytics', 'adobe analytics', 'mixpanel', 'amplitude',
        'segment', 'heap', 'hotjar', 'optimizely', 'vwo',
        'data studio', 'looker studio'
    ],
    'soft_skills': [
        'communication', 'leadership', 'project management', 'agile', 'scrum',
        'stakeholder management', 'presentation skills'
    ]
}

# Flatten for quick lookup
ALL_SKILLS = set()
SKILL_TO_CATEGORY = {}
for category, skills in SKILL_TAXONOMY.items():
    for skill in skills:
        ALL_SKILLS.add(skill.lower())
        SKILL_TO_CATEGORY[skill.lower()] = category


# =============================================================================
# Entity Extractor (Enhanced)
# =============================================================================

class EnhancedEntityExtractor:
    """Extract entities and skills with improved accuracy"""

    def __init__(self):
        self.nlp = nlp
        self.skill_patterns = self._compile_skill_patterns()

    def _compile_skill_patterns(self) -> Dict[str, re.Pattern]:
        """Pre-compile regex patterns for skill extraction"""
        patterns = {}
        for skill in ALL_SKILLS:
            # Handle multi-word skills and special characters
            escaped = re.escape(skill)
            # Allow for variations like "power bi" / "powerbi"
            pattern = r'\b' + escaped + r'\b'
            patterns[skill] = re.compile(pattern, re.IGNORECASE)
        return patterns

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract named entities using spaCy"""
        doc = self.nlp(text)

        # False positives to filter out
        false_positives = ALL_SKILLS | {
            'gpa', 'bc', 'ca', 'ny', 'present', 'current',
            'data analytics engineering', 'computer science', 'data science',
            'b.s.', 'm.s.', 'ph.d.', 'mba', 'bachelor', 'master',
            'experience', 'education', 'skills', 'projects', 'summary'
        }

        # Filter organizations
        organizations = []
        for ent in doc.ents:
            if ent.label_ == 'ORG':
                text_lower = ent.text.lower().strip()
                # Skip false positives
                if text_lower in false_positives:
                    continue
                # Skip if contains education keywords (those are institutions)
                if any(kw in text_lower for kw in ['university', 'college', 'institute', 'school']):
                    continue
                # Skip bullet points
                if ent.text.startswith('•') or ent.text.startswith('-'):
                    continue
                if len(text_lower) >= 2:
                    organizations.append(ent.text.strip())

        # Filter locations
        locations = []
        for ent in doc.ents:
            if ent.label_ == 'GPE':
                text_lower = ent.text.lower().strip()
                # Skip skills misclassified as locations
                if text_lower not in false_positives and len(text_lower) >= 3:
                    locations.append(ent.text)

        return {
            'organizations': list(set(organizations)),
            'locations': list(set(locations)),
            'dates': list(set(ent.text for ent in doc.ents if ent.label_ == 'DATE')),
            'persons': list(set(ent.text for ent in doc.ents if ent.label_ == 'PERSON')),
            'education': self._extract_education(text),
            'job_titles': self._extract_job_titles(text)
        }

    def extract_skills(self, text: str) -> Dict[str, List[str]]:
        """Extract skills organized by category"""
        text_lower = text.lower()
        found_skills = defaultdict(list)

        for skill, pattern in self.skill_patterns.items():
            if pattern.search(text_lower):
                category = SKILL_TO_CATEGORY.get(skill, 'other')
                found_skills[category].append(skill)

        # Deduplicate within categories
        return {cat: list(set(skills)) for cat, skills in found_skills.items()}

    def extract_skills_flat(self, text: str) -> List[str]:
        """Extract skills as flat list (for backward compatibility)"""
        categorized = self.extract_skills(text)
        return [skill for skills in categorized.values() for skill in skills]

    def _extract_education(self, text: str) -> List[str]:
        """Extract education-related entities"""
        education_patterns = [
            r'\b(Ph\.?D\.?|Doctor\s+of\s+Philosophy)\b',
            r'\b(M\.?S\.?|Master\'?s?\s+of\s+Science)\b',
            r'\b(B\.?S\.?|Bachelor\'?s?\s+of\s+Science)\b',
            r'\b(M\.?A\.?|Master\'?s?\s+of\s+Arts)\b',
            r'\b(B\.?A\.?|Bachelor\'?s?\s+of\s+Arts)\b',
            r'\b(MBA|Master\s+of\s+Business\s+Administration)\b',
            r'\b(\w+\s+University)\b',
            r'\b(University\s+of\s+\w+)\b',
            r'\b(\w+\s+College)\b',
            r'\b(\w+\s+Institute)\b'
        ]
        found = []
        for pattern in education_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            found.extend(matches)
        return list(set(found))

    def _extract_job_titles(self, text: str) -> List[str]:
        """Extract job titles"""
        title_patterns = [
            r'\b((?:Senior|Junior|Lead|Principal|Staff|Associate)?\s*Data\s+(?:Analyst|Scientist|Engineer))\b',
            r'\b((?:Senior|Junior|Lead)?\s*Business\s+(?:Analyst|Intelligence\s+Analyst))\b',
            r'\b((?:Senior|Junior|Lead|Principal|Staff)?\s*Software\s+(?:Engineer|Developer))\b',
            r'\b((?:Senior|Junior|Lead)?\s*Machine\s+Learning\s+Engineer)\b',
            r'\b((?:Senior|Junior|Lead)?\s*ML\s+Engineer)\b',
            r'\b((?:Senior|Junior)?\s*(?:Frontend|Backend|Full[\s-]?Stack)\s+(?:Engineer|Developer))\b',
            r'\b((?:Senior|Junior)?\s*DevOps\s+Engineer)\b',
            r'\b((?:Senior|Junior)?\s*Research\s+Scientist)\b',
            r'\b(Teaching\s+Assistant)\b',
            r'\b((?:Project|Product|Program)\s+Manager)\b'
        ]
        found = []
        for pattern in title_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            found.extend([m.strip() for m in matches if m.strip()])
        return list(set(found))


# =============================================================================
# Semantic Matcher (Enhanced with BERT)
# =============================================================================

class EnhancedSemanticMatcher:
    """Compute semantic similarity using TF-IDF and BERT embeddings"""

    def __init__(self, use_bert: bool = True):
        self.use_bert = use_bert and SENTENCE_TRANSFORMER_AVAILABLE

        # TF-IDF vectorizer
        self.tfidf = TfidfVectorizer(
            max_features=500,
            ngram_range=(1, 3),
            stop_words='english',
            min_df=1
        )

        # BERT model (lazy loading)
        self._bert_model = None

    @property
    def bert_model(self):
        """Lazy load BERT model"""
        if self._bert_model is None and self.use_bert:
            print("Loading Sentence Transformer model...")
            self._bert_model = SentenceTransformer('all-MiniLM-L6-v2')
        return self._bert_model

    def compute_tfidf_similarity(self, text1: str, text2: str) -> float:
        """Compute TF-IDF cosine similarity"""
        try:
            corpus = [text1.lower(), text2.lower()]
            tfidf_matrix = self.tfidf.fit_transform(corpus)
            similarity = cosine_similarity(
                tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(similarity * 100)
        except Exception:
            return 0.0

    def compute_bert_similarity(self, text1: str, text2: str) -> Optional[float]:
        """Compute BERT embedding cosine similarity"""
        if not self.use_bert:
            return None

        try:
            embeddings = self.bert_model.encode([text1, text2])
            similarity = cosine_similarity(
                [embeddings[0]], [embeddings[1]])[0][0]
            return float(similarity * 100)
        except Exception as e:
            print(f"BERT similarity error: {e}")
            return None

    def compute_section_similarities(
        self,
        resume_sections: Dict[str, str],
        job_text: str
    ) -> Dict[str, float]:
        """Compute similarity for each resume section"""
        section_scores = {}

        for section_name, section_text in resume_sections.items():
            if section_text.strip():
                if self.use_bert:
                    score = self.compute_bert_similarity(
                        section_text, job_text)
                else:
                    score = self.compute_tfidf_similarity(
                        section_text, job_text)
                section_scores[section_name] = score or 0.0

        return section_scores

    def compute_skill_overlap(
        self,
        resume_skills: List[str],
        job_skills: List[str]
    ) -> SkillMatch:
        """Calculate skill overlap metrics"""
        resume_set = set(s.lower() for s in resume_skills)
        job_set = set(s.lower() for s in job_skills)

        matched = resume_set & job_set
        missing = job_set - resume_set
        extra = resume_set - job_set

        precision = len(matched) / len(resume_set) if resume_set else 0
        recall = len(matched) / len(job_set) if job_set else 0
        f1 = 2 * precision * recall / \
            (precision + recall) if (precision + recall) > 0 else 0

        return SkillMatch(
            matched=sorted(list(matched)),
            missing=sorted(list(missing)),
            extra=sorted(list(extra)),
            precision=precision,
            recall=recall,
            f1_score=f1
        )


# =============================================================================
# Feedback Generator
# =============================================================================

class FeedbackGenerator:
    """Generate actionable feedback and recommendations"""

    def __init__(self):
        self.templates = self._load_templates()

    def _load_templates(self) -> Dict[str, List[str]]:
        """Load feedback templates"""
        return {
            'missing_skill': [
                "Consider highlighting experience with {skill} if you have any related projects.",
                "The job requires {skill}. You could mention relevant coursework or self-study.",
                "{skill} is a key requirement. Consider adding a project using this technology.",
            ],
            'matched_skill': [
                "Great! Your {skill} experience aligns well with the job requirements.",
                "Your proficiency in {skill} is a strong match for this position.",
            ],
            'category_gap': [
                "The job emphasizes {category} skills. Consider building experience in this area.",
                "You may want to strengthen your {category} skills for better alignment.",
            ],
            'strong_fit': [
                "Your profile shows strong alignment with this position.",
                "You have excellent coverage of the required skills.",
            ],
            'moderate_fit': [
                "You have a solid foundation for this role with some areas to develop.",
                "Consider emphasizing your relevant experience more prominently.",
            ],
            'weak_fit': [
                "This role may require significant skill development.",
                "Consider gaining experience in the key missing areas before applying.",
            ]
        }

    def generate_recommendations(
        self,
        skill_match: SkillMatch,
        overall_score: float,
        job_title: str = "this position"
    ) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []

        # Overall fit assessment
        if overall_score >= 70:
            fit_level = "Strong"
            recommendations.append(self.templates['strong_fit'][0])
        elif overall_score >= 40:
            fit_level = "Moderate"
            recommendations.append(self.templates['moderate_fit'][0])
        else:
            fit_level = "Weak"
            recommendations.append(self.templates['weak_fit'][0])

        # Highlight matched skills
        if skill_match.matched:
            top_matches = skill_match.matched[:3]
            recommendations.append(
                f"Emphasize your experience with: {', '.join(top_matches)}."
            )

        # Address missing skills
        if skill_match.missing:
            # Group by category
            missing_by_category = defaultdict(list)
            for skill in skill_match.missing:
                cat = SKILL_TO_CATEGORY.get(skill, 'other')
                missing_by_category[cat].append(skill)

            # Prioritize by category importance
            priority_order = ['programming_languages', 'cloud_platforms', 'databases',
                              'ml_ai_frameworks', 'big_data', 'devops_tools']

            for category in priority_order:
                if category in missing_by_category:
                    skills = missing_by_category[category][:2]
                    recommendations.append(
                        f"Priority development area: {', '.join(skills)} ({category.replace('_', ' ')})."
                    )
                    break

            # General missing skills advice
            if len(skill_match.missing) > 3:
                recommendations.append(
                    f"Focus on developing: {', '.join(skill_match.missing[:3])}."
                )

        return recommendations, fit_level

    def generate_cover_letter_snippet(
        self,
        matched_skills: List[str],
        job_title: str,
        company_name: str = "your company"
    ) -> str:
        """Generate a cover letter opening paragraph - personal and specific style"""
        if not matched_skills:
            return "Unable to generate cover letter snippet - no matched skills found."

        # Prioritize technical skills over soft skills
        technical_categories = {
            'programming_languages', 'ml_ai_frameworks', 'data_science', 'databases',
            'cloud_platforms', 'big_data', 'devops_tools', 'visualization',
            'web_frameworks', 'frontend', 'data_tools', 'testing'
        }

        # Separate technical and soft skills
        technical_skills = []
        for skill in matched_skills:
            category = SKILL_TO_CATEGORY.get(skill.lower(), 'other')
            if category in technical_categories:
                technical_skills.append(skill)

        if not technical_skills:
            technical_skills = matched_skills[:5]

        # Get top 3-4 technical skills
        top_skills = technical_skills[:4]
        skills_text = ", ".join(top_skills[:-1]) + f", and {top_skills[-1]}" if len(
            top_skills) > 1 else top_skills[0] if top_skills else "relevant technologies"

        # Generate a more personal, specific cover letter
        snippet = f"""Dear Hiring Manager,

The technical backbone of this {job_title} role - working with {skills_text} - is familiar ground for me. I've spent time building production pipelines where small mistakes multiply downstream, so I take the craft of data work seriously.

What draws me to this position isn't just the skill match. It's the opportunity to apply what I know while learning the specifics of your domain. I don't pretend to know everything about your industry yet, but I've consistently shown I can learn fast when given clean data and the right context.

I'd welcome the chance to discuss how my background in {top_skills[0] if top_skills else 'data analysis'} and related tools could contribute to your team's work.

Sincerely,
[Your Name]"""

        return snippet.strip()

    def generate_resume_suggestions(
        self,
        missing_skills: List[str],
        extra_skills: List[str]
    ) -> List[str]:
        """Suggest resume improvements"""
        suggestions = []

        # Keywords to add
        if missing_skills:
            suggestions.append(
                f"Keywords to consider adding (if applicable): {', '.join(missing_skills[:5])}"
            )

        # Skills to highlight
        if extra_skills:
            relevant_extra = [s for s in extra_skills if s in ALL_SKILLS][:3]
            if relevant_extra:
                suggestions.append(
                    f"Unique skills to highlight: {', '.join(relevant_extra)}"
                )

        return suggestions


# =============================================================================
# Main Pipeline
# =============================================================================

class ResumeJobMatcherV2:
    """Enhanced end-to-end matching pipeline"""

    def __init__(self, use_bert: bool = True):
        self.entity_extractor = EnhancedEntityExtractor()
        self.semantic_matcher = EnhancedSemanticMatcher(use_bert=use_bert)
        self.feedback_generator = FeedbackGenerator()

    def process_resume(self, resume_text: str) -> Dict:
        """Process resume with comprehensive extraction"""
        entities = self.entity_extractor.extract_entities(resume_text)
        skills_categorized = self.entity_extractor.extract_skills(resume_text)
        skills_flat = self.entity_extractor.extract_skills_flat(resume_text)

        # Extract sections
        sections = self._extract_sections(resume_text)

        return {
            'entities': entities,
            'skills': skills_flat,
            'skills_by_category': skills_categorized,
            'sections': sections,
            'text': resume_text,
            'word_count': len(resume_text.split())
        }

    def process_job(self, job_text: str) -> Dict:
        """Process job posting"""
        skills_categorized = self.entity_extractor.extract_skills(job_text)
        skills_flat = self.entity_extractor.extract_skills_flat(job_text)

        # Try to extract job title
        job_titles = self.entity_extractor._extract_job_titles(job_text)
        job_title = job_titles[0] if job_titles else "Position"

        return {
            'required_skills': skills_flat,
            'skills_by_category': skills_categorized,
            'job_title': job_title,
            'text': job_text
        }

    def match(
        self,
        resume_text: str,
        job_text: str,
        resume_skills: List[str],
        job_skills: List[str],
        resume_sections: Optional[Dict[str, str]] = None
    ) -> MatchResult:
        """Complete matching with all metrics"""

        # Semantic similarities
        tfidf_score = self.semantic_matcher.compute_tfidf_similarity(
            resume_text, job_text
        )
        bert_score = self.semantic_matcher.compute_bert_similarity(
            resume_text, job_text
        )

        # Skill overlap
        skill_match = self.semantic_matcher.compute_skill_overlap(
            resume_skills, job_skills
        )

        # Section-level analysis
        section_scores = {}
        if resume_sections:
            section_scores = self.semantic_matcher.compute_section_similarities(
                resume_sections, job_text
            )

        # Calculate overall score
        # Weight: 30% semantic (prefer BERT if available) + 70% skill match
        semantic_score = bert_score if bert_score is not None else tfidf_score
        overall_score = semantic_score * 0.3 + skill_match.f1_score * 100 * 0.7

        # Generate recommendations
        recommendations, fit_level = self.feedback_generator.generate_recommendations(
            skill_match, overall_score
        )

        return MatchResult(
            overall_score=overall_score,
            semantic_score_tfidf=tfidf_score,
            semantic_score_bert=bert_score,
            skill_match=skill_match,
            section_scores=section_scores,
            recommendations=recommendations,
            fit_level=fit_level
        )

    def _extract_sections(self, text: str) -> Dict[str, str]:
        """Extract resume sections"""
        sections = {
            'education': '',
            'experience': '',
            'skills': '',
            'projects': ''
        }

        patterns = {
            'education': r'(?i)(education|academic|qualification|degree)',
            'experience': r'(?i)(experience|employment|work\s+history|professional)',
            'skills': r'(?i)(skills|technical\s+skills|competencies|technologies)',
            'projects': r'(?i)(projects|portfolio|work\s+samples)'
        }

        text_lower = text.lower()

        for section, pattern in patterns.items():
            match = re.search(pattern, text)
            if match:
                start = match.start()
                # Find next section header or end
                next_match = None
                for other_section, other_pattern in patterns.items():
                    if other_section != section:
                        other_match = re.search(other_pattern, text[start+50:])
                        if other_match:
                            if next_match is None or other_match.start() < next_match:
                                next_match = other_match.start()

                end = start + 50 + next_match if next_match else len(text)
                sections[section] = text[start:end]

        return sections

    def get_full_analysis(
        self,
        resume_text: str,
        job_text: str
    ) -> Dict:
        """Get complete analysis with all outputs"""

        # Process both documents
        resume_data = self.process_resume(resume_text)
        job_data = self.process_job(job_text)

        # Match
        match_result = self.match(
            resume_text,
            job_text,
            resume_data['skills'],
            job_data['required_skills'],
            resume_data['sections']
        )

        # Generate cover letter snippet
        cover_letter = self.feedback_generator.generate_cover_letter_snippet(
            match_result.skill_match.matched,
            job_data['job_title']
        )

        # Resume suggestions
        resume_suggestions = self.feedback_generator.generate_resume_suggestions(
            match_result.skill_match.missing,
            match_result.skill_match.extra
        )

        return {
            'resume_data': resume_data,
            'job_data': job_data,
            'match_result': {
                'overall_score': match_result.overall_score,
                'semantic_score_tfidf': match_result.semantic_score_tfidf,
                'semantic_score_bert': match_result.semantic_score_bert,
                'skill_f1': match_result.skill_match.f1_score,
                'matched_skills': match_result.skill_match.matched,
                'missing_skills': match_result.skill_match.missing,
                'extra_skills': match_result.skill_match.extra,
                'fit_level': match_result.fit_level
            },
            'recommendations': match_result.recommendations,
            'cover_letter_snippet': cover_letter,
            'resume_suggestions': resume_suggestions
        }


# =============================================================================
# Backward Compatibility
# =============================================================================

# Alias for existing code
ResumeJobMatcher = ResumeJobMatcherV2
EntityExtractor = EnhancedEntityExtractor
SemanticMatcher = EnhancedSemanticMatcher


if __name__ == "__main__":
    # Quick test
    # Set True if sentence-transformers installed
    matcher = ResumeJobMatcherV2(use_bert=False)

    test_resume = """
    Data Analyst with 3 years experience in Python and SQL.
    Expert in Tableau, Power BI, and data visualization.
    Experience with PostgreSQL and MySQL databases.
    """

    test_job = """
    Business Intelligence Analyst
    Requirements: Python, SQL, Tableau, Power BI
    Nice to have: AWS, Docker, Machine Learning
    """

    result = matcher.get_full_analysis(test_resume, test_job)
    print(f"Overall Score: {result['match_result']['overall_score']:.1f}%")
    print(f"Fit Level: {result['match_result']['fit_level']}")
    print(f"Matched: {result['match_result']['matched_skills']}")
    print(f"Missing: {result['match_result']['missing_skills']}")
