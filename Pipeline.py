import re
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    nlp = spacy.load("en_core_web_sm")
except:
    import os
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


class EntityExtractor:
    """Extract entities using spaCy NER"""

    def __init__(self):
        self.nlp = nlp
        self.skill_taxonomy = {
            'programming': ['python', 'java', 'javascript', 'c++', 'r', 'sql', 'scala', 'go'],
            'ml_frameworks': ['tensorflow', 'pytorch', 'keras', 'scikit-learn', 'xgboost'],
            'visualization': ['tableau', 'power bi', 'looker', 'matplotlib', 'seaborn', 'plotly'],
            'cloud': ['aws', 'azure', 'gcp'],
            'databases': ['sql', 'postgresql', 'mysql', 'mongodb', 'redis'],
            'tools': ['docker', 'kubernetes', 'git', 'airflow', 'spark'],
            'analytics': ['machine learning', 'data analysis', 'statistical analysis', 'a/b testing'],
            'misc': ['streamlit', 'excel', 'alteryx'],
        }

    def extract_entities(self, text):
        """Extract named entities using spaCy NER"""
        doc = self.nlp(text)

        # Filter out known tools/frameworks from organizations
        known_tools = {'python', 'java', 'javascript', 'sql', 'r', 'go', 'scala', 'c++',
                       'tableau', 'power bi', 'looker', 'matplotlib', 'seaborn', 'plotly',
                       'tensorflow', 'pytorch', 'keras', 'scikit-learn', 'xgboost',
                       'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'git', 'airflow', 'spark',
                       'sqlite', 'postgresql', 'mysql', 'mongodb', 'redis', 'langchain', 'h2o automl'}

        organizations = [ent.text for ent in doc.ents if ent.label_ == 'ORG'
                         and ent.text.lower() not in known_tools]

        entities = {
            'organizations': list(set(organizations)),
            'locations': [ent.text for ent in doc.ents if ent.label_ == 'GPE'],
            'dates': [ent.text for ent in doc.ents if ent.label_ == 'DATE'],
            'persons': [ent.text for ent in doc.ents if ent.label_ == 'PERSON']
        }
        return entities

    def extract_skills(self, text):
        """Extract skills using taxonomy matching"""
        text_lower = text.lower()
        found_skills = []

        for category, skills in self.skill_taxonomy.items():
            for skill in skills:
                pattern = r'\b' + re.escape(skill) + r'\b'
                if re.search(pattern, text_lower):
                    found_skills.append(skill)

        return list(set(found_skills))


class SemanticMatcher:
    """Compute semantic similarity between texts"""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=200, ngram_range=(1, 2), stop_words='english')

    def compute_similarity(self, text1, text2):
        """Compute TF-IDF cosine similarity"""
        corpus = [text1.lower(), text2.lower()]
        tfidf_matrix = self.vectorizer.fit_transform(corpus)
        similarity = cosine_similarity(
            tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return similarity * 100

    def skill_overlap(self, resume_skills, job_skills):
        """Calculate skill overlap metrics"""
        resume_set = set(resume_skills)
        job_set = set(job_skills)

        intersection = resume_set & job_set

        precision = len(intersection) / len(resume_set) if resume_set else 0
        recall = len(intersection) / len(job_set) if job_set else 0
        f1 = 2 * (precision * recall) / (precision +
                                         recall) if (precision + recall) > 0 else 0

        return {
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'matched_skills': list(intersection),
            'missing_skills': list(job_set - resume_set),
        }


class ResumeJobMatcher:
    """End-to-end matching pipeline"""

    def __init__(self):
        self.entity_extractor = EntityExtractor()
        self.semantic_matcher = SemanticMatcher()

    def process_resume(self, resume_text):
        """Process resume with entity extraction"""
        entities = self.entity_extractor.extract_entities(resume_text)
        skills = self.entity_extractor.extract_skills(resume_text)

        return {
            'entities': entities,
            'skills': skills,
            'text': resume_text
        }

    def process_job(self, job_text):
        """Process job posting"""
        skills = self.entity_extractor.extract_skills(job_text)

        return {
            'required_skills': skills,
            'text': job_text
        }

    def match(self, resume_text, job_text, resume_skills, job_skills):
        """Match resume to job"""
        # Semantic similarity
        semantic_score = self.semantic_matcher.compute_similarity(
            resume_text, job_text)

        # Skill overlap
        skill_match = self.semantic_matcher.skill_overlap(
            resume_skills, job_skills)

        # Overall score (40% semantic + 60% skill)
        overall_score = semantic_score * 0.4 + \
            skill_match['f1_score'] * 100 * 0.6

        return {
            'overall_score': overall_score,
            'semantic_score': semantic_score,
            'skill_match': skill_match,
            'matched_skills': skill_match['matched_skills'],
            'missing_skills': skill_match['missing_skills'],
        }
