"""
Enhanced Data Loader for Resume Job Matcher
============================================
Improved text preprocessing and section extraction.

Author: Noel John
Course: IE 7500 Applied NLP, Northeastern University
"""

import re
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import pdfplumber
from docx import Document


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ResumeData:
    """Structured resume data"""
    raw_text: str
    cleaned_text: str
    sections: Dict[str, str]
    word_count: int
    char_count: int
    metadata: Dict[str, any]


@dataclass  
class JobData:
    """Structured job posting data"""
    raw_text: str
    cleaned_text: str
    title: str
    requirements: List[str]
    nice_to_have: List[str]


# =============================================================================
# Text Preprocessor
# =============================================================================

class TextPreprocessor:
    """Advanced text preprocessing for resumes and job descriptions"""
    
    # Common resume section headers
    SECTION_PATTERNS = {
        'contact': r'(?i)^(contact|personal\s+info|details)',
        'summary': r'(?i)^(summary|objective|profile|about\s+me)',
        'education': r'(?i)^(education|academic|qualification|degree)',
        'experience': r'(?i)^(experience|employment|work\s+history|professional\s+experience)',
        'skills': r'(?i)^(skills|technical\s+skills|competencies|technologies|expertise)',
        'projects': r'(?i)^(projects|portfolio|work\s+samples|personal\s+projects)',
        'certifications': r'(?i)^(certifications?|licenses?|credentials)',
        'achievements': r'(?i)^(achievements?|accomplishments?|awards?)',
        'publications': r'(?i)^(publications?|papers?|research)',
        'languages': r'(?i)^(languages?|linguistic)'
    }
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        
        # Remove URLs
        text = re.sub(r'http[s]?://\S+', ' ', text)
        
        # Remove email addresses (but keep indicator)
        text = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', '[EMAIL]', text)
        
        # Remove phone numbers (but keep indicator)
        text = re.sub(r'[\+]?[(]?[0-9]{1,3}[)]?[-\s\.]?[(]?[0-9]{1,3}[)]?[-\s\.]?[0-9]{3,6}', '[PHONE]', text)
        
        # Normalize special characters
        text = re.sub(r'[•●○■□▪▫◦‣⁃]', '•', text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove excessive periods/dots
        text = re.sub(r'\.{2,}', '.', text)
        
        return text.strip()
    
    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """Normalize whitespace while preserving structure"""
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if line:
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    @classmethod
    def extract_sections(cls, text: str) -> Dict[str, str]:
        """Extract resume sections using pattern matching"""
        sections = {key: '' for key in cls.SECTION_PATTERNS.keys()}
        sections['other'] = ''
        
        lines = text.split('\n')
        current_section = 'other'
        section_content = []
        
        for line in lines:
            line_stripped = line.strip()
            
            # Check if this line is a section header
            is_header = False
            for section_name, pattern in cls.SECTION_PATTERNS.items():
                if re.match(pattern, line_stripped):
                    # Save previous section
                    if section_content:
                        sections[current_section] += '\n'.join(section_content) + '\n'
                    
                    current_section = section_name
                    section_content = []
                    is_header = True
                    break
            
            if not is_header and line_stripped:
                section_content.append(line_stripped)
        
        # Save last section
        if section_content:
            sections[current_section] += '\n'.join(section_content)
        
        # Clean up sections
        return {k: v.strip() for k, v in sections.items() if v.strip()}
    
    @staticmethod
    def extract_bullet_points(text: str) -> List[str]:
        """Extract bullet points from text"""
        # Split by common bullet patterns
        bullets = re.split(r'[•●○■□▪▫◦‣⁃\-\*]|\d+[.)]\s', text)
        return [b.strip() for b in bullets if b.strip() and len(b.strip()) > 10]


# =============================================================================
# Resume Loaders
# =============================================================================

class ResumeFileLoader:
    """Load resumes from various file formats"""
    
    @staticmethod
    def load_pdf(filepath: str) -> str:
        """Extract text from PDF"""
        text_parts = []
        
        try:
            with pdfplumber.open(filepath) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
        except Exception as e:
            raise Exception(f"Error reading PDF {filepath}: {e}")
        
        return '\n'.join(text_parts)
    
    @staticmethod
    def load_docx(filepath: str) -> str:
        """Extract text from DOCX"""
        try:
            doc = Document(filepath)
            paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
            return '\n'.join(paragraphs)
        except Exception as e:
            raise Exception(f"Error reading DOCX {filepath}: {e}")
    
    @staticmethod
    def load_txt(filepath: str) -> str:
        """Load plain text file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            with open(filepath, 'r', encoding='latin-1') as f:
                return f.read()
    
    @classmethod
    def load(cls, filepath: str) -> str:
        """Auto-detect format and load"""
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        ext = filepath.suffix.lower()
        
        loaders = {
            '.pdf': cls.load_pdf,
            '.docx': cls.load_docx,
            '.doc': cls.load_docx,
            '.txt': cls.load_txt,
            '.text': cls.load_txt
        }
        
        loader = loaders.get(ext)
        if loader is None:
            raise ValueError(f"Unsupported file format: {ext}")
        
        return loader(str(filepath))


# =============================================================================
# Kaggle Dataset Loader
# =============================================================================

class KaggleResumeLoader:
    """Load and process Kaggle Resume Dataset"""
    
    def __init__(self, data_dir: str = "data/kaggle_resumes"):
        self.data_dir = Path(data_dir)
        self.preprocessor = TextPreprocessor()
    
    def load_dataset(
        self,
        filename: str = "UpdatedResumeDataSet.csv",
        clean: bool = True
    ) -> pd.DataFrame:
        """Load the Kaggle resume dataset"""
        csv_path = self.data_dir / filename
        
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {csv_path}\n"
                f"Download from: https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset"
            )
        
        df = pd.read_csv(csv_path)
        
        if clean:
            df['Resume_Cleaned'] = df['Resume'].apply(self.preprocessor.clean_text)
        
        return df
    
    def get_categories(self, df: pd.DataFrame) -> List[str]:
        """Get all job categories"""
        return sorted(df['Category'].unique().tolist())
    
    def get_category_stats(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """Get statistics per category"""
        stats = {}
        
        for category in df['Category'].unique():
            cat_df = df[df['Category'] == category]
            resumes = cat_df['Resume'].tolist()
            
            word_counts = [len(r.split()) for r in resumes]
            
            stats[category] = {
                'count': len(cat_df),
                'avg_word_count': sum(word_counts) / len(word_counts),
                'min_word_count': min(word_counts),
                'max_word_count': max(word_counts)
            }
        
        return stats
    
    def sample_by_category(
        self,
        df: pd.DataFrame,
        n_per_category: int = 10,
        random_state: int = 42
    ) -> pd.DataFrame:
        """Sample resumes stratified by category"""
        sampled = df.groupby('Category').apply(
            lambda x: x.sample(n=min(len(x), n_per_category), random_state=random_state)
        )
        return sampled.reset_index(drop=True)


# =============================================================================
# Job Description Parser
# =============================================================================

class JobDescriptionParser:
    """Parse and structure job descriptions"""
    
    @staticmethod
    def parse(job_text: str) -> JobData:
        """Parse a job description into structured format"""
        
        cleaned = TextPreprocessor.clean_text(job_text)
        
        # Extract title (usually first line or after "Position:")
        title_match = re.search(r'(?:position|title|role):\s*(.+?)(?:\n|$)', job_text, re.I)
        if title_match:
            title = title_match.group(1).strip()
        else:
            # Use first line as title
            title = job_text.strip().split('\n')[0].strip()
        
        # Extract requirements
        requirements = []
        req_section = re.search(
            r'(?:requirements?|qualifications?|must\s+have)[\s:]*(.+?)(?:nice\s+to\s+have|preferred|bonus|$)',
            job_text, re.I | re.S
        )
        if req_section:
            req_text = req_section.group(1)
            requirements = TextPreprocessor.extract_bullet_points(req_text)
        
        # Extract nice-to-have
        nice_to_have = []
        nice_section = re.search(
            r'(?:nice\s+to\s+have|preferred|bonus|plus)[\s:]*(.+?)$',
            job_text, re.I | re.S
        )
        if nice_section:
            nice_text = nice_section.group(1)
            nice_to_have = TextPreprocessor.extract_bullet_points(nice_text)
        
        return JobData(
            raw_text=job_text,
            cleaned_text=cleaned,
            title=title,
            requirements=requirements,
            nice_to_have=nice_to_have
        )


# =============================================================================
# Complete Data Pipeline
# =============================================================================

class DataPipeline:
    """Complete data loading and preprocessing pipeline"""
    
    def __init__(self, data_dir: str = "data/kaggle_resumes"):
        self.kaggle_loader = KaggleResumeLoader(data_dir)
        self.file_loader = ResumeFileLoader()
        self.preprocessor = TextPreprocessor()
        self.job_parser = JobDescriptionParser()
    
    def load_kaggle_dataset(self, n_samples: Optional[int] = None) -> Tuple[pd.DataFrame, List[str]]:
        """Load Kaggle dataset"""
        df = self.kaggle_loader.load_dataset()
        categories = self.kaggle_loader.get_categories(df)
        
        if n_samples:
            samples_per_cat = max(1, n_samples // len(categories))
            df = self.kaggle_loader.sample_by_category(df, n_per_category=samples_per_cat)
        
        return df, categories
    
    def process_resume_file(self, filepath: str) -> ResumeData:
        """Load and process a resume file"""
        raw_text = self.file_loader.load(filepath)
        cleaned_text = self.preprocessor.clean_text(raw_text)
        sections = self.preprocessor.extract_sections(raw_text)
        
        return ResumeData(
            raw_text=raw_text,
            cleaned_text=cleaned_text,
            sections=sections,
            word_count=len(cleaned_text.split()),
            char_count=len(cleaned_text),
            metadata={'filepath': filepath}
        )
    
    def process_resume_text(self, text: str) -> ResumeData:
        """Process resume text directly"""
        cleaned_text = self.preprocessor.clean_text(text)
        sections = self.preprocessor.extract_sections(text)
        
        return ResumeData(
            raw_text=text,
            cleaned_text=cleaned_text,
            sections=sections,
            word_count=len(cleaned_text.split()),
            char_count=len(cleaned_text),
            metadata={}
        )
    
    def process_job_description(self, job_text: str) -> JobData:
        """Process a job description"""
        return self.job_parser.parse(job_text)


# =============================================================================
# Backward Compatibility
# =============================================================================

# Keep old class names working
ResumePreprocessor = TextPreprocessor


if __name__ == "__main__":
    # Test the pipeline
    pipeline = DataPipeline()
    
    # Test resume text processing
    test_resume = """
    JOHN DOE
    john.doe@email.com | (555) 123-4567
    
    SUMMARY
    Data Analyst with 3 years of experience in Python and SQL.
    
    EXPERIENCE
    Data Analyst, ABC Corp (2021-Present)
    • Built dashboards using Tableau and Power BI
    • Analyzed customer data using Python and SQL
    
    EDUCATION
    B.S. Computer Science, XYZ University, 2021
    
    SKILLS
    Python, SQL, Tableau, Power BI, Excel, Git
    """
    
    resume_data = pipeline.process_resume_text(test_resume)
    print("Resume Sections Found:")
    for section, content in resume_data.sections.items():
        print(f"  {section}: {len(content)} chars")
    
    # Test job description parsing
    test_job = """
    Data Analyst Position
    
    Requirements:
    - 3+ years experience with Python and SQL
    - Proficiency in Tableau or Power BI
    - Strong analytical skills
    
    Nice to Have:
    - Machine Learning experience
    - AWS certification
    """
    
    job_data = pipeline.process_job_description(test_job)
    print(f"\nJob Title: {job_data.title}")
    print(f"Requirements: {len(job_data.requirements)} items")
    print(f"Nice to Have: {len(job_data.nice_to_have)} items")
