
import re
import pandas as pd
from pathlib import Path
import pdfplumber
from docx import Document


class KaggleResumeLoader:
    """Load and process Kaggle Resume Dataset"""

    def __init__(self, data_dir="data/kaggle_resumes"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def load_csv(self, filename="UpdatedResumeDataSet.csv"):
        """Load the main CSV file from Kaggle dataset"""
        csv_path = self.data_dir / filename

        if not csv_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at: {csv_path}\n"
                f"Download from: https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset"
            )

        df = pd.read_csv(csv_path)
        return df

    def get_available_categories(self, df):
        """Get all job categories in dataset"""
        return df['Category'].unique().tolist() if df is not None else []


class ResumePreprocessor:
    """Handles text preprocessing and section extraction"""

    @staticmethod
    def clean_text(text):
        """Clean and normalize text"""
        if not text or pd.isna(text):
            return ""

        # Remove special characters but keep periods
        text = re.sub(r'[^\w\s\.]', ' ', text)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove extra periods
        text = re.sub(r'\.+', '.', text)
        return text.strip()

    @staticmethod
    def extract_sections(text):
        """Extract resume sections (education, experience, skills, projects)"""
        sections = {
            'education': '',
            'experience': '',
            'skills': '',
            'projects': ''
        }

        if not text:
            return sections

        text_lower = text.lower()

        # Define section markers
        patterns = {
            'education': r'(?i)(education|academic|qualification)',
            'experience': r'(?i)(experience|employment|work history)',
            'skills': r'(?i)(skills|technical skills|competencies)',
            'projects': r'(?i)(projects|portfolio)'
        }

        for section, pattern in patterns.items():
            match = re.search(pattern, text_lower)
            if match:
                start = match.start()
                # Find next section or end of text
                next_starts = [m.start() for m in re.finditer(
                    r'(?i)^[A-Z\s]+$', text[start+20:], re.MULTILINE)]
                end = next_starts[0] + start + 20 if next_starts else len(text)
                sections[section] = text[start:end]

        return sections

    def preprocess_resume(self, resume_text):
        """Complete preprocessing pipeline for a resume"""
        cleaned = self.clean_text(resume_text)
        sections = self.extract_sections(cleaned)

        return {
            'raw_text': resume_text,
            'cleaned_text': cleaned,
            'sections': sections,
            'word_count': len(cleaned.split()),
            'char_count': len(cleaned)
        }


class ResumeFileLoader:
    """Load resumes from PDF and DOCX files"""

    @staticmethod
    def load_pdf(filepath):
        """Extract text from PDF resume"""
        try:
            with pdfplumber.open(filepath) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            raise Exception(f"Error reading PDF {filepath}: {e}")

    @staticmethod
    def load_docx(filepath):
        """Extract text from DOCX resume"""
        try:
            doc = Document(filepath)
            text = "\n".join([para.text for para in doc.paragraphs])
            return text
        except Exception as e:
            raise Exception(f"Error reading DOCX {filepath}: {e}")

    @staticmethod
    def load_txt(filepath):
        """Load plain text resume"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            raise Exception(f"Error reading TXT {filepath}: {e}")

    @classmethod
    def load_resume(cls, filepath):
        """Auto-detect format and load resume"""
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        ext = filepath.suffix.lower()

        if ext == '.pdf':
            return cls.load_pdf(filepath)
        elif ext == '.docx':
            return cls.load_docx(filepath)
        elif ext == '.txt':
            return cls.load_txt(filepath)
        else:
            raise ValueError(f"Unsupported file format: {ext}")


class DataPipeline:
    """Complete data loading and preprocessing pipeline"""

    def __init__(self):
        self.kaggle_loader = KaggleResumeLoader()
        self.file_loader = ResumeFileLoader()
        self.preprocessor = ResumePreprocessor()

    def load_kaggle_dataset(self, n_samples=50):
        """Load Kaggle dataset for batch processing"""
        try:
            df = self.kaggle_loader.load_csv()
            categories = self.kaggle_loader.get_available_categories(df)
            return df, categories
        except FileNotFoundError as e:
            raise e

    def preprocess_resume_file(self, filepath):
        """Load and preprocess a resume file"""
        resume_text = self.file_loader.load_resume(filepath)
        preprocessed = self.preprocessor.preprocess_resume(resume_text)
        return preprocessed
