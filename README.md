# Resume-Job Matcher: NLP-Powered Career Matching System


## IE 7500 Applied NLP - Final Project
**Author:** Noel John  
**Course:** IE 7500 Applied NLP, Northeastern University  
**Semester:** Fall 2025

---

## 📋 Project Overview

An end-to-end NLP system that matches resumes to job descriptions using Named Entity Recognition (NER), TF-IDF similarity, and BERT embeddings. The system extracts skills and entities from resumes, compares them against job requirements using semantic understanding, and provides actionable feedback, including personalized cover letter snippets.

### Key Features
- **Multi-format Support:** PDF, DOCX, TXT resume parsing
- **Named Entity Recognition:** Custom NER for resume-specific entities (skills, degrees, certifications)
- **Dual Similarity Methods:** TF-IDF baseline + BERT embeddings for semantic matching
- **Skill Taxonomy:** 413 skills across 17 categories
- **Actionable Feedback:** Match scores, recommendations, cover letter generation
- **Interactive Web App:** Streamlit interface for easy use

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/resume-job-matcher.git
cd resume-job-matcher
```

### 2. Create Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 4. Verify Installation
```bash
python test_quick.py
```

### 5. Run the Web Application
```bash
streamlit run app_v2.py
```
Open http://localhost:8501 in your browser.

---

## 📁 Project Structure

```
resume-job-matcher/
│
├── pipeline_v2.py          # Core NLP pipeline (NER + TF-IDF + BERT)
├── ner_extractor.py        # Named Entity Recognition module
├── data_loader_v2.py       # Data loading and preprocessing
├── app_v2.py               # Streamlit web application
│
├── test_suite.py           # Comprehensive test suite
├── ner_evaluation.py       # NER-specific evaluation
├── kaggle_evaluation.py    # Real-world evaluation on Kaggle data
│
├── requirements.txt        # Python dependencies
├── README.md               # This file
│
└── evaluation/             # Evaluation results
    └── evaluator.py        # Evaluation utilities
```

---

## 🔧 NLP Techniques Implemented

### 1. Named Entity Recognition (NER)
- **spaCy NER:** Pre-trained `en_core_web_sm` model for ORG, GPE, DATE entities
- **Custom Pattern Matching:** Regex patterns for resume-specific entities
- **Entity Types:** Skills, Degrees, Institutions, Job Titles, Companies, Certifications

```python
from ner_extractor import ResumeNERExtractor

extractor = ResumeNERExtractor()
entities = extractor.extract(resume_text)
print(entities.get_skills_flat())  # ['python', 'sql', 'machine learning']
```

### 2. TF-IDF Vectorization (Baseline)
- **N-gram Range:** (1, 3) for capturing phrases
- **Max Features:** 500
- **Similarity:** Cosine similarity between resume and job vectors

```python
from pipeline_v2 import EnhancedSemanticMatcher

matcher = EnhancedSemanticMatcher(use_bert=False)
similarity = matcher.compute_tfidf_similarity(resume_text, job_text)
```

### 3. BERT Embeddings (Transformer-based Semantic Matching)
- **Model:** `all-MiniLM-L6-v2` from Sentence Transformers
- **Embedding Size:** 384 dimensions
- **Advantage:** Captures semantic meaning beyond exact word matches
- **Example:** "ML" matches "machine learning", "data pipelines" matches "ETL"

```python
from pipeline_v2 import EnhancedSemanticMatcher

matcher = EnhancedSemanticMatcher(use_bert=True)
bert_similarity = matcher.compute_bert_similarity(resume_text, job_text)
```

#### Why BERT Improves Matching:
| Scenario | TF-IDF | BERT |
|----------|--------|------|
| "ML" vs "machine learning" | ❌ No match (different tokens) | ✅ Match (same meaning) |
| "data pipelines" vs "ETL" | ❌ No match | ✅ Match (related concepts) |
| "Python developer" vs "Python engineer" | ⚠️ Partial | ✅ High similarity |

---

## 📊 Evaluation Results

### Synthetic Test Data (test_suite.py)
| Metric | Value |
|--------|-------|
| Skill Extraction F1 | **0.84** |
| Best Match Accuracy | **67%** |
| Processing Speed | **<50ms** (with BERT) |

### NER Evaluation (ner_evaluation.py)
| Entity Type | Precision | Recall | F1 |
|-------------|-----------|--------|-----|
| Skills | 0.81 | 0.96 | **0.87** |
| Job Titles | 0.79 | 0.67 | **0.68** |
| Institutions | 0.69 | 0.50 | **0.55** |
| Overall | - | - | **0.50** |

### Semantic Similarity Comparison
| Method | Avg Similarity | Speed |
|--------|----------------|-------|
| TF-IDF (Baseline) | 15-25% | ~5ms |
| BERT Embeddings | 40-60% | ~50ms |

*BERT provides higher similarity scores because it captures semantic meaning, not just word overlap.*

### Real-World Evaluation (Kaggle Dataset)
- **Dataset:** 962 resumes, 24 job categories
- **Avg Skills Extracted:** 8-12 per resume
- **Processing Speed:** <100ms per resume (with BERT)

---

## 💻 Usage Examples

### Command Line
```python
from pipeline_v2 import ResumeJobMatcherV2

# Initialize matcher with BERT enabled
matcher = ResumeJobMatcherV2(use_bert=True)

# Analyze resume against job
resume = "Data Analyst with Python, SQL, Tableau experience..."
job = "Requirements: Python, SQL, Power BI, AWS..."

results = matcher.get_full_analysis(resume, job)

print(f"Match Score: {results['match_result']['overall_score']:.1f}%")
print(f"Semantic (BERT): {results['match_result']['semantic_score_bert']:.1f}%")
print(f"Matched Skills: {results['match_result']['matched_skills']}")
print(f"Missing Skills: {results['match_result']['missing_skills']}")
print(f"Cover Letter: {results['cover_letter_snippet']}")
```

### Web Application
```bash
streamlit run app_v2.py
```
1. Upload your resume (PDF/DOCX/TXT)
2. Paste the job description
3. Click "Analyze Match"
4. View results: score, BERT similarity, skills gap, recommendations, cover letter

---

## 🧪 Running Tests

### Full Test Suite
```bash
python test_suite.py
```

### NER Evaluation
```bash
python ner_evaluation.py
```

### Real-World Evaluation (requires Kaggle dataset)
```bash
python kaggle_evaluation.py "path/to/UpdatedResumeDataSet.csv"
```

---

## 📈 Reproducing Results

### Random Seeds
- All evaluations use `random_state=42` for reproducibility
- spaCy model: `en_core_web_sm` (deterministic)
- BERT model: `all-MiniLM-L6-v2` (deterministic inference)

### Expected Output (test_suite.py)
```
OVERALL SKILL EXTRACTION:
  Avg Precision: 0.729
  Avg Recall:    1.000
  Avg F1 Score:  0.843

Best Match Accuracy: 67%
Avg Processing Time: ~50ms (with BERT)
```

### Computational Requirements
- **RAM:** 4GB minimum (8GB recommended for BERT)
- **CPU:** Any modern processor
- **GPU:** Not required (BERT runs on CPU)
- **Disk:** ~1GB for dependencies (including BERT model)
- **Time:** <2 minutes for full test suite

---

## 🛠️ Dependencies

```
spacy>=3.5.0
scikit-learn>=1.2.0
pandas>=1.5.0
numpy>=1.21.0
pdfplumber>=0.9.0
python-docx>=0.8.11
streamlit>=1.28.0
sentence-transformers>=2.2.0
torch>=2.0.0
```

Install all dependencies:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

---

## 📚 References

1. Honnibal, M., & Montani, I. (2017). spaCy 2: Natural language understanding with Bloom embeddings, convolutional neural networks and incremental parsing.
2. Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. EMNLP 2019.
3. Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. NAACL.
4. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. JMLR 12.
5. Bird, S., Klein, E., & Loper, E. (2009). Natural Language Processing with Python. O'Reilly Media.
6. Kaggle Resume Dataset: https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset
7. Vaswani, A., et al. (2017). Attention Is All You Need. NeurIPS.
8. Mikolov, T., et al. (2013). Distributed Representations of Words and Phrases and their Compositionality. NeurIPS.

---

## 👤 Author

**Noel John**  
MSc Data Analytics Engineering  
Northeastern University Vancouver  

---

## 📄 License

This project is for academic purposes (IE 7500 Applied NLP, Northeastern University).
