# Resume Job Matcher 📄

NLP-powered system that analyzes resume-job fit using Named Entity Recognition and TF-IDF semantic matching.

**Author:** Noel John  
**Course:** IE 7500 Applied NLP, Northeastern University  
**Week 9 Mid-Project Submission**

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 2. Run Streamlit demo
streamlit run app.py
```

---

## 📂 Project Structure

```
NLP/
├── Pipeline.py              # Core NLP pipeline (entity extraction, matching)
├── data_loader.py           # Data loading utilities
├── app.py                   # Streamlit web interface
├── requirements.txt         # Dependencies
├── README.md               # This file
```

---

## 🧠 Technical Approach

### NLP Techniques Implemented

**1. Named Entity Recognition (spaCy)**
- Extracts: Skills, Organizations, Dates, Locations
- Uses hierarchical skill taxonomy (200+ terms)
- Categories: Programming, ML Frameworks, Visualization, Cloud, Databases, Tools, Analytics
- **Performance:** F1 Score = 0.77

**2. TF-IDF Semantic Matching**
- Computes cosine similarity between resume and job text
- Analyzes keyword overlap with precision/recall metrics
- Weighted scoring: 40% semantic + 60% skill match
- **Performance:** Processes <2 seconds per pair

---

## 📊 Week 9 Test Results

**Test Configuration:**
- Resume: Personal resume (Noel John, 426 words)
- Job: CAE Business Intelligence Analyst internship
- Date: October 28, 2025

**Results:**
- Overall Match: **43.5%**
- Skills Matched: **5** (Python, SQL, Tableau, Power BI, PostgreSQL)
- Skills Missing: **6** (AWS, Azure, Docker, Git, Machine Learning, Data Analysis)
- Entity Extraction F1: **0.77**
- Semantic Similarity: **14.9%**
- Skill Match F1: **0.625**

---

## 🎯 Key Features

- ✅ Loads resumes from PDF, DOCX, or TXT formats
- ✅ Hierarchical skill taxonomy (200+ technical terms)
- ✅ Dual NLP techniques: spaCy NER + TF-IDF matching
- ✅ Interactive Streamlit interface
- ✅ Real-time processing (<2 seconds)
- ✅ Honest match scoring (not inflated)

---

## 💻 Usage

### Streamlit Interface

```bash
streamlit run app.py
```

- Automatically loads: `Noel John CV Student Analyst.pdf`
- Default job: CAE Business Intelligence Analyst position
- Click "Analyze Match" for instant results
- Shows: Match score, matched skills, missing skills

### Generate Test Metrics

```bash
python test.py
```

Outputs:
- `test_results.json` - Raw metrics data
- `report_text.txt` - Formatted text for report

---

## 📈 Dataset

**Kaggle Resume Dataset:**
- Size: 962 resumes
- Categories: 24 job types
- Source: https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset
- Status: Downloaded and loaded successfully

**Testing Approach:**
- Week 9: Sample testing with personal resume
- Week 11: Full evaluation on 200+ resume-job pairs

---

## 🔧 Dependencies

Core packages:
- spacy (NER)
- scikit-learn (TF-IDF)
- pandas (data loading)
- pdfplumber (PDF processing)
- streamlit (web interface)

Full list in `requirements.txt`

---

## 📝 Code Overview

### Pipeline.py (Main NLP Code)

**EntityExtractor:**
- spaCy NER for organizations, dates, locations
- Skill taxonomy matching (200+ terms)
- Filters false positives from entity detection

**SemanticMatcher:**
- TF-IDF vectorization (1-2 grams, 200 features)
- Cosine similarity computation
- Skill overlap metrics (precision, recall, F1)

**ResumeJobMatcher:**
- End-to-end pipeline orchestration
- Weighted scoring algorithm
- Result formatting

### data_loader.py

**KaggleResumeLoader:**
- Loads CSV dataset
- Returns categories and resume data

**ResumePreprocessor:**
- Text cleaning and normalization
- Section extraction (Education, Experience, Skills, Projects)

**ResumeFileLoader:**
- Supports PDF, DOCX, TXT formats
- Auto-detection based on file extension

---

## 🎓 Week 9 Accomplishments

**Completed:**
- ✅ Data loading pipeline (Kaggle dataset integrated)
- ✅ Text preprocessing (cleaning, section extraction)
- ✅ Two NLP techniques (NER + TF-IDF)
- ✅ Working Streamlit demonstration
- ✅ Evaluation metrics (F1: 0.77)
- ✅ Real testing with personal resume

**In Progress:**
- ⏳ BERT embeddings (Week 10-11)
- ⏳ Template feedback generation (Week 12)
- ⏳ Large-scale evaluation (Week 11-12)
- ⏳ User study (Week 13)

---

## 🔍 Example Output

```
Match Analysis: CAE Business Intelligence Analyst

Overall Match Score: 43.5%
Semantic Similarity: 14.9%
Skill Match F1: 0.625

✓ Matched Skills (5):
  - Python
  - SQL
  - Tableau
  - Power BI
  - PostgreSQL

✗ Missing Skills (6):
  - AWS
  - Azure
  - Docker
  - Git
  - Machine Learning
  - Data Analysis

Recommendation: MODERATE FIT
- Highlight these strengths: Python, SQL, Tableau
- Priority skills to develop: AWS, Docker, Git
```

---

## 📚 References

1. Kaggle Resume Dataset: https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset
2. spaCy NER: https://spacy.io/usage/linguistic-features#named-entities
3. scikit-learn TF-IDF: https://scikit-learn.org/stable/modules/feature_extraction.html

---
