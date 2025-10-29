import streamlit as st
import pdfplumber
from Pipeline import ResumeJobMatcher

st.set_page_config(page_title="Resume Matcher", layout="wide")

RESUME_PATH = r"C:\Users\NJ\Desktop\NLP\Noel John CV Student Analyst.pdf"


# Load PDF
with pdfplumber.open(RESUME_PATH) as pdf:
    resume_text = ""
    for page in pdf.pages:
        resume_text += page.extract_text() + "\n"

# Job description
job_text = """Business Intelligence Analyst

REQUIREMENTS:
- Strong proficiency in sql and python
- Experience with power bi and tableau
- Knowledge of data analysis and etl
- Experience with postgresql and git
- Knowledge of machine learning

NICE TO HAVE:
- aws, azure, docker
- streamlit, excel
- data visualization"""

st.title("📄 Resume Matcher")
st.divider()

if st.button("🔍 Analyze Match", type="primary"):
    matcher = ResumeJobMatcher()

    # Process resume and job
    resume_data = matcher.process_resume(resume_text)
    job_data = matcher.process_job(job_text)

    # Match
    match_results = matcher.match(
        resume_text,
        job_text,
        resume_data['skills'],
        job_data['required_skills']
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Match", f"{match_results['overall_score']:.0f}%")
    col2.metric("Matched", len(match_results['matched_skills']))
    col3.metric("Missing", len(match_results['missing_skills']))

    st.subheader("✅ Matched Skills")
    for skill in sorted(match_results['matched_skills']):
        st.success(f"✓ {skill}")

    st.subheader("❌ Missing Skills")
    for skill in sorted(match_results['missing_skills']):
        st.error(f"✗ {skill}")
