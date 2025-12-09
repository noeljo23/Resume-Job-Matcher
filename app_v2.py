"""
Resume Job Matcher - Enhanced Streamlit Application
===================================================
Full-featured web interface for resume-job matching with:
- PDF/DOCX/TXT resume upload
- Paste any job description
- Detailed match analysis
- Actionable recommendations
- Cover letter snippets

Author: Noel John
Course: IE 7500 Applied NLP, Northeastern University
"""

import streamlit as st
import pdfplumber
from docx import Document
import io
import json
from datetime import datetime

# Import the enhanced pipeline
from pipeline_v2 import ResumeJobMatcherV2, SKILL_TAXONOMY

# =============================================================================
# Page Config
# =============================================================================

st.set_page_config(
    page_title="Resume Job Matcher",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #6b7280;
        margin-bottom: 2rem;
    }
    .skill-tag-matched {
        background-color: #065f46;
        color: #d1fae5;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.875rem;
        margin: 0.25rem;
        display: inline-block;
    }
    .skill-tag-missing {
        background-color: #991b1b;
        color: #fee2e2;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.875rem;
        margin: 0.25rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Helper Functions
# =============================================================================

@st.cache_resource
def load_matcher():
    """Load the matcher (cached)"""
    # Set use_bert=True if sentence-transformers is installed
    try:
        from sentence_transformers import SentenceTransformer
        use_bert = True
    except ImportError:
        use_bert = False
    return ResumeJobMatcherV2(use_bert=use_bert)


def extract_text_from_pdf(file) -> str:
    """Extract text from uploaded PDF"""
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def extract_text_from_docx(file) -> str:
    """Extract text from uploaded DOCX"""
    doc = Document(file)
    return "\n".join([para.text for para in doc.paragraphs])


def extract_text_from_txt(file) -> str:
    """Extract text from uploaded TXT"""
    return file.read().decode('utf-8')


def load_resume(uploaded_file) -> str:
    """Load resume from uploaded file"""
    if uploaded_file is None:
        return ""

    file_type = uploaded_file.name.split('.')[-1].lower()

    if file_type == 'pdf':
        return extract_text_from_pdf(uploaded_file)
    elif file_type == 'docx':
        return extract_text_from_docx(uploaded_file)
    elif file_type == 'txt':
        return extract_text_from_txt(uploaded_file)
    else:
        st.error(f"Unsupported file type: {file_type}")
        return ""


def get_fit_color(fit_level: str) -> str:
    """Get color based on fit level"""
    colors = {
        "Strong": "#10b981",
        "Moderate": "#f59e0b",
        "Weak": "#ef4444"
    }
    return colors.get(fit_level, "#6b7280")


def display_score_gauge(score: float, label: str):
    """Display a score as a progress bar"""
    color = "#10b981" if score >= 70 else "#f59e0b" if score >= 40 else "#ef4444"
    st.markdown(f"**{label}**")
    st.progress(min(score/100, 1.0))
    st.markdown(f"<span style='color: {color}; font-size: 1.5rem; font-weight: bold;'>{score:.1f}%</span>",
                unsafe_allow_html=True)


# =============================================================================
# Main Application
# =============================================================================

def main():
    # Header
    st.markdown('<p class="main-header">📄 Resume Job Matcher</p>',
                unsafe_allow_html=True)
    st.markdown('<p class="sub-header">NLP-powered analysis to help you land your dream job</p>',
                unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")

        st.subheader("About")
        st.info("""
        This tool uses:
        - **spaCy NER** for entity extraction
        - **TF-IDF + BERT** for semantic matching
        - **Custom taxonomy** with 200+ skills
        
        Upload your resume and paste a job description to get:
        - Match score and analysis
        - Skills gap identification
        - Actionable recommendations
        - Cover letter snippets
        """)

        st.divider()

        # Skill categories reference
        with st.expander("📚 Skill Categories"):
            for category, skills in SKILL_TAXONOMY.items():
                st.markdown(f"**{category.replace('_', ' ').title()}**")
                st.caption(", ".join(skills[:10]) + "...")

    # Main content - Two columns
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📤 Upload Resume")
        uploaded_file = st.file_uploader(
            "Upload your resume (PDF, DOCX, or TXT)",
            type=['pdf', 'docx', 'txt'],
            help="Your resume will be analyzed to extract skills and entities"
        )

        if uploaded_file:
            resume_text = load_resume(uploaded_file)
            st.success(
                f"✅ Loaded: {uploaded_file.name} ({len(resume_text.split())} words)")

            with st.expander("Preview Resume Text"):
                st.text(
                    resume_text[:2000] + "..." if len(resume_text) > 2000 else resume_text)
        else:
            resume_text = ""
            st.info("👆 Upload your resume to get started")

    with col2:
        st.subheader("📋 Job Description")
        job_text = st.text_area(
            "Paste the job description here",
            height=300,
            placeholder="""Business Intelligence Analyst

REQUIREMENTS:
- Strong proficiency in SQL and Python
- Experience with Power BI and Tableau
- Knowledge of data analysis and ETL
- Experience with PostgreSQL and Git

NICE TO HAVE:
- AWS, Azure, Docker
- Machine Learning experience
- Streamlit dashboards""",
            help="Paste the full job posting for best results"
        )

        if job_text:
            st.success(
                f"✅ Job description loaded ({len(job_text.split())} words)")

    st.divider()

    # Analyze button
    analyze_col1, analyze_col2, analyze_col3 = st.columns([1, 2, 1])
    with analyze_col2:
        analyze_button = st.button(
            "🔍 Analyze Match",
            type="primary",
            use_container_width=True,
            disabled=not (resume_text and job_text)
        )

    # Results
    if analyze_button and resume_text and job_text:
        with st.spinner("Analyzing your resume against the job description..."):
            matcher = load_matcher()
            results = matcher.get_full_analysis(resume_text, job_text)

        st.divider()

        # Score Overview
        st.subheader("📊 Match Analysis")

        score_col1, score_col2, score_col3, score_col4 = st.columns(4)

        with score_col1:
            fit_color = get_fit_color(results['match_result']['fit_level'])
            st.metric(
                "Overall Match",
                f"{results['match_result']['overall_score']:.0f}%",
                delta=results['match_result']['fit_level'],
            )

        with score_col2:
            st.metric(
                "Skills Matched",
                len(results['match_result']['matched_skills']),
                delta=f"of {len(results['job_data']['required_skills'])} required"
            )

        with score_col3:
            st.metric(
                "Skills Gap",
                len(results['match_result']['missing_skills']),
                delta="to develop" if results['match_result']['missing_skills'] else "None!"
            )

        with score_col4:
            bert_score = results['match_result']['semantic_score_bert']
            tfidf_score = results['match_result']['semantic_score_tfidf']
            if bert_score is not None:
                st.metric("Semantic (BERT)", f"{bert_score:.1f}%")
            else:
                st.metric("Semantic (TF-IDF)", f"{tfidf_score:.1f}%")

        # Detailed metrics
        st.divider()

        detail_col1, detail_col2 = st.columns(2)

        with detail_col1:
            st.subheader("✅ Matched Skills")
            if results['match_result']['matched_skills']:
                for skill in sorted(results['match_result']['matched_skills']):
                    st.markdown(
                        f'<span class="skill-tag-matched">✓ {skill}</span>', unsafe_allow_html=True)
            else:
                st.warning("No direct skill matches found")

            # Extra skills (bonus)
            if results['match_result']['extra_skills']:
                st.markdown("---")
                st.markdown(
                    "**🎁 Bonus Skills (you have, job doesn't require):**")
                for skill in sorted(results['match_result']['extra_skills'])[:10]:
                    st.caption(f"• {skill}")

        with detail_col2:
            st.subheader("❌ Missing Skills")
            if results['match_result']['missing_skills']:
                for skill in sorted(results['match_result']['missing_skills']):
                    st.markdown(
                        f'<span class="skill-tag-missing">✗ {skill}</span>', unsafe_allow_html=True)
            else:
                st.success("🎉 You have all required skills!")

        # Recommendations
        st.divider()
        st.subheader("💡 Recommendations")

        for i, rec in enumerate(results['recommendations'], 1):
            st.info(f"**{i}.** {rec}")

        # Resume suggestions
        if results['resume_suggestions']:
            st.markdown("**📝 Resume Suggestions:**")
            for suggestion in results['resume_suggestions']:
                st.warning(suggestion)

        # Cover Letter Snippet
        st.divider()
        st.subheader("✍️ Cover Letter Snippet")

        if results['cover_letter_snippet']:
            st.success(results['cover_letter_snippet'])
            st.caption(
                "💡 Customize this snippet with specific examples from your experience")

        # Extracted Entities
        st.divider()
        with st.expander("🔎 Extracted Entities (Debug)"):
            ent_col1, ent_col2 = st.columns(2)

            with ent_col1:
                st.markdown("**From Resume:**")
                st.json({
                    'skills': results['resume_data']['skills'],
                    'organizations': results['resume_data']['entities']['organizations'],
                    'locations': results['resume_data']['entities']['locations'],
                    'word_count': results['resume_data']['word_count']
                })

            with ent_col2:
                st.markdown("**From Job:**")
                st.json({
                    'required_skills': results['job_data']['required_skills'],
                    'job_title': results['job_data']['job_title']
                })

        # Export Results
        st.divider()
        export_col1, export_col2, export_col3 = st.columns([1, 1, 1])

        with export_col2:
            export_data = {
                'timestamp': datetime.now().isoformat(),
                'overall_score': results['match_result']['overall_score'],
                'fit_level': results['match_result']['fit_level'],
                'matched_skills': results['match_result']['matched_skills'],
                'missing_skills': results['match_result']['missing_skills'],
                'recommendations': results['recommendations']
            }

            st.download_button(
                "📥 Export Results (JSON)",
                data=json.dumps(export_data, indent=2),
                file_name=f"match_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

    # Footer
    st.divider()
    st.caption("""
    Built with ❤️ by Noel John | IE 7500 Applied NLP | Northeastern University  
    Using: spaCy, scikit-learn, Sentence Transformers, Streamlit
    """)


if __name__ == "__main__":
    main()
