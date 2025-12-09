"""
Test Suite for Resume Job Matcher
=================================
Comprehensive testing with real metrics for the final report.

Author: Noel John
Course: IE 7500 Applied NLP, Northeastern University
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from sklearn.metrics import precision_score, recall_score, f1_score

from pipeline_v2 import ResumeJobMatcherV2, EnhancedEntityExtractor, SKILL_TAXONOMY
from data_loader_v2 import DataPipeline, ResumeFileLoader


# =============================================================================
# Test Cases
# =============================================================================

# Sample resumes for testing
TEST_RESUMES = {
    'data_analyst': """
        JANE SMITH
        Data Analyst | jane.smith@email.com
        
        SUMMARY
        Results-driven Data Analyst with 4 years of experience in Python, SQL, 
        and business intelligence tools. Expert in transforming complex data into 
        actionable insights.
        
        EXPERIENCE
        
        Senior Data Analyst, TechCorp Inc. (2021 - Present)
        • Built automated ETL pipelines using Python and Airflow
        • Created executive dashboards in Tableau and Power BI
        • Reduced reporting time by 60% through process automation
        • Performed A/B testing for product optimization
        
        Data Analyst, DataCo (2019 - 2021)
        • Analyzed customer behavior using SQL and Python
        • Developed predictive models with scikit-learn
        • Maintained PostgreSQL and MySQL databases
        
        EDUCATION
        M.S. Data Science, State University, 2019
        B.S. Statistics, State University, 2017
        
        SKILLS
        Python, SQL, Tableau, Power BI, PostgreSQL, MySQL, 
        Scikit-learn, Pandas, NumPy, Git, Excel, A/B Testing,
        Statistical Analysis, Data Visualization
    """,

    'software_engineer': """
        ALEX JOHNSON
        Software Engineer | alex.j@email.com
        
        EXPERIENCE
        
        Software Engineer, BigTech Corp (2020 - Present)
        • Developed microservices using Python and FastAPI
        • Built CI/CD pipelines with Jenkins and Docker
        • Deployed applications on AWS (EC2, S3, Lambda)
        • Implemented RESTful APIs consumed by mobile apps
        
        Junior Developer, StartupXYZ (2018 - 2020)
        • Built web applications using React and Node.js
        • Managed databases with PostgreSQL and MongoDB
        • Used Git for version control
        
        EDUCATION
        B.S. Computer Science, Tech University, 2018
        
        SKILLS
        Python, JavaScript, React, Node.js, FastAPI, Docker,
        AWS, PostgreSQL, MongoDB, Git, Jenkins, REST API,
        Microservices, Linux
    """,

    'ml_engineer': """
        SARAH CHEN
        Machine Learning Engineer | sarah.c@email.com
        
        EXPERIENCE
        
        ML Engineer, AI Innovations (2021 - Present)
        • Developed NLP models using PyTorch and Transformers
        • Built computer vision pipelines with TensorFlow
        • Deployed models on AWS SageMaker
        • Implemented MLOps practices with MLflow and Kubeflow
        
        Data Scientist, Analytics Inc (2019 - 2021)
        • Created predictive models using scikit-learn and XGBoost
        • Built data pipelines with Apache Spark
        • Performed deep learning experiments with Keras
        
        EDUCATION
        Ph.D. Machine Learning, Research University, 2019
        
        SKILLS
        Python, PyTorch, TensorFlow, Keras, Scikit-learn,
        NLP, Computer Vision, Deep Learning, Machine Learning,
        AWS, SageMaker, MLflow, Spark, SQL, Docker
    """
}

# Sample job descriptions
TEST_JOBS = {
    'bi_analyst': """
        Business Intelligence Analyst
        
        REQUIREMENTS:
        - Strong proficiency in SQL and Python
        - Experience with Power BI and Tableau
        - Knowledge of data analysis and ETL
        - Experience with PostgreSQL
        - Version control with Git
        - Strong communication skills
        
        NICE TO HAVE:
        - AWS or Azure cloud experience
        - Machine Learning knowledge
        - Docker containerization
        - Streamlit dashboards
    """,

    'data_engineer': """
        Data Engineer
        
        REQUIREMENTS:
        - Expert Python programming
        - Strong SQL and database design skills
        - Experience with Apache Spark and Hadoop
        - Knowledge of Airflow or similar orchestration
        - AWS or GCP cloud platforms
        - Docker and Kubernetes
        - Git version control
        
        NICE TO HAVE:
        - Kafka streaming experience
        - dbt for data transformation
        - Terraform for infrastructure
    """,

    'ml_scientist': """
        Machine Learning Scientist
        
        REQUIREMENTS:
        - Advanced Python programming
        - Deep learning frameworks (PyTorch, TensorFlow)
        - Experience with NLP or Computer Vision
        - Strong statistical analysis skills
        - AWS SageMaker or similar platforms
        - MLflow or experiment tracking
        
        NICE TO HAVE:
        - Research publications
        - Reinforcement learning
        - LLM fine-tuning experience
    """
}


# =============================================================================
# Test Runner
# =============================================================================

# Check if BERT is available
try:
    from sentence_transformers import SentenceTransformer
    BERT_AVAILABLE = True
    print("✓ BERT (sentence-transformers) is available")
except ImportError:
    BERT_AVAILABLE = False
    print("⚠ BERT not available, using TF-IDF only")


class TestRunner:
    """Run comprehensive tests on the Resume Job Matcher"""

    def __init__(self):
        self.matcher = ResumeJobMatcherV2(use_bert=BERT_AVAILABLE)
        self.results = []
        if BERT_AVAILABLE:
            print("✓ Matcher initialized with BERT embeddings")
        else:
            print("⚠ Matcher initialized with TF-IDF only")

    def test_bert_vs_tfidf(self) -> Dict:
        """Compare BERT vs TF-IDF semantic similarity"""
        print("\n" + "="*60)
        print("TEST 0: BERT vs TF-IDF Comparison")
        print("="*60)

        if not BERT_AVAILABLE:
            print("  BERT not available, skipping comparison...")
            return {'skipped': True}

        # Test cases where BERT should outperform TF-IDF
        test_pairs = [
            {
                'name': 'Abbreviation matching',
                'text1': 'Experience with machine learning and artificial intelligence',
                'text2': 'Looking for ML and AI skills',
                'expected': 'BERT should match ML=machine learning, AI=artificial intelligence'
            },
            {
                'name': 'Synonym matching',
                'text1': 'Built data pipelines and ETL workflows',
                'text2': 'Experience with data engineering and transformation',
                'expected': 'BERT should match pipelines≈engineering, ETL≈transformation'
            },
            {
                'name': 'Role similarity',
                'text1': 'Python developer with backend experience',
                'text2': 'Python engineer for server-side development',
                'expected': 'BERT should match developer≈engineer, backend≈server-side'
            }
        ]

        from pipeline_v2 import EnhancedSemanticMatcher

        matcher_bert = EnhancedSemanticMatcher(use_bert=True)
        matcher_tfidf = EnhancedSemanticMatcher(use_bert=False)

        results = []
        print("\n  {:30} {:>12} {:>12} {:>10}".format(
            'Test Case', 'TF-IDF', 'BERT', 'Winner'))
        print("  " + "-" * 66)

        for test in test_pairs:
            tfidf_score = matcher_tfidf.compute_tfidf_similarity(
                test['text1'], test['text2'])
            bert_score = matcher_bert.compute_bert_similarity(
                test['text1'], test['text2'])

            winner = "BERT ✓" if bert_score > tfidf_score else "TF-IDF"

            print("  {:30} {:>10.1f}% {:>10.1f}% {:>10}".format(
                test['name'], tfidf_score, bert_score, winner
            ))

            results.append({
                'name': test['name'],
                'tfidf': tfidf_score,
                'bert': bert_score,
                'bert_wins': bert_score > tfidf_score
            })

        bert_wins = sum(1 for r in results if r['bert_wins'])
        print("  " + "-" * 66)
        print(f"  BERT wins {bert_wins}/{len(results)} comparisons")

        return {
            'comparisons': results,
            'bert_win_rate': bert_wins / len(results)
        }

    def test_skill_extraction(self) -> Dict:
        """Test skill extraction accuracy"""
        print("\n" + "="*60)
        print("TEST 1: Skill Extraction Accuracy")
        print("="*60)

        # Ground truth for test resumes
        ground_truth = {
            'data_analyst': [
                'python', 'sql', 'tableau', 'power bi', 'postgresql', 'mysql',
                'scikit-learn', 'git', 'excel', 'a/b testing',
                'statistical analysis', 'data visualization'
            ],
            'software_engineer': [
                'python', 'javascript', 'react', 'node.js', 'fastapi', 'docker',
                'aws', 'postgresql', 'mongodb', 'git', 'jenkins', 'rest api',
                'microservices', 'linux'
            ],
            'ml_engineer': [
                'python', 'pytorch', 'tensorflow', 'keras', 'scikit-learn',
                'nlp', 'computer vision', 'deep learning', 'machine learning',
                'aws', 'mlflow', 'spark', 'sql', 'docker'
            ]
        }

        results = {}

        for resume_name, resume_text in TEST_RESUMES.items():
            extracted = self.matcher.entity_extractor.extract_skills_flat(
                resume_text)
            expected = ground_truth.get(resume_name, [])

            # Calculate metrics
            extracted_set = set(s.lower() for s in extracted)
            expected_set = set(s.lower() for s in expected)

            matched = extracted_set & expected_set
            precision = len(matched) / \
                len(extracted_set) if extracted_set else 0
            recall = len(matched) / len(expected_set) if expected_set else 0
            f1 = 2 * precision * recall / \
                (precision + recall) if (precision + recall) > 0 else 0

            results[resume_name] = {
                'extracted': list(extracted_set),
                'expected': list(expected_set),
                'matched': list(matched),
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }

            print(f"\n{resume_name.upper()}:")
            print(f"  Extracted: {len(extracted_set)} skills")
            print(f"  Expected:  {len(expected_set)} skills")
            print(f"  Matched:   {len(matched)} skills")
            print(f"  Precision: {precision:.3f}")
            print(f"  Recall:    {recall:.3f}")
            print(f"  F1 Score:  {f1:.3f}")

        # Average metrics
        avg_precision = sum(r['precision']
                            for r in results.values()) / len(results)
        avg_recall = sum(r['recall'] for r in results.values()) / len(results)
        avg_f1 = sum(r['f1_score'] for r in results.values()) / len(results)

        print(f"\nOVERALL SKILL EXTRACTION:")
        print(f"  Avg Precision: {avg_precision:.3f}")
        print(f"  Avg Recall:    {avg_recall:.3f}")
        print(f"  Avg F1 Score:  {avg_f1:.3f}")

        return {
            'per_resume': results,
            'avg_precision': avg_precision,
            'avg_recall': avg_recall,
            'avg_f1': avg_f1
        }

    def test_matching_accuracy(self) -> Dict:
        """Test resume-job matching"""
        print("\n" + "="*60)
        print("TEST 2: Resume-Job Matching")
        print("="*60)

        # Test all combinations
        results = {}

        for resume_name, resume_text in TEST_RESUMES.items():
            results[resume_name] = {}

            for job_name, job_text in TEST_JOBS.items():
                analysis = self.matcher.get_full_analysis(
                    resume_text, job_text)

                results[resume_name][job_name] = {
                    'overall_score': analysis['match_result']['overall_score'],
                    'skill_f1': analysis['match_result']['skill_f1'],
                    'semantic_tfidf': analysis['match_result']['semantic_score_tfidf'],
                    'fit_level': analysis['match_result']['fit_level'],
                    'matched_skills': analysis['match_result']['matched_skills'],
                    'missing_skills': analysis['match_result']['missing_skills']
                }

        # Print matrix
        print("\nMATCH SCORE MATRIX (%):")
        print(f"{'Resume':<20}", end="")
        for job_name in TEST_JOBS.keys():
            print(f"{job_name:<15}", end="")
        print()
        print("-" * 65)

        for resume_name in TEST_RESUMES.keys():
            print(f"{resume_name:<20}", end="")
            for job_name in TEST_JOBS.keys():
                score = results[resume_name][job_name]['overall_score']
                print(f"{score:<15.1f}", end="")
            print()

        # Expected best matches
        expected_best = {
            'data_analyst': 'bi_analyst',
            'software_engineer': 'data_engineer',
            'ml_engineer': 'ml_scientist'
        }

        # Verify best matches
        print("\nBEST MATCH VERIFICATION:")
        correct = 0
        for resume_name, expected_job in expected_best.items():
            scores = results[resume_name]
            best_match = max(
                scores.keys(), key=lambda j: scores[j]['overall_score'])
            is_correct = best_match == expected_job
            correct += int(is_correct)

            print(
                f"  {resume_name}: Expected={expected_job}, Got={best_match} {'✓' if is_correct else '✗'}")

        accuracy = correct / len(expected_best)
        print(f"\nBest Match Accuracy: {accuracy:.0%}")

        return {
            'matrix': results,
            'best_match_accuracy': accuracy
        }

    def test_processing_speed(self, n_iterations: int = 10) -> Dict:
        """Test processing speed"""
        print("\n" + "="*60)
        print("TEST 3: Processing Speed")
        print("="*60)

        resume_text = TEST_RESUMES['data_analyst']
        job_text = TEST_JOBS['bi_analyst']

        times = []
        for i in range(n_iterations):
            start = time.time()
            _ = self.matcher.get_full_analysis(resume_text, job_text)
            elapsed = time.time() - start
            times.append(elapsed)

        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)

        print(f"  Iterations: {n_iterations}")
        print(f"  Avg Time:   {avg_time*1000:.1f}ms")
        print(f"  Min Time:   {min_time*1000:.1f}ms")
        print(f"  Max Time:   {max_time*1000:.1f}ms")

        return {
            'avg_ms': avg_time * 1000,
            'min_ms': min_time * 1000,
            'max_ms': max_time * 1000,
            'iterations': n_iterations
        }

    def test_entity_extraction(self) -> Dict:
        """Test named entity extraction"""
        print("\n" + "="*60)
        print("TEST 4: Named Entity Extraction")
        print("="*60)

        extractor = EnhancedEntityExtractor()

        resume_text = TEST_RESUMES['data_analyst']
        entities = extractor.extract_entities(resume_text)

        print(f"\nExtracted from data_analyst resume:")
        print(f"  Organizations: {entities['organizations']}")
        print(f"  Locations:     {entities['locations']}")
        print(f"  Dates:         {entities['dates']}")
        print(f"  Education:     {entities['education']}")
        print(f"  Job Titles:    {entities['job_titles']}")

        return {
            'entities': entities
        }

    def test_feedback_generation(self) -> Dict:
        """Test feedback and recommendations"""
        print("\n" + "="*60)
        print("TEST 5: Feedback Generation")
        print("="*60)

        resume_text = TEST_RESUMES['data_analyst']
        job_text = TEST_JOBS['bi_analyst']

        analysis = self.matcher.get_full_analysis(resume_text, job_text)

        print(
            f"\nMatch Score: {analysis['match_result']['overall_score']:.1f}%")
        print(f"Fit Level: {analysis['match_result']['fit_level']}")

        print("\nRecommendations:")
        for i, rec in enumerate(analysis['recommendations'], 1):
            print(f"  {i}. {rec}")

        print("\nResume Suggestions:")
        for suggestion in analysis['resume_suggestions']:
            print(f"  • {suggestion}")

        print("\nCover Letter Snippet Preview:")
        snippet = analysis['cover_letter_snippet']
        print(f"  {snippet[:200]}..." if len(
            snippet) > 200 else f"  {snippet}")

        return {
            'recommendations': analysis['recommendations'],
            'suggestions': analysis['resume_suggestions'],
            'cover_letter': analysis['cover_letter_snippet']
        }

    def run_all_tests(self) -> Dict:
        """Run all tests and generate report"""
        print("\n" + "="*70)
        print("RESUME JOB MATCHER - COMPREHENSIVE TEST SUITE")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70)

        # Run BERT vs TF-IDF comparison first
        bert_results = self.test_bert_vs_tfidf()

        results = {
            'timestamp': datetime.now().isoformat(),
            'bert_available': BERT_AVAILABLE,
            'bert_comparison': bert_results,
            'skill_extraction': self.test_skill_extraction(),
            'matching': self.test_matching_accuracy(),
            'speed': self.test_processing_speed(),
            'entity_extraction': self.test_entity_extraction(),
            'feedback': self.test_feedback_generation()
        }

        # Summary
        print("\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)
        print(
            f"  BERT Enabled:            {'Yes ✓' if BERT_AVAILABLE else 'No (TF-IDF only)'}")
        if BERT_AVAILABLE and 'bert_win_rate' in bert_results:
            print(
                f"  BERT vs TF-IDF Win Rate: {bert_results['bert_win_rate']:.0%}")
        print(
            f"  Skill Extraction F1:     {results['skill_extraction']['avg_f1']:.3f}")
        print(
            f"  Best Match Accuracy:     {results['matching']['best_match_accuracy']:.0%}")
        print(f"  Avg Processing Time:     {results['speed']['avg_ms']:.1f}ms")
        print("="*70)

        return results


# =============================================================================
# Personal Resume Test
# =============================================================================

def test_personal_resume(resume_path: str, job_text: str):
    """Test with your personal resume"""
    print("\n" + "="*70)
    print("PERSONAL RESUME TEST")
    print("="*70)

    try:
        resume_text = ResumeFileLoader.load(resume_path)
        print(f"Loaded: {resume_path}")
        print(f"Word count: {len(resume_text.split())}")

        matcher = ResumeJobMatcherV2(use_bert=False)
        analysis = matcher.get_full_analysis(resume_text, job_text)

        print(f"\nRESULTS:")
        print(
            f"  Overall Match Score: {analysis['match_result']['overall_score']:.1f}%")
        print(
            f"  Fit Level:           {analysis['match_result']['fit_level']}")
        print(
            f"  Semantic Similarity: {analysis['match_result']['semantic_score_tfidf']:.1f}%")
        print(
            f"  Skill Match F1:      {analysis['match_result']['skill_f1']:.3f}")

        print(f"\nSKILLS ANALYSIS:")
        print(
            f"  Skills Extracted:    {len(analysis['resume_data']['skills'])}")
        print(
            f"  Skills Required:     {len(analysis['job_data']['required_skills'])}")
        print(
            f"  Matched Skills:      {len(analysis['match_result']['matched_skills'])}")
        print(
            f"  Missing Skills:      {len(analysis['match_result']['missing_skills'])}")

        print(
            f"\n✓ Matched: {', '.join(sorted(analysis['match_result']['matched_skills']))}")
        print(
            f"✗ Missing: {', '.join(sorted(analysis['match_result']['missing_skills']))}")

        return analysis

    except Exception as e:
        print(f"Error: {e}")
        return None


# =============================================================================
# Main
# =============================================================================

def main():
    """Run tests"""

    runner = TestRunner()
    results = runner.run_all_tests()

    # Save results
    output_path = Path("test_results.json")
    json_results = {
        'timestamp': results['timestamp'],
        'skill_extraction': {
            'avg_precision': results['skill_extraction']['avg_precision'],
            'avg_recall': results['skill_extraction']['avg_recall'],
            'avg_f1': results['skill_extraction']['avg_f1']
        },
        'matching': {
            'best_match_accuracy': results['matching']['best_match_accuracy']
        },
        'speed': results['speed']
    }

    with open(output_path, 'w') as f:
        json.dump(json_results, f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
