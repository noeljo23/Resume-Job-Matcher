"""
Evaluation Module for Resume Job Matcher
=========================================
Comprehensive evaluation on Kaggle dataset with:
- Cross-validation testing
- Per-category analysis
- Statistical metrics
- Baseline comparisons

Author: Noel John
Course: IE 7500 Applied NLP, Northeastern University
"""

import os
import json
import time
import random
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass, asdict

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    accuracy_score, confusion_matrix, classification_report
)

from pipeline_v2 import ResumeJobMatcherV2, EnhancedEntityExtractor, SKILL_TAXONOMY


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class EvaluationMetrics:
    """Store evaluation metrics"""
    entity_precision: float
    entity_recall: float
    entity_f1: float
    skill_precision: float
    skill_recall: float
    skill_f1: float
    semantic_similarity_avg: float
    processing_time_avg: float
    sample_count: int


@dataclass
class CategoryResults:
    """Results for a specific job category"""
    category: str
    sample_count: int
    avg_skills_extracted: float
    avg_match_score: float
    common_skills: List[str]
    metrics: EvaluationMetrics


# =============================================================================
# Evaluation Pipeline
# =============================================================================

class ResumeMatcherEvaluator:
    """Comprehensive evaluation of the Resume Job Matcher"""
    
    def __init__(self, data_dir: str = "data/kaggle_resumes"):
        self.data_dir = Path(data_dir)
        self.matcher = ResumeJobMatcherV2(use_bert=False)  # TF-IDF for speed
        self.entity_extractor = EnhancedEntityExtractor()
        
        # Results storage
        self.results = []
        self.category_results = {}
    
    def load_kaggle_dataset(self, filename: str = "UpdatedResumeDataSet.csv") -> pd.DataFrame:
        """Load the Kaggle resume dataset"""
        csv_path = self.data_dir / filename
        
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at: {csv_path}\n"
                f"Download from: https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset\n"
                f"Place the CSV file in: {self.data_dir}"
            )
        
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} resumes from {csv_path}")
        print(f"Categories: {df['Category'].nunique()}")
        
        return df
    
    def create_synthetic_job_descriptions(self, categories: List[str]) -> Dict[str, str]:
        """Create synthetic job descriptions for each category"""
        
        # Category-specific job templates
        job_templates = {
            'Data Science': """
                Data Scientist Position
                Requirements: Python, R, SQL, Machine Learning, Deep Learning, 
                TensorFlow, PyTorch, Statistics, Data Analysis, Pandas, NumPy,
                Scikit-learn, Data Visualization, Tableau, A/B Testing
                Nice to have: AWS, Spark, Docker, NLP, Computer Vision
            """,
            'HR': """
                HR Specialist Position
                Requirements: Excel, Communication, Stakeholder Management,
                Project Management, HRIS, Recruitment, Employee Relations
                Nice to have: SAP, Workday, Data Analysis, Reporting
            """,
            'Advocate': """
                Legal Analyst Position
                Requirements: Legal Research, Contract Analysis, Communication,
                Documentation, Microsoft Office, Excel, Attention to Detail
                Nice to have: Database Management, Project Management
            """,
            'Arts': """
                Creative Designer Position
                Requirements: Adobe Creative Suite, Photoshop, Illustrator,
                Design, Typography, Visual Communication, Portfolio
                Nice to have: Video Editing, Motion Graphics, UI/UX
            """,
            'Web Designing': """
                Web Developer Position
                Requirements: HTML, CSS, JavaScript, React, Node.js,
                Frontend Development, Responsive Design, Git
                Nice to have: TypeScript, Vue, Angular, AWS, Docker
            """,
            'Mechanical Engineer': """
                Mechanical Engineer Position
                Requirements: CAD, AutoCAD, SolidWorks, Engineering Design,
                Manufacturing, Project Management, Technical Documentation
                Nice to have: Python, MATLAB, FEA, Simulation
            """,
            'Sales': """
                Sales Representative Position
                Requirements: CRM, Salesforce, Communication, Negotiation,
                Lead Generation, Customer Relations, Excel, Presentations
                Nice to have: Data Analysis, Marketing, Social Media
            """,
            'Health and fitness': """
                Health Program Coordinator Position
                Requirements: Health Education, Program Management, Communication,
                Microsoft Office, Data Analysis, Reporting
                Nice to have: Public Health, Research, Statistics
            """,
            'Civil Engineer': """
                Civil Engineer Position
                Requirements: AutoCAD, Civil 3D, Structural Analysis,
                Project Management, Engineering Design, Technical Writing
                Nice to have: Python, GIS, BIM, Revit
            """,
            'Java Developer': """
                Java Developer Position
                Requirements: Java, Spring Boot, SQL, Git, REST API,
                Microservices, Maven, JUnit, Agile
                Nice to have: AWS, Docker, Kubernetes, CI/CD
            """,
            'Business Analyst': """
                Business Analyst Position
                Requirements: SQL, Excel, Data Analysis, Requirements Gathering,
                Stakeholder Management, Tableau, Power BI, Agile
                Nice to have: Python, Jira, Process Modeling
            """,
            'SAP Developer': """
                SAP Developer Position
                Requirements: SAP, ABAP, SQL, Integration, Technical Documentation
                Nice to have: Python, REST API, Cloud, S/4HANA
            """,
            'Automation Testing': """
                QA Automation Engineer Position
                Requirements: Selenium, Python, Java, Test Automation,
                CI/CD, Git, Agile, SQL
                Nice to have: Jenkins, Docker, API Testing
            """,
            'Electrical Engineering': """
                Electrical Engineer Position
                Requirements: Circuit Design, PCB, MATLAB, AutoCAD,
                Technical Documentation, Project Management
                Nice to have: Python, Embedded Systems, PLC
            """,
            'Operations Manager': """
                Operations Manager Position
                Requirements: Operations Management, Process Improvement,
                Project Management, Excel, Data Analysis, Leadership
                Nice to have: Six Sigma, Lean, ERP, SQL
            """,
            'Python Developer': """
                Python Developer Position
                Requirements: Python, Django, Flask, SQL, Git,
                REST API, Linux, Data Structures
                Nice to have: AWS, Docker, Machine Learning, FastAPI
            """,
            'DevOps Engineer': """
                DevOps Engineer Position
                Requirements: Docker, Kubernetes, AWS, Linux, CI/CD,
                Git, Jenkins, Terraform, Python
                Nice to have: Azure, GCP, Ansible, Monitoring
            """,
            'Network Security Engineer': """
                Network Security Engineer Position
                Requirements: Network Security, Firewall, Linux, Python,
                Security Analysis, Penetration Testing
                Nice to have: AWS, Cloud Security, SIEM
            """,
            'PMO': """
                Project Manager Position
                Requirements: Project Management, Agile, Scrum, Jira,
                Stakeholder Management, Excel, Communication
                Nice to have: PMP, Risk Management, SQL
            """,
            'Database': """
                Database Administrator Position
                Requirements: SQL, MySQL, PostgreSQL, Oracle,
                Database Design, Performance Tuning, Backup
                Nice to have: MongoDB, AWS, Python, Automation
            """,
            'Hadoop': """
                Big Data Engineer Position
                Requirements: Hadoop, Spark, Hive, SQL, Python,
                Data Pipeline, ETL, Linux
                Nice to have: Kafka, AWS, Scala, Machine Learning
            """,
            'ETL Developer': """
                ETL Developer Position
                Requirements: ETL, SQL, Python, Data Warehouse,
                SSIS, Data Integration, Informatica
                Nice to have: AWS, Spark, Airflow
            """,
            'DotNet Developer': """
                .NET Developer Position
                Requirements: C#, .NET, ASP.NET, SQL Server, Git,
                REST API, Entity Framework
                Nice to have: Azure, Docker, React, Microservices
            """,
            'Blockchain': """
                Blockchain Developer Position
                Requirements: Blockchain, Solidity, Python, JavaScript,
                Smart Contracts, Ethereum, Git
                Nice to have: Node.js, React, DeFi
            """,
            'Testing': """
                QA Engineer Position
                Requirements: Software Testing, Test Cases, SQL,
                Bug Tracking, Agile, Documentation
                Nice to have: Automation, Selenium, Python
            """
        }
        
        # Default job for unknown categories
        default_job = """
            Analyst Position
            Requirements: Excel, Communication, Data Analysis,
            Problem Solving, Documentation, Teamwork
            Nice to have: SQL, Python, Tableau
        """
        
        return {cat: job_templates.get(cat, default_job) for cat in categories}
    
    def evaluate_entity_extraction(
        self,
        extracted: List[str],
        ground_truth: List[str]
    ) -> Dict[str, float]:
        """Evaluate entity extraction accuracy"""
        if not ground_truth:
            return {'precision': 1.0, 'recall': 1.0, 'f1_score': 1.0}
        
        all_entities = list(set(extracted + ground_truth))
        y_true = [1 if ent in ground_truth else 0 for ent in all_entities]
        y_pred = [1 if ent in extracted else 0 for ent in all_entities]
        
        return {
            'precision': precision_score(y_true, y_pred, zero_division=1),
            'recall': recall_score(y_true, y_pred, zero_division=1),
            'f1_score': f1_score(y_true, y_pred, zero_division=1)
        }
    
    def evaluate_single_resume(
        self,
        resume_text: str,
        job_text: str,
        category: str
    ) -> Dict:
        """Evaluate a single resume against a job description"""
        start_time = time.time()
        
        try:
            # Process resume
            resume_data = self.matcher.process_resume(resume_text)
            job_data = self.matcher.process_job(job_text)
            
            # Match
            match_result = self.matcher.match(
                resume_text,
                job_text,
                resume_data['skills'],
                job_data['required_skills'],
                resume_data.get('sections')
            )
            
            processing_time = time.time() - start_time
            
            return {
                'category': category,
                'success': True,
                'skills_extracted': len(resume_data['skills']),
                'skills_matched': len(match_result.skill_match.matched),
                'skills_missing': len(match_result.skill_match.missing),
                'skill_f1': match_result.skill_match.f1_score,
                'semantic_score': match_result.semantic_score_tfidf,
                'overall_score': match_result.overall_score,
                'fit_level': match_result.fit_level,
                'processing_time': processing_time,
                'resume_skills': resume_data['skills'],
                'job_skills': job_data['required_skills'],
                'matched_skills': match_result.skill_match.matched
            }
            
        except Exception as e:
            return {
                'category': category,
                'success': False,
                'error': str(e),
                'processing_time': time.time() - start_time
            }
    
    def run_evaluation(
        self,
        df: pd.DataFrame,
        n_samples: int = 200,
        random_seed: int = 42
    ) -> Dict:
        """Run full evaluation on dataset"""
        
        print(f"\n{'='*60}")
        print("RESUME JOB MATCHER EVALUATION")
        print(f"{'='*60}")
        print(f"Samples: {n_samples}")
        print(f"Categories: {df['Category'].nunique()}")
        print(f"{'='*60}\n")
        
        # Sample data
        random.seed(random_seed)
        categories = df['Category'].unique().tolist()
        
        # Stratified sampling - equal samples per category
        samples_per_category = max(1, n_samples // len(categories))
        sampled_df = df.groupby('Category').apply(
            lambda x: x.sample(n=min(len(x), samples_per_category), random_state=random_seed)
        ).reset_index(drop=True)
        
        print(f"Sampled {len(sampled_df)} resumes across {len(categories)} categories")
        
        # Create job descriptions
        job_descriptions = self.create_synthetic_job_descriptions(categories)
        
        # Evaluate
        results = []
        category_stats = defaultdict(list)
        
        for idx, row in sampled_df.iterrows():
            category = row['Category']
            resume_text = row['Resume']
            job_text = job_descriptions.get(category, "")
            
            result = self.evaluate_single_resume(resume_text, job_text, category)
            results.append(result)
            
            if result['success']:
                category_stats[category].append(result)
            
            # Progress
            if (idx + 1) % 50 == 0:
                print(f"Processed {idx + 1}/{len(sampled_df)} resumes...")
        
        # Aggregate results
        successful = [r for r in results if r['success']]
        
        overall_metrics = {
            'total_samples': len(results),
            'successful_samples': len(successful),
            'success_rate': len(successful) / len(results) * 100,
            'avg_skills_extracted': np.mean([r['skills_extracted'] for r in successful]),
            'avg_skills_matched': np.mean([r['skills_matched'] for r in successful]),
            'avg_skill_f1': np.mean([r['skill_f1'] for r in successful]),
            'avg_semantic_score': np.mean([r['semantic_score'] for r in successful]),
            'avg_overall_score': np.mean([r['overall_score'] for r in successful]),
            'avg_processing_time': np.mean([r['processing_time'] for r in successful]),
            'fit_distribution': {
                'Strong': sum(1 for r in successful if r['fit_level'] == 'Strong'),
                'Moderate': sum(1 for r in successful if r['fit_level'] == 'Moderate'),
                'Weak': sum(1 for r in successful if r['fit_level'] == 'Weak')
            }
        }
        
        # Per-category analysis
        category_analysis = {}
        for category, cat_results in category_stats.items():
            if cat_results:
                # Most common skills
                all_skills = []
                for r in cat_results:
                    all_skills.extend(r['resume_skills'])
                skill_counts = defaultdict(int)
                for skill in all_skills:
                    skill_counts[skill] += 1
                top_skills = sorted(skill_counts.items(), key=lambda x: -x[1])[:10]
                
                category_analysis[category] = {
                    'sample_count': len(cat_results),
                    'avg_skills_extracted': np.mean([r['skills_extracted'] for r in cat_results]),
                    'avg_overall_score': np.mean([r['overall_score'] for r in cat_results]),
                    'avg_skill_f1': np.mean([r['skill_f1'] for r in cat_results]),
                    'top_skills': [s[0] for s in top_skills],
                    'fit_distribution': {
                        'Strong': sum(1 for r in cat_results if r['fit_level'] == 'Strong'),
                        'Moderate': sum(1 for r in cat_results if r['fit_level'] == 'Moderate'),
                        'Weak': sum(1 for r in cat_results if r['fit_level'] == 'Weak')
                    }
                }
        
        self.results = results
        self.category_results = category_analysis
        
        return {
            'overall': overall_metrics,
            'per_category': category_analysis,
            'raw_results': results
        }
    
    def generate_report(self, evaluation_results: Dict, output_dir: str = "evaluation") -> str:
        """Generate evaluation report"""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        overall = evaluation_results['overall']
        per_category = evaluation_results['per_category']
        
        report = []
        report.append("=" * 70)
        report.append("RESUME JOB MATCHER - EVALUATION REPORT")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 70)
        report.append("")
        
        # Overall Metrics
        report.append("OVERALL METRICS")
        report.append("-" * 40)
        report.append(f"Total Samples Tested:     {overall['total_samples']}")
        report.append(f"Successful Processing:    {overall['successful_samples']} ({overall['success_rate']:.1f}%)")
        report.append(f"Avg Skills Extracted:     {overall['avg_skills_extracted']:.1f}")
        report.append(f"Avg Skills Matched:       {overall['avg_skills_matched']:.1f}")
        report.append(f"Avg Skill F1 Score:       {overall['avg_skill_f1']:.3f}")
        report.append(f"Avg Semantic Similarity:  {overall['avg_semantic_score']:.1f}%")
        report.append(f"Avg Overall Match Score:  {overall['avg_overall_score']:.1f}%")
        report.append(f"Avg Processing Time:      {overall['avg_processing_time']*1000:.1f}ms")
        report.append("")
        
        # Fit Distribution
        report.append("FIT LEVEL DISTRIBUTION")
        report.append("-" * 40)
        total = sum(overall['fit_distribution'].values())
        for level, count in overall['fit_distribution'].items():
            pct = count / total * 100 if total > 0 else 0
            report.append(f"  {level}: {count} ({pct:.1f}%)")
        report.append("")
        
        # Per-Category Results
        report.append("PER-CATEGORY RESULTS")
        report.append("-" * 40)
        report.append(f"{'Category':<25} {'Samples':<8} {'Avg Score':<10} {'Skill F1':<10} {'Top Skills'}")
        report.append("-" * 80)
        
        for category, stats in sorted(per_category.items()):
            top_skills = ", ".join(stats['top_skills'][:3])
            report.append(
                f"{category:<25} {stats['sample_count']:<8} "
                f"{stats['avg_overall_score']:<10.1f} {stats['avg_skill_f1']:<10.3f} "
                f"{top_skills}"
            )
        
        report.append("")
        report.append("=" * 70)
        report.append("END OF REPORT")
        report.append("=" * 70)
        
        report_text = "\n".join(report)
        
        # Save report
        report_path = output_path / "evaluation_report.txt"
        with open(report_path, 'w') as f:
            f.write(report_text)
        
        # Save JSON results
        json_path = output_path / "evaluation_results.json"
        
        # Convert to JSON-serializable format
        json_results = {
            'timestamp': datetime.now().isoformat(),
            'overall': overall,
            'per_category': per_category
        }
        
        with open(json_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"\nReport saved to: {report_path}")
        print(f"JSON results saved to: {json_path}")
        
        return report_text
    
    def run_baseline_comparison(
        self,
        df: pd.DataFrame,
        n_samples: int = 50
    ) -> Dict:
        """Compare against simple keyword baseline"""
        
        print("\nRunning baseline comparison...")
        
        # Sample
        sample_df = df.sample(n=min(n_samples, len(df)), random_state=42)
        categories = sample_df['Category'].unique()
        job_descriptions = self.create_synthetic_job_descriptions(list(categories))
        
        results = {
            'our_method': [],
            'baseline': []
        }
        
        for _, row in sample_df.iterrows():
            category = row['Category']
            resume = row['Resume']
            job = job_descriptions.get(category, "")
            
            # Our method
            resume_data = self.matcher.process_resume(resume)
            job_data = self.matcher.process_job(job)
            
            our_skills = set(resume_data['skills'])
            job_skills = set(job_data['required_skills'])
            
            our_matched = our_skills & job_skills
            our_recall = len(our_matched) / len(job_skills) if job_skills else 0
            
            # Baseline: Simple word overlap
            resume_words = set(resume.lower().split())
            job_words = set(job.lower().split())
            baseline_matched = resume_words & job_words
            baseline_recall = len(baseline_matched) / len(job_words) if job_words else 0
            
            results['our_method'].append(our_recall)
            results['baseline'].append(baseline_recall)
        
        comparison = {
            'our_method_avg_recall': np.mean(results['our_method']),
            'baseline_avg_recall': np.mean(results['baseline']),
            'improvement': (np.mean(results['our_method']) - np.mean(results['baseline'])) * 100
        }
        
        print(f"\nBASELINE COMPARISON:")
        print(f"  Our Method Avg Recall: {comparison['our_method_avg_recall']:.3f}")
        print(f"  Baseline Avg Recall:   {comparison['baseline_avg_recall']:.3f}")
        print(f"  Improvement:           {comparison['improvement']:+.1f}%")
        
        return comparison


# =============================================================================
# Main
# =============================================================================

def main():
    """Run full evaluation"""
    
    # Initialize evaluator
    evaluator = ResumeMatcherEvaluator(data_dir="data/kaggle_resumes")
    
    try:
        # Load data
        df = evaluator.load_kaggle_dataset()
        
        # Run evaluation
        results = evaluator.run_evaluation(df, n_samples=200)
        
        # Generate report
        report = evaluator.generate_report(results)
        print("\n" + report)
        
        # Baseline comparison
        evaluator.run_baseline_comparison(df, n_samples=50)
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nTo run evaluation:")
        print("1. Download dataset from: https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset")
        print("2. Create directory: data/kaggle_resumes/")
        print("3. Place UpdatedResumeDataSet.csv in that directory")
        print("4. Run this script again")


if __name__ == "__main__":
    main()
