"""
Real-World Evaluation on Kaggle Resume Dataset
===============================================
Tests NER and skill extraction on actual resumes from Kaggle.

Dataset: UpdatedResumeDataSet.csv (962 resumes, 24 categories)
Source: https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset

Author: Noel John
Course: IE 7500 Applied NLP, Northeastern University
"""

import pandas as pd
import numpy as np
import time
import json
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

# Import our pipeline
from pipeline_v2 import ResumeJobMatcherV2, SKILL_TAXONOMY, ALL_SKILLS, SKILL_TO_CATEGORY


# =============================================================================
# Category-Specific Expected Skills (Ground Truth Approximation)
# =============================================================================

# What skills we EXPECT to find in each resume category
CATEGORY_EXPECTED_SKILLS = {
    'Data Science': [
        'python', 'sql', 'machine learning', 'deep learning', 'tensorflow',
        'pandas', 'numpy', 'scikit-learn', 'tableau', 'r', 'statistics'
    ],
    'HR': [
        'excel', 'communication', 'leadership', 'project management'
    ],
    'Advocate': [
        'communication', 'leadership'
    ],
    'Arts': [
        'communication', 'presentation skills'
    ],
    'Web Designing': [
        'html', 'css', 'javascript', 'react', 'angular', 'php', 'wordpress'
    ],
    'Mechanical Engineer': [
        'autocad', 'matlab', 'excel', 'project management'
    ],
    'Sales': [
        'communication', 'excel', 'salesforce', 'leadership'
    ],
    'Health and fitness': [
        'communication'
    ],
    'Civil Engineer': [
        'autocad', 'excel', 'project management'
    ],
    'Java Developer': [
        'java', 'sql', 'spring', 'spring boot', 'javascript', 'html', 'git', 'docker'
    ],
    'Business Analyst': [
        'sql', 'excel', 'tableau', 'power bi', 'python', 'data analysis', 'communication'
    ],
    'SAP Developer': [
        'sap', 'sql', 'java', 'excel'
    ],
    'Automation Testing': [
        'selenium', 'java', 'python', 'sql', 'git', 'jenkins'
    ],
    'Electrical Engineering': [
        'matlab', 'autocad', 'excel', 'python'
    ],
    'Operations Manager': [
        'excel', 'project management', 'leadership', 'communication'
    ],
    'Python Developer': [
        'python', 'django', 'flask', 'sql', 'html', 'css', 'javascript', 'git', 'docker'
    ],
    'DevOps Engineer': [
        'docker', 'kubernetes', 'aws', 'jenkins', 'linux', 'python', 'git', 'terraform'
    ],
    'Network Security Engineer': [
        'linux', 'python', 'aws', 'networking', 'security'
    ],
    'PMO': [
        'excel', 'project management', 'agile', 'scrum', 'jira', 'communication'
    ],
    'Database': [
        'sql', 'mysql', 'postgresql', 'oracle', 'mongodb', 'python'
    ],
    'Hadoop': [
        'hadoop', 'spark', 'hive', 'python', 'sql', 'java', 'kafka'
    ],
    'ETL Developer': [
        'sql', 'python', 'etl', 'data warehouse', 'informatica'
    ],
    'DotNet Developer': [
        'c#', '.net', 'sql', 'javascript', 'html', 'css', 'azure'
    ],
    'Blockchain': [
        'python', 'javascript', 'sql', 'docker'
    ],
    'Testing': [
        'selenium', 'sql', 'java', 'python', 'git'
    ]
}


# =============================================================================
# Evaluation Functions
# =============================================================================

def load_kaggle_data(filepath: str) -> pd.DataFrame:
    """Load and clean the Kaggle resume dataset"""
    print(f"Loading data from: {filepath}")
    
    df = pd.read_csv(filepath, encoding='utf-8', on_bad_lines='skip')
    
    # Clean column names
    df.columns = df.columns.str.strip()
    
    print(f"Loaded {len(df)} resumes")
    print(f"Categories: {df['Category'].nunique()}")
    print(f"\nCategory distribution:")
    print(df['Category'].value_counts())
    
    return df


def clean_resume_text(text: str) -> str:
    """Clean resume text"""
    if pd.isna(text):
        return ""
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Remove common artifacts
    text = text.replace('\\r\\n', ' ')
    text = text.replace('\\n', ' ')
    text = text.replace('Â', ' ')
    text = text.replace('â€™', "'")
    text = text.replace('â€"', "-")
    
    return text


def evaluate_skill_extraction(
    matcher: ResumeJobMatcherV2,
    df: pd.DataFrame,
    sample_size: int = 100
) -> Dict:
    """Evaluate skill extraction on real resumes"""
    
    print("\n" + "=" * 70)
    print("SKILL EXTRACTION EVALUATION ON KAGGLE DATASET")
    print("=" * 70)
    
    results_by_category = defaultdict(lambda: {
        'total': 0,
        'avg_skills': [],
        'skill_counts': defaultdict(int),
        'expected_found': [],
        'processing_times': []
    })
    
    # Sample from each category
    categories = df['Category'].unique()
    samples_per_category = max(1, sample_size // len(categories))
    
    total_processed = 0
    all_skills_extracted = []
    
    for category in categories:
        category_df = df[df['Category'] == category]
        sample_df = category_df.sample(min(samples_per_category, len(category_df)), random_state=42)
        
        for idx, row in sample_df.iterrows():
            resume_text = clean_resume_text(row['Resume'])
            
            if len(resume_text) < 100:  # Skip very short resumes
                continue
            
            # Time the extraction
            start_time = time.time()
            extracted_skills = matcher.entity_extractor.extract_skills_flat(resume_text)
            processing_time = (time.time() - start_time) * 1000  # ms
            
            # Store results
            results_by_category[category]['total'] += 1
            results_by_category[category]['avg_skills'].append(len(extracted_skills))
            results_by_category[category]['processing_times'].append(processing_time)
            
            # Count skill frequency
            for skill in extracted_skills:
                results_by_category[category]['skill_counts'][skill] += 1
            
            # Check expected skills
            expected = CATEGORY_EXPECTED_SKILLS.get(category, [])
            if expected:
                found = set(s.lower() for s in extracted_skills) & set(s.lower() for s in expected)
                recall = len(found) / len(expected) if expected else 0
                results_by_category[category]['expected_found'].append(recall)
            
            all_skills_extracted.extend(extracted_skills)
            total_processed += 1
    
    # Print results
    print(f"\nProcessed {total_processed} resumes across {len(categories)} categories")
    print("\n" + "-" * 70)
    print(f"{'Category':<25} {'Count':<8} {'Avg Skills':<12} {'Avg Time (ms)':<15} {'Expected Recall'}")
    print("-" * 70)
    
    overall_skills = []
    overall_times = []
    overall_recalls = []
    
    for category in sorted(results_by_category.keys()):
        data = results_by_category[category]
        if data['total'] == 0:
            continue
            
        avg_skills = np.mean(data['avg_skills']) if data['avg_skills'] else 0
        avg_time = np.mean(data['processing_times']) if data['processing_times'] else 0
        avg_recall = np.mean(data['expected_found']) if data['expected_found'] else 0
        
        overall_skills.extend(data['avg_skills'])
        overall_times.extend(data['processing_times'])
        overall_recalls.extend(data['expected_found'])
        
        print(f"{category:<25} {data['total']:<8} {avg_skills:<12.1f} {avg_time:<15.1f} {avg_recall:.2f}")
    
    print("-" * 70)
    print(f"{'OVERALL':<25} {total_processed:<8} {np.mean(overall_skills):<12.1f} {np.mean(overall_times):<15.1f} {np.mean(overall_recalls):.2f}")
    
    # Top skills across all resumes
    skill_freq = defaultdict(int)
    for skill in all_skills_extracted:
        skill_freq[skill] += 1
    
    print("\n" + "=" * 70)
    print("TOP 20 MOST FREQUENTLY EXTRACTED SKILLS")
    print("=" * 70)
    
    top_skills = sorted(skill_freq.items(), key=lambda x: -x[1])[:20]
    for skill, count in top_skills:
        category = SKILL_TO_CATEGORY.get(skill, 'other')
        print(f"  {skill:<25} {count:>5} occurrences  ({category})")
    
    return {
        'total_processed': total_processed,
        'categories': len(categories),
        'avg_skills_per_resume': np.mean(overall_skills),
        'avg_processing_time_ms': np.mean(overall_times),
        'avg_expected_recall': np.mean(overall_recalls),
        'top_skills': top_skills[:10],
        'results_by_category': {k: {
            'count': v['total'],
            'avg_skills': np.mean(v['avg_skills']) if v['avg_skills'] else 0,
            'avg_recall': np.mean(v['expected_found']) if v['expected_found'] else 0
        } for k, v in results_by_category.items()}
    }


def evaluate_ner_quality(
    matcher: ResumeJobMatcherV2,
    df: pd.DataFrame,
    sample_size: int = 50
) -> Dict:
    """Evaluate NER entity extraction quality"""
    
    print("\n" + "=" * 70)
    print("NER ENTITY EXTRACTION EVALUATION")
    print("=" * 70)
    
    sample_df = df.sample(min(sample_size, len(df)), random_state=42)
    
    entity_counts = {
        'organizations': [],
        'locations': [],
        'dates': [],
        'education': [],
        'job_titles': []
    }
    
    for idx, row in sample_df.iterrows():
        resume_text = clean_resume_text(row['Resume'])
        
        if len(resume_text) < 100:
            continue
        
        entities = matcher.entity_extractor.extract_entities(resume_text)
        
        entity_counts['organizations'].append(len(entities.get('organizations', [])))
        entity_counts['locations'].append(len(entities.get('locations', [])))
        entity_counts['dates'].append(len(entities.get('dates', [])))
        entity_counts['education'].append(len(entities.get('education', [])))
        entity_counts['job_titles'].append(len(entities.get('job_titles', [])))
    
    print(f"\nNER Results on {len(sample_df)} resumes:")
    print("-" * 50)
    print(f"{'Entity Type':<20} {'Avg Count':<15} {'Max':<10} {'Min'}")
    print("-" * 50)
    
    for entity_type, counts in entity_counts.items():
        if counts:
            print(f"{entity_type:<20} {np.mean(counts):<15.1f} {max(counts):<10} {min(counts)}")
    
    return {
        'sample_size': len(sample_df),
        'entity_averages': {k: np.mean(v) for k, v in entity_counts.items() if v}
    }


def run_full_evaluation(filepath: str, sample_size: int = 200):
    """Run complete evaluation"""
    
    print("\n" + "=" * 70)
    print("RESUME JOB MATCHER - REAL WORLD EVALUATION")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Load data
    df = load_kaggle_data(filepath)
    
    # Initialize matcher
    print("\nInitializing matcher...")
    matcher = ResumeJobMatcherV2(use_bert=False)
    
    # Run evaluations
    skill_results = evaluate_skill_extraction(matcher, df, sample_size)
    ner_results = evaluate_ner_quality(matcher, df, sample_size // 4)
    
    # Summary
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"""
    Dataset: Kaggle Resume Dataset
    Total Resumes Processed: {skill_results['total_processed']}
    Categories: {skill_results['categories']}
    
    SKILL EXTRACTION:
      Average skills per resume: {skill_results['avg_skills_per_resume']:.1f}
      Average processing time: {skill_results['avg_processing_time_ms']:.1f} ms
      Expected skill recall: {skill_results['avg_expected_recall']:.2%}
    
    NER EXTRACTION:
      Avg organizations found: {ner_results['entity_averages'].get('organizations', 0):.1f}
      Avg locations found: {ner_results['entity_averages'].get('locations', 0):.1f}
      Avg job titles found: {ner_results['entity_averages'].get('job_titles', 0):.1f}
    """)
    
    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'dataset': 'Kaggle Resume Dataset',
        'skill_extraction': skill_results,
        'ner_extraction': ner_results
    }
    
    output_path = Path('evaluation/kaggle_evaluation_results.json')
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_path}")
    
    return results


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import sys
    
    # Default path - update this to your local path
    default_path = "data/kaggle_resumes/UpdatedResumeDataSet.csv"
    
    # Allow command line argument
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        filepath = default_path
    
    # Check if file exists
    if not Path(filepath).exists():
        print(f"File not found: {filepath}")
        print("Usage: python kaggle_evaluation.py <path_to_csv>")
        print(f"Example: python kaggle_evaluation.py {default_path}")
        sys.exit(1)
    
    # Run evaluation
    run_full_evaluation(filepath, sample_size=200)
