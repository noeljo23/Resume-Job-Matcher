"""
NER Evaluation Module
=====================
Comprehensive evaluation of Named Entity Recognition for resumes.

Tests:
1. Skill extraction accuracy
2. Education entity extraction
3. Experience entity extraction
4. End-to-end NER pipeline

Author: Noel John
Course: IE 7500 Applied NLP, Northeastern University
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List
from collections import defaultdict

from ner_extractor import (
    ResumeNERExtractor,
    NEREvaluator,
    ResumeEntities,
    SKILL_TAXONOMY,
    ALL_SKILLS
)


# =============================================================================
# Test Data with Ground Truth
# =============================================================================

TEST_CASES = [
    {
        'name': 'Data Scientist Resume',
        'text': """
            JANE SMITH
            Senior Data Scientist | jane.smith@email.com | San Francisco, CA
            
            SUMMARY
            Data Scientist with 6 years of experience in machine learning and 
            statistical analysis. Expert in Python, TensorFlow, and AWS.
            
            EXPERIENCE
            
            Senior Data Scientist, Google (Jan 2021 - Present)
            • Developed NLP models using PyTorch and Hugging Face Transformers
            • Built recommendation systems processing 1B+ events daily
            • Led team of 4 ML engineers
            
            Data Scientist, Meta (Jun 2018 - Dec 2020)
            • Created computer vision models with TensorFlow and Keras
            • Deployed models on AWS SageMaker
            • Reduced inference latency by 40%
            
            EDUCATION
            Ph.D. in Computer Science, Stanford University, 2018
            B.S. in Mathematics, MIT, 2013
            
            SKILLS
            Python, PyTorch, TensorFlow, Keras, SQL, AWS, Docker, Kubernetes,
            Machine Learning, Deep Learning, NLP, Computer Vision, Spark
            
            CERTIFICATIONS
            AWS Certified Machine Learning Specialty
            Google Cloud Professional Data Engineer
        """,
        'ground_truth': {
            'skills': [
                'python', 'pytorch', 'tensorflow', 'keras', 'sql', 'aws',
                'docker', 'kubernetes', 'machine learning', 'deep learning',
                'nlp', 'computer vision', 'spark', 'hugging face', 'sagemaker',
                'statistical analysis'
            ],
            'degrees': ['Ph.D.', 'B.S.'],
            'institutions': ['Stanford University', 'MIT'],
            'job_titles': ['Senior Data Scientist', 'Data Scientist'],
            'companies': ['Google', 'Meta'],
            'certifications': [
                'AWS Certified Machine Learning Specialty',
                'Google Cloud Professional Data Engineer'
            ]
        }
    },
    {
        'name': 'Software Engineer Resume',
        'text': """
            ALEX JOHNSON
            Full Stack Software Engineer
            alex.johnson@email.com | Seattle, WA
            
            EXPERIENCE
            
            Software Engineer, Amazon (2020 - Present)
            • Built microservices using Java and Spring Boot
            • Developed CI/CD pipelines with Jenkins and Docker
            • Managed databases with PostgreSQL and DynamoDB
            
            Junior Developer, Startup Inc (2018 - 2020)
            • Created React frontend applications
            • Built REST APIs with Node.js and Express
            • Used Git for version control
            
            EDUCATION
            Bachelor of Science in Computer Science
            University of Washington, 2018
            
            SKILLS
            Java, Python, JavaScript, React, Node.js, Spring Boot,
            Docker, Kubernetes, AWS, PostgreSQL, MongoDB, Git
        """,
        'ground_truth': {
            'skills': [
                'java', 'python', 'javascript', 'react', 'node.js', 'spring boot',
                'docker', 'kubernetes', 'aws', 'postgresql', 'mongodb', 'git',
                'express', 'jenkins', 'dynamodb', 'rest api', 'microservices', 'ci/cd'
            ],
            'degrees': ['Bachelor of Science'],
            'institutions': ['University of Washington'],
            'job_titles': ['Software Engineer', 'Full Stack Software Engineer', 'Junior Developer'],
            'companies': ['Amazon', 'Startup Inc']
        }
    },
    {
        'name': 'Business Analyst Resume',
        'text': """
            SARAH CHEN
            Business Intelligence Analyst
            sarah.chen@email.com | New York, NY
            
            EXPERIENCE
            
            Senior Business Analyst, JPMorgan Chase (2019 - Present)
            • Created executive dashboards using Tableau and Power BI
            • Performed data analysis with SQL and Python
            • Developed ETL pipelines with Alteryx
            
            Business Analyst, Deloitte (2016 - 2019)
            • Analyzed financial data using Excel and SQL
            • Built reports in Power BI
            • Managed stakeholder requirements
            
            EDUCATION
            MBA, Columbia Business School, 2016
            B.A. in Economics, NYU, 2014
            
            SKILLS
            SQL, Python, Tableau, Power BI, Excel, Alteryx,
            Data Analysis, Data Visualization, Statistical Analysis
            
            CERTIFICATIONS
            Tableau Desktop Certified Professional
            PMP - Project Management Professional
        """,
        'ground_truth': {
            'skills': [
                'sql', 'python', 'tableau', 'power bi', 'excel', 'alteryx',
                'data analysis', 'data visualization', 'statistical analysis', 'etl'
            ],
            'degrees': ['MBA', 'B.A.'],
            'institutions': ['Columbia Business School', 'NYU'],
            'job_titles': ['Senior Business Analyst', 'Business Analyst', 'Business Intelligence Analyst'],
            'companies': ['JPMorgan Chase', 'Deloitte']
        }
    },
    {
        'name': 'ML Engineer Resume',
        'text': """
            MICHAEL WANG
            Machine Learning Engineer
            m.wang@email.com | Boston, MA
            
            EXPERIENCE
            
            ML Engineer, NVIDIA (2021 - Present)
            • Developed deep learning models for autonomous driving
            • Optimized neural networks using TensorRT
            • Built training pipelines with PyTorch and CUDA
            
            Research Scientist, OpenAI (2019 - 2021)
            • Contributed to GPT model development
            • Implemented reinforcement learning algorithms
            • Published 3 papers at NeurIPS and ICML
            
            EDUCATION
            Ph.D. in Artificial Intelligence, Carnegie Mellon University, 2019
            M.S. in Computer Science, University of Illinois, 2015
            B.S. in Computer Engineering, Purdue University, 2013
            
            SKILLS
            Python, C++, PyTorch, TensorFlow, CUDA, Deep Learning,
            Reinforcement Learning, NLP, Computer Vision, MLOps
        """,
        'ground_truth': {
            'skills': [
                'python', 'c++', 'pytorch', 'tensorflow', 'deep learning',
                'reinforcement learning', 'nlp', 'computer vision', 'mlops'
            ],
            'degrees': ['Ph.D.', 'M.S.', 'B.S.'],
            'institutions': ['Carnegie Mellon University', 'University of Illinois', 'Purdue University'],
            'job_titles': ['ML Engineer', 'Machine Learning Engineer', 'Research Scientist'],
            'companies': ['NVIDIA', 'OpenAI']
        }
    }
]


# =============================================================================
# NER Test Runner
# =============================================================================

class NERTestRunner:
    """Run comprehensive NER evaluation tests"""

    def __init__(self):
        self.extractor = ResumeNERExtractor()
        self.evaluator = NEREvaluator()
        self.results = {}

    def run_single_test(self, test_case: Dict) -> Dict:
        """Run NER on a single test case"""
        name = test_case['name']
        text = test_case['text']
        ground_truth = test_case['ground_truth']

        # Extract entities
        entities = self.extractor.extract(text)

        # Evaluate each entity type
        results = {
            'name': name,
            'extracted': entities.to_dict(),
            'metrics': {}
        }

        # Skills evaluation
        if 'skills' in ground_truth:
            results['metrics']['skills'] = self.evaluator.evaluate(
                entities.get_skills_flat(),
                ground_truth['skills']
            )

        # Degrees evaluation
        if 'degrees' in ground_truth:
            predicted_degrees = [e.text for e in entities.degrees]
            results['metrics']['degrees'] = self.evaluator.evaluate(
                predicted_degrees,
                ground_truth['degrees']
            )

        # Institutions evaluation
        if 'institutions' in ground_truth:
            predicted_institutions = [e.text for e in entities.institutions]
            results['metrics']['institutions'] = self.evaluator.evaluate(
                predicted_institutions,
                ground_truth['institutions']
            )

        # Job titles evaluation
        if 'job_titles' in ground_truth:
            predicted_titles = [e.text for e in entities.job_titles]
            results['metrics']['job_titles'] = self.evaluator.evaluate(
                predicted_titles,
                ground_truth['job_titles']
            )

        # Companies evaluation
        if 'companies' in ground_truth:
            predicted_companies = [e.text for e in entities.companies]
            results['metrics']['companies'] = self.evaluator.evaluate(
                predicted_companies,
                ground_truth['companies']
            )

        # Certifications evaluation
        if 'certifications' in ground_truth:
            predicted_certs = [e.text for e in entities.certifications]
            results['metrics']['certifications'] = self.evaluator.evaluate(
                predicted_certs,
                ground_truth['certifications']
            )

        return results

    def run_all_tests(self) -> Dict:
        """Run all test cases"""
        print("\n" + "=" * 70)
        print("NER EVALUATION - COMPREHENSIVE TEST SUITE")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)

        all_results = []
        aggregated_metrics = defaultdict(
            lambda: {'precision': [], 'recall': [], 'f1': []})

        for test_case in TEST_CASES:
            print(f"\nTesting: {test_case['name']}...")
            result = self.run_single_test(test_case)
            all_results.append(result)

            # Print individual results
            print(f"  Entity Type       | Precision | Recall | F1")
            print(f"  " + "-" * 50)

            for entity_type, metrics in result['metrics'].items():
                print(
                    f"  {entity_type:<17} | {metrics['precision']:.3f}     | {metrics['recall']:.3f}  | {metrics['f1']:.3f}")

                # Aggregate
                aggregated_metrics[entity_type]['precision'].append(
                    metrics['precision'])
                aggregated_metrics[entity_type]['recall'].append(
                    metrics['recall'])
                aggregated_metrics[entity_type]['f1'].append(metrics['f1'])

        # Calculate averages
        print("\n" + "=" * 70)
        print("AGGREGATED RESULTS (Average across all test cases)")
        print("=" * 70)
        print(f"  Entity Type       | Precision | Recall | F1")
        print(f"  " + "-" * 50)

        avg_metrics = {}
        for entity_type, values in aggregated_metrics.items():
            avg_precision = sum(values['precision']) / len(values['precision'])
            avg_recall = sum(values['recall']) / len(values['recall'])
            avg_f1 = sum(values['f1']) / len(values['f1'])

            avg_metrics[entity_type] = {
                'precision': avg_precision,
                'recall': avg_recall,
                'f1': avg_f1
            }

            print(
                f"  {entity_type:<17} | {avg_precision:.3f}     | {avg_recall:.3f}  | {avg_f1:.3f}")

        # Overall average
        all_f1s = [m['f1'] for m in avg_metrics.values()]
        overall_f1 = sum(all_f1s) / len(all_f1s) if all_f1s else 0

        print(f"  " + "-" * 50)
        print(f"  {'OVERALL':<17} | {'-':<9} | {'-':<6} | {overall_f1:.3f}")

        self.results = {
            'timestamp': datetime.now().isoformat(),
            'test_count': len(TEST_CASES),
            'individual_results': all_results,
            'aggregated_metrics': avg_metrics,
            'overall_f1': overall_f1
        }

        return self.results

    def test_skill_taxonomy_coverage(self) -> Dict:
        """Test skill taxonomy coverage"""
        print("\n" + "=" * 70)
        print("SKILL TAXONOMY COVERAGE")
        print("=" * 70)

        total_skills = len(ALL_SKILLS)
        skills_by_category = defaultdict(int)

        for skill in ALL_SKILLS:
            from ner_extractor import SKILL_TO_CATEGORY
            category = SKILL_TO_CATEGORY.get(skill, 'other')
            skills_by_category[category] += 1

        print(f"\nTotal skills in taxonomy: {total_skills}")
        print(f"\nSkills by category:")

        for category, count in sorted(skills_by_category.items(), key=lambda x: -x[1]):
            print(f"  {category:<25} {count:>4} skills")

        return {
            'total_skills': total_skills,
            'by_category': dict(skills_by_category)
        }

    def generate_report(self, output_dir: str = "evaluation") -> str:
        """Generate evaluation report"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if not self.results:
            self.run_all_tests()

        report_lines = [
            "=" * 70,
            "NER EVALUATION REPORT",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 70,
            "",
            "SUMMARY",
            "-" * 40,
            f"Test Cases: {self.results['test_count']}",
            f"Overall F1 Score: {self.results['overall_f1']:.3f}",
            "",
            "METRICS BY ENTITY TYPE",
            "-" * 40,
        ]

        for entity_type, metrics in self.results['aggregated_metrics'].items():
            report_lines.append(
                f"{entity_type}: P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, F1={metrics['f1']:.3f}"
            )

        report_lines.extend([
            "",
            "=" * 70,
            "END OF REPORT",
            "=" * 70
        ])

        report_text = "\n".join(report_lines)

        # Save report
        report_path = output_path / "ner_evaluation_report.txt"
        with open(report_path, 'w') as f:
            f.write(report_text)

        # Save JSON
        json_path = output_path / "ner_evaluation_results.json"
        with open(json_path, 'w') as f:
            json.dump({
                'timestamp': self.results['timestamp'],
                'test_count': self.results['test_count'],
                'aggregated_metrics': self.results['aggregated_metrics'],
                'overall_f1': self.results['overall_f1']
            }, f, indent=2)

        print(f"\nReport saved to: {report_path}")
        print(f"JSON saved to: {json_path}")

        return report_text


# =============================================================================
# Quick Demo
# =============================================================================

def demo_ner_extraction():
    """Demo NER extraction on a sample resume"""
    print("\n" + "=" * 70)
    print("NER EXTRACTION DEMO")
    print("=" * 70)

    sample = """
    NOEL JOHN
    Data Analytics Graduate Student
    john.noe@northeastern.edu | Vancouver, BC
    
    EDUCATION
    M.S. in Data Analytics Engineering, Northeastern University (Expected 2026)
    GPA: 3.92
    
    EXPERIENCE
    
    Business Analyst, Media.net (2020 - 2023)
    • Built production dashboards using Python and SQL
    • Processed millions of daily records
    • Automated ETL workflows
    
    Teaching Assistant, Northeastern University (2024 - Present)
    • Assist students with data analytics coursework
    
    SKILLS
    Python, SQL, Tableau, Power BI, PostgreSQL, Git,
    Machine Learning, Data Analysis, ETL, Streamlit
    
    PROJECTS
    - RGB2Point: Deep learning for 3D point cloud generation
    - Bikini Bottom Current Classifier: Ocean current analysis
    """

    extractor = ResumeNERExtractor()
    entities = extractor.extract(sample)

    print("\nExtracted Entities:")
    print("-" * 40)

    print(f"\nSKILLS ({len(entities.skills)}):")
    for skill in sorted(set(e.text for e in entities.skills)):
        print(f"  • {skill}")

    print(f"\nDEGREES ({len(entities.degrees)}):")
    for e in entities.degrees:
        print(f"  • {e.text}")

    print(f"\nINSTITUTIONS ({len(entities.institutions)}):")
    for e in entities.institutions:
        print(f"  • {e.text}")

    print(f"\nJOB TITLES ({len(entities.job_titles)}):")
    for e in entities.job_titles:
        print(f"  • {e.text}")

    print(f"\nCOMPANIES ({len(entities.companies)}):")
    for e in entities.companies:
        print(f"  • {e.text}")

    print(f"\nLOCATIONS ({len(entities.locations)}):")
    for e in entities.locations:
        print(f"  • {e.text}")


# =============================================================================
# Main
# =============================================================================

def main():
    """Run NER evaluation"""

    # Run demo
    demo_ner_extraction()

    # Run full evaluation
    runner = NERTestRunner()
    results = runner.run_all_tests()

    # Test taxonomy coverage
    runner.test_skill_taxonomy_coverage()

    # Generate report
    runner.generate_report()

    print("\n" + "=" * 70)
    print("NER EVALUATION COMPLETE")
    print(f"Overall F1 Score: {results['overall_f1']:.3f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
