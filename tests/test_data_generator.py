"""
Test cases for EdTech Data Generator
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data_generator import EdTechDataGenerator

class TestEdTechDataGenerator:
    """Test cases for EdTechDataGenerator class"""
    
    def test_init(self):
        """Test generator initialization"""
        generator = EdTechDataGenerator(n_samples=100, random_state=42)
        assert generator.n_samples == 100
        assert generator.random_state == 42
        assert len(generator.companies) == 4
        
    def test_data_generation(self):
        """Test data generation functionality"""
        generator = EdTechDataGenerator(n_samples=100, random_state=42)
        df = generator.generate_student_data()
        
        # Check data shape and basic properties
        assert len(df) == 100
        assert 'student_id' in df.columns
        assert 'churn' in df.columns
        assert 'company' in df.columns
        
        # Check data types
        assert df['age'].dtype in [np.int64, np.int32]
        assert df['churn'].dtype in [np.int64, np.int32]
        assert pd.api.types.is_string_dtype(df['company'])
        
        # Check value ranges
        assert df['age'].min() >= 18
        assert df['age'].max() <= 45
        assert df['churn'].isin([0, 1]).all()
        assert df['assignment_completion_rate'].between(0, 1).all()
        
    def test_companies_and_courses(self):
        """Test that all companies and courses are properly represented"""
        generator = EdTechDataGenerator(n_samples=1000, random_state=42)
        df = generator.generate_student_data()
        
        # Check companies
        expected_companies = ['Newton School', 'Scaler', 'Upgrad', 'Simplilearn']
        actual_companies = df['company'].unique()
        assert all(company in expected_companies for company in actual_companies)
        
        # Check courses per company
        for company in expected_companies:
            company_courses = df[df['company'] == company]['course'].unique()
            expected_courses = generator.course_categories[company]
            assert all(course in expected_courses for course in company_courses)
    
    def test_churn_calculation(self):
        """Test churn probability calculation"""
        generator = EdTechDataGenerator(n_samples=1000, random_state=42)
        
        # Test with extreme values
        high_risk_prob = generator._calculate_churn_probability(
            age=45, work_exp=0, login_freq=0.5, assignment_rate=0.2,
            quiz_score=30, project_score=40, mentor_interactions=0,
            support_satisfaction=3, progress=10, days_enrolled=100,
            total_course_days=180, job_status='Employed', payment_mode='EMI'
        )
        
        low_risk_prob = generator._calculate_churn_probability(
            age=25, work_exp=3, login_freq=6, assignment_rate=0.95,
            quiz_score=90, project_score=95, mentor_interactions=5,
            support_satisfaction=9, progress=80, days_enrolled=60,
            total_course_days=180, job_status='Student', payment_mode='Scholarship'
        )
        
        assert 0 <= high_risk_prob <= 1
        assert 0 <= low_risk_prob <= 1
        assert high_risk_prob > low_risk_prob
    
    def test_save_data(self, tmp_path):
        """Test data saving functionality"""
        generator = EdTechDataGenerator(n_samples=100, random_state=42)
        
        # Save to temporary directory
        output_dir = tmp_path / "test_data"
        generator.save_data(str(output_dir))
        
        # Check if file was created
        data_file = output_dir / "edtech_student_data.csv"
        assert data_file.exists()
        
        # Load and verify data
        df = pd.read_csv(data_file)
        assert len(df) == 100
        assert 'churn' in df.columns

if __name__ == "__main__":
    pytest.main([__file__])
