"""
Data Generator for EdTech Student Churn Prediction
Creates synthetic student data for Newton School, Scaler, Upgrad, and Simplilearn
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from pathlib import Path

class EdTechDataGenerator:
    def __init__(self, n_samples=10000, random_state=42):
        self.n_samples = n_samples
        self.random_state = random_state
        np.random.seed(random_state)
        random.seed(random_state)
        
        # EdTech companies
        self.companies = ['Newton School', 'Scaler', 'Upgrad', 'Simplilearn']
        
        # Course categories
        self.course_categories = {
            'Newton School': ['Full Stack Development', 'Data Science', 'Backend Development'],
            'Scaler': ['System Design', 'Data Structures', 'Machine Learning', 'DevOps'],
            'Upgrad': ['MBA', 'Data Science', 'Digital Marketing', 'Product Management'],
            'Simplilearn': ['Cloud Computing', 'Cybersecurity', 'AI/ML', 'Digital Marketing']
        }
        
        # Course durations (in months)
        self.course_durations = {
            'Full Stack Development': 6, 'Data Science': 8, 'Backend Development': 4,
            'System Design': 3, 'Data Structures': 4, 'Machine Learning': 6, 'DevOps': 5,
            'MBA': 24, 'Digital Marketing': 6, 'Product Management': 8,
            'Cloud Computing': 4, 'Cybersecurity': 6, 'AI/ML': 8
        }
    
    def generate_student_data(self):
        """Generate synthetic student data"""
        data = []
        
        for i in range(self.n_samples):
            # Basic student information
            student_id = f"STU_{i+1:06d}"
            company = np.random.choice(self.companies)
            course = np.random.choice(self.course_categories[company])
            
            # Demographics
            age = np.random.normal(26, 4)
            age = max(18, min(45, int(age)))
            
            gender = np.random.choice(['Male', 'Female', 'Other'], p=[0.6, 0.35, 0.05])
            
            education_levels = ['Bachelor', 'Master', 'PhD', 'Diploma']
            education = np.random.choice(education_levels, p=[0.5, 0.3, 0.1, 0.1])
            
            # Geographic information
            cities = ['Mumbai', 'Delhi', 'Bangalore', 'Hyderabad', 'Chennai', 'Pune', 'Kolkata', 'Ahmedabad']
            city = np.random.choice(cities)
            
            # Professional background
            work_experience = max(0, np.random.normal(3, 2))
            work_experience = min(20, int(work_experience))
            
            current_job_status = np.random.choice(['Employed', 'Unemployed', 'Student'], p=[0.6, 0.25, 0.15])
            
            # Course-related features
            course_duration = self.course_durations[course]
            enrollment_date = datetime.now() - timedelta(days=np.random.randint(30, 730))
            
            # Engagement metrics
            login_frequency = max(0, np.random.normal(4, 2))  # logins per week
            avg_session_duration = max(10, np.random.normal(45, 15))  # minutes
            
            assignments_submitted = np.random.poisson(lam=8)
            assignments_total = assignments_submitted + np.random.poisson(lam=2)
            assignment_completion_rate = assignments_submitted / max(1, assignments_total)
            
            # Financial factors
            course_fees_mapping = {
                'Full Stack Development': 150000, 'Data Science': 200000, 'Backend Development': 100000,
                'System Design': 80000, 'Data Structures': 70000, 'Machine Learning': 180000, 'DevOps': 120000,
                'MBA': 500000, 'Digital Marketing': 90000, 'Product Management': 250000,
                'Cloud Computing': 100000, 'Cybersecurity': 150000, 'AI/ML': 200000
            }
            course_fee = course_fees_mapping[course]
            
            payment_mode = np.random.choice(['Full Payment', 'EMI', 'Scholarship'], p=[0.3, 0.6, 0.1])
            financial_aid = 1 if payment_mode == 'Scholarship' else 0
            
            # Performance metrics
            quiz_scores = np.random.beta(2, 2) * 100  # Quiz scores (0-100)
            project_scores = np.random.beta(2.5, 1.5) * 100  # Project scores tend to be higher
            
            # Support and engagement
            forum_posts = np.random.poisson(lam=3)
            mentor_interactions = np.random.poisson(lam=2)
            peer_interactions = np.random.poisson(lam=5)
            
            support_tickets_raised = np.random.poisson(lam=1)
            support_satisfaction = np.random.normal(7, 2)  # 1-10 scale
            support_satisfaction = max(1, min(10, support_satisfaction))
            
            # Device and technical factors
            device_type = np.random.choice(['Desktop', 'Mobile', 'Tablet'], p=[0.5, 0.4, 0.1])
            internet_quality = np.random.choice(['Excellent', 'Good', 'Average', 'Poor'], p=[0.3, 0.4, 0.25, 0.05])
            
            # Additional behavioral features
            weekend_activity = np.random.beta(1.5, 2)  # Weekend engagement ratio
            late_night_study = 1 if np.random.random() < 0.3 else 0
            
            days_since_enrollment = (datetime.now() - enrollment_date).days
            progress_percentage = min(100, max(0, np.random.normal(50, 30)))
            
            # Calculate churn probability based on features
            churn_prob = self._calculate_churn_probability(
                age, work_experience, login_frequency, assignment_completion_rate,
                quiz_scores, project_scores, mentor_interactions, support_satisfaction,
                progress_percentage, days_since_enrollment, course_duration * 30,
                current_job_status, payment_mode
            )
            
            # Generate churn label
            churn = 1 if np.random.random() < churn_prob else 0
            
            # Create record
            record = {
                'student_id': student_id,
                'company': company,
                'course': course,
                'age': age,
                'gender': gender,
                'education_level': education,
                'city': city,
                'work_experience': work_experience,
                'current_job_status': current_job_status,
                'course_duration_months': course_duration,
                'course_fee': course_fee,
                'payment_mode': payment_mode,
                'financial_aid': financial_aid,
                'enrollment_date': enrollment_date.strftime('%Y-%m-%d'),
                'days_since_enrollment': days_since_enrollment,
                'login_frequency_per_week': round(login_frequency, 2),
                'avg_session_duration_minutes': round(avg_session_duration, 2),
                'assignments_submitted': assignments_submitted,
                'assignments_total': assignments_total,
                'assignment_completion_rate': round(assignment_completion_rate, 3),
                'quiz_avg_score': round(quiz_scores, 2),
                'project_avg_score': round(project_scores, 2),
                'progress_percentage': round(progress_percentage, 2),
                'forum_posts': forum_posts,
                'mentor_interactions': mentor_interactions,
                'peer_interactions': peer_interactions,
                'support_tickets_raised': support_tickets_raised,
                'support_satisfaction_score': round(support_satisfaction, 1),
                'device_type': device_type,
                'internet_quality': internet_quality,
                'weekend_activity_ratio': round(weekend_activity, 3),
                'late_night_study': late_night_study,
                'churn': churn
            }
            
            data.append(record)
        
        return pd.DataFrame(data)
    
    def _calculate_churn_probability(self, age, work_exp, login_freq, assignment_rate,
                                   quiz_score, project_score, mentor_interactions,
                                   support_satisfaction, progress, days_enrolled,
                                   total_course_days, job_status, payment_mode):
        """Calculate churn probability based on student features"""
        
        # Base probability
        prob = 0.3
        
        # Age factor (very young or older students might churn more)
        if age < 22 or age > 35:
            prob += 0.1
        
        # Low engagement indicators
        if login_freq < 2:
            prob += 0.2
        if assignment_rate < 0.5:
            prob += 0.25
        if quiz_score < 50:
            prob += 0.15
        if project_score < 60:
            prob += 0.1
        
        # Support and mentorship
        if mentor_interactions < 1:
            prob += 0.1
        if support_satisfaction < 5:
            prob += 0.15
        
        # Progress tracking
        expected_progress = min(100, (days_enrolled / total_course_days) * 100)
        if progress < expected_progress * 0.7:  # Significantly behind
            prob += 0.2
        
        # Employment status
        if job_status == 'Unemployed':
            prob -= 0.05  # Might be more motivated
        elif job_status == 'Employed':
            prob += 0.05  # Might have less time
        
        # Payment mode
        if payment_mode == 'EMI':
            prob += 0.05  # Financial pressure
        elif payment_mode == 'Scholarship':
            prob -= 0.1  # More committed
        
        return max(0, min(1, prob))
    
    def save_data(self, output_dir="./data/raw/"):
        """Generate and save the dataset"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        df = self.generate_student_data()
        output_path = Path(output_dir) / "edtech_student_data.csv"
        df.to_csv(output_path, index=False)
        
        print(f"Dataset generated and saved to: {output_path}")
        print(f"Total samples: {len(df)}")
        print(f"Churn rate: {df['churn'].mean():.2%}")
        print(f"Features: {len(df.columns)}")
        
        # Print company-wise statistics
        print("\nCompany-wise distribution:")
        print(df['company'].value_counts())
        
        print("\nChurn rate by company:")
        print(df.groupby('company')['churn'].mean().sort_values(ascending=False))
        
        return df

if __name__ == "__main__":
    generator = EdTechDataGenerator(n_samples=15000)
    df = generator.save_data()
