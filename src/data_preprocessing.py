"""
Data Preprocessing Module for EdTech Student Churn Prediction
Handles feature engineering, encoding, scaling, and data preparation
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from pathlib import Path
import joblib
import yaml
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    def __init__(self, config_path='./config/config.yaml'):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_selector = None
        self.imputer_numeric = SimpleImputer(strategy='median')
        self.imputer_categorical = SimpleImputer(strategy='most_frequent')
        
    def load_data(self, data_path=None):
        """Load the raw data"""
        if data_path is None:
            data_path = Path(self.config['data']['raw_data_path']) / 'edtech_student_data.csv'
        
        print(f"Loading data from: {data_path}")
        df = pd.read_csv(data_path)
        print(f"Data loaded successfully. Shape: {df.shape}")
        return df
    
    def handle_missing_values(self, df):
        """Handle missing values in the dataset"""
        print("Handling missing values...")
        
        # Separate numeric and categorical columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        # Handle numeric missing values
        if len(numeric_cols) > 0:
            df[numeric_cols] = self.imputer_numeric.fit_transform(df[numeric_cols])
        
        # Handle categorical missing values
        if len(categorical_cols) > 0:
            df[categorical_cols] = self.imputer_categorical.fit_transform(df[categorical_cols])
        
        return df
    
    def feature_engineering(self, df):
        """Create new features from existing ones"""
        print("Engineering new features...")
        
        # Convert enrollment_date to datetime
        df['enrollment_date'] = pd.to_datetime(df['enrollment_date'])
        
        # Time-based features
        df['enrollment_month'] = df['enrollment_date'].dt.month
        df['enrollment_quarter'] = df['enrollment_date'].dt.quarter
        df['enrollment_year'] = df['enrollment_date'].dt.year
        
        # Engagement ratio features
        df['engagement_score'] = (
            df['login_frequency_per_week'] * 0.3 +
            df['assignment_completion_rate'] * 100 * 0.4 +
            df['avg_session_duration_minutes'] * 0.1 +
            df['forum_posts'] * 2 +
            df['mentor_interactions'] * 3
        ) / 100
        
        # Performance score
        df['academic_performance'] = (
            df['quiz_avg_score'] * 0.4 +
            df['project_avg_score'] * 0.6
        )
        
        # Course progress vs expected progress
        expected_days = df['course_duration_months'] * 30
        df['progress_ratio'] = df['progress_percentage'] / (
            (df['days_since_enrollment'] / expected_days * 100).clip(upper=100)
        )
        df['progress_ratio'] = df['progress_ratio'].fillna(1).clip(upper=3)
        
        # Financial burden indicator
        df['fee_to_experience_ratio'] = df['course_fee'] / (df['work_experience'] + 1)
        
        # Support effectiveness
        df['support_effectiveness'] = df['support_satisfaction_score'] / (df['support_tickets_raised'] + 1)
        
        # Learning consistency (combination of various engagement metrics)
        df['learning_consistency'] = (
            df['weekend_activity_ratio'] * 0.3 +
            (df['avg_session_duration_minutes'] / 60) * 0.4 +
            df['assignment_completion_rate'] * 0.3
        )
        
        # Age group categorization
        df['age_group'] = pd.cut(df['age'], 
                                bins=[0, 25, 30, 35, 100], 
                                labels=['Young', 'Mid-Young', 'Mid-Senior', 'Senior'])
        
        # Course difficulty indicator (based on duration and fee)
        course_difficulty = {
            'Data Structures': 'Medium', 'System Design': 'Hard', 'Full Stack Development': 'Hard',
            'Data Science': 'Hard', 'Backend Development': 'Medium', 'Machine Learning': 'Hard',
            'DevOps': 'Medium', 'MBA': 'Hard', 'Digital Marketing': 'Easy', 
            'Product Management': 'Medium', 'Cloud Computing': 'Medium', 
            'Cybersecurity': 'Medium', 'AI/ML': 'Hard'
        }
        df['course_difficulty'] = df['course'].map(course_difficulty)
        
        # Drop original date column
        df = df.drop(['enrollment_date'], axis=1)
        
        print(f"Feature engineering completed. New shape: {df.shape}")
        return df
    
    def encode_categorical_features(self, df, fit=True):
        """Encode categorical features"""
        print("Encoding categorical features...")
        
        categorical_cols = ['company', 'course', 'gender', 'education_level', 'city', 
                          'current_job_status', 'payment_mode', 'device_type', 
                          'internet_quality', 'age_group', 'course_difficulty']
        
        # One-hot encoding for high cardinality features
        high_cardinality_cols = ['company', 'course', 'city']
        
        for col in high_cardinality_cols:
            if col in df.columns:
                if fit:
                    # Create dummy variables
                    dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                    df = pd.concat([df, dummies], axis=1)
                    df = df.drop([col], axis=1)
        
        # Label encoding for low cardinality features
        low_cardinality_cols = [col for col in categorical_cols if col not in high_cardinality_cols]
        
        for col in low_cardinality_cols:
            if col in df.columns:
                if fit:
                    self.label_encoders[col] = LabelEncoder()
                    df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
                else:
                    if col in self.label_encoders:
                        df[col] = self.label_encoders[col].transform(df[col].astype(str))
        
        return df
    
    def scale_features(self, X_train, X_test=None, fit=True):
        """Scale numerical features"""
        print("Scaling numerical features...")
        
        if fit:
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
            
            if X_test is not None:
                X_test_scaled = self.scaler.transform(X_test)
                X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
                return X_train_scaled, X_test_scaled
            
            return X_train_scaled
        else:
            X_scaled = self.scaler.transform(X_train)
            X_scaled = pd.DataFrame(X_scaled, columns=X_train.columns, index=X_train.index)
            return X_scaled
    
    def select_features(self, X, y, k=20):
        """Select top k features based on statistical tests"""
        print(f"Selecting top {k} features...")
        
        # Use chi2 for non-negative features, f_classif for others
        self.feature_selector = SelectKBest(score_func=f_classif, k=k)
        X_selected = self.feature_selector.fit_transform(X, y)
        
        # Get selected feature names
        selected_features = X.columns[self.feature_selector.get_support()].tolist()
        print(f"Selected features: {selected_features}")
        
        return pd.DataFrame(X_selected, columns=selected_features, index=X.index), selected_features
    
    def handle_class_imbalance(self, X, y):
        """Handle class imbalance using SMOTE"""
        print("Handling class imbalance with SMOTE...")
        
        smote = SMOTE(random_state=self.config['data']['random_state'])
        X_balanced, y_balanced = smote.fit_resample(X, y)
        
        print(f"Original class distribution: {np.bincount(y)}")
        print(f"Balanced class distribution: {np.bincount(y_balanced)}")
        
        return pd.DataFrame(X_balanced, columns=X.columns), pd.Series(y_balanced)
    
    def preprocess_data(self, df, target_col='churn', test_size=0.2, apply_smote=True, 
                       feature_selection=True, n_features=20):
        """Complete data preprocessing pipeline"""
        print("Starting data preprocessing pipeline...")
        
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Feature engineering
        df = self.feature_engineering(df)
        
        # Encode categorical features
        df = self.encode_categorical_features(df, fit=True)
        
        # Separate features and target
        X = df.drop([target_col, 'student_id'], axis=1, errors='ignore')
        y = df[target_col]
        
        print(f"Features shape: {X.shape}")
        print(f"Target distribution: {y.value_counts().to_dict()}")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size,
            random_state=self.config['data']['random_state'],
            stratify=y
        )
        
        # Scale features
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test, fit=True)
        
        # Feature selection (optional)
        if feature_selection and n_features < X_train_scaled.shape[1]:
            X_train_selected, selected_features = self.select_features(X_train_scaled, y_train, k=n_features)
            X_test_selected = X_test_scaled[selected_features]
        else:
            X_train_selected = X_train_scaled
            X_test_selected = X_test_scaled
            selected_features = X_train_scaled.columns.tolist()
        
        # Handle class imbalance (optional)
        if apply_smote:
            X_train_balanced, y_train_balanced = self.handle_class_imbalance(X_train_selected, y_train)
        else:
            X_train_balanced = X_train_selected
            y_train_balanced = y_train
        
        print("Data preprocessing completed!")
        print(f"Final training set shape: {X_train_balanced.shape}")
        print(f"Final test set shape: {X_test_selected.shape}")
        
        return {
            'X_train': X_train_balanced,
            'X_test': X_test_selected,
            'y_train': y_train_balanced,
            'y_test': y_test,
            'selected_features': selected_features,
            'feature_names': X_train_balanced.columns.tolist()
        }
    
    def save_preprocessors(self, output_dir='./models/preprocessors/'):
        """Save all preprocessors for later use"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save scalers and encoders
        joblib.dump(self.scaler, Path(output_dir) / 'scaler.pkl')
        joblib.dump(self.label_encoders, Path(output_dir) / 'label_encoders.pkl')
        joblib.dump(self.imputer_numeric, Path(output_dir) / 'imputer_numeric.pkl')
        joblib.dump(self.imputer_categorical, Path(output_dir) / 'imputer_categorical.pkl')
        
        if self.feature_selector:
            joblib.dump(self.feature_selector, Path(output_dir) / 'feature_selector.pkl')
        
        print(f"Preprocessors saved to: {output_dir}")
    
    def load_preprocessors(self, input_dir='./models/preprocessors/'):
        """Load preprocessors from saved files"""
        self.scaler = joblib.load(Path(input_dir) / 'scaler.pkl')
        self.label_encoders = joblib.load(Path(input_dir) / 'label_encoders.pkl')
        self.imputer_numeric = joblib.load(Path(input_dir) / 'imputer_numeric.pkl')
        self.imputer_categorical = joblib.load(Path(input_dir) / 'imputer_categorical.pkl')
        
        feature_selector_path = Path(input_dir) / 'feature_selector.pkl'
        if feature_selector_path.exists():
            self.feature_selector = joblib.load(feature_selector_path)
        
        print(f"Preprocessors loaded from: {input_dir}")

if __name__ == "__main__":
    # Example usage
    preprocessor = DataPreprocessor()
    
    # Load and preprocess data
    df = preprocessor.load_data()
    processed_data = preprocessor.preprocess_data(df)
    
    # Save preprocessors
    preprocessor.save_preprocessors()
    
    print("Preprocessing completed successfully!")
