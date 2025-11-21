"""
Model Training Module for EdTech Student Churn Prediction
Includes multiple algorithms, hyperparameter tuning, and MLflow tracking
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import optuna
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
import mlflow.catboost
from pathlib import Path
import yaml
import joblib
import warnings
warnings.filterwarnings('ignore')

class ModelTrainer:
    def __init__(self, config_path='./config/config.yaml'):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        # Initialize MLflow
        mlflow.set_tracking_uri(self.config['mlflow']['tracking_uri'])
        mlflow.set_experiment(self.config['mlflow']['experiment_name'])
        
        self.models = {}
        self.best_model = None
        self.best_score = 0
        
    def get_model(self, model_name, params=None):
        """Get model instance with parameters"""
        if params is None:
            params = {}
            
        if model_name == 'logistic_regression':
            return LogisticRegression(
                random_state=self.config['data']['random_state'],
                max_iter=1000,
                **params
            )
        elif model_name == 'random_forest':
            return RandomForestClassifier(
                random_state=self.config['data']['random_state'],
                n_jobs=-1,
                **params
            )
        elif model_name == 'xgboost':
            return xgb.XGBClassifier(
                random_state=self.config['data']['random_state'],
                eval_metric='logloss',
                **params
            )
        elif model_name == 'lightgbm':
            return lgb.LGBMClassifier(
                random_state=self.config['data']['random_state'],
                verbose=-1,
                **params
            )
        elif model_name == 'catboost':
            return CatBoostClassifier(
                random_state=self.config['data']['random_state'],
                verbose=False,
                **params
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    def get_default_params(self, model_name):
        """Get default hyperparameters for each model"""
        params = {
            'logistic_regression': {
                'C': [0.1, 1.0, 10.0],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            },
            'random_forest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'xgboost': {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 6, 10],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            },
            'lightgbm': {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 6, 10],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            },
            'catboost': {
                'iterations': [100, 200, 300],
                'depth': [3, 6, 10],
                'learning_rate': [0.01, 0.1, 0.2],
                'l2_leaf_reg': [1, 3, 5, 7, 9]
            }
        }
        return params.get(model_name, {})
    
    def objective_function(self, trial, model_name, X_train, y_train, X_val, y_val):
        """Objective function for Optuna hyperparameter optimization"""
        
        if model_name == 'logistic_regression':
            params = {
                'C': trial.suggest_float('C', 0.01, 100.0, log=True),
                'penalty': trial.suggest_categorical('penalty', ['l1', 'l2']),
                'solver': trial.suggest_categorical('solver', ['liblinear', 'saga'])
            }
            
        elif model_name == 'random_forest':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10)
            }
            
        elif model_name == 'xgboost':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10)
            }
            
        elif model_name == 'lightgbm':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10)
            }
            
        elif model_name == 'catboost':
            params = {
                'iterations': trial.suggest_int('iterations', 50, 500),
                'depth': trial.suggest_int('depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
                'border_count': trial.suggest_int('border_count', 32, 255)
            }
        
        # Train model
        model = self.get_model(model_name, params)
        model.fit(X_train, y_train)
        
        # Predict and evaluate
        y_pred = model.predict(X_val)
        f1 = f1_score(y_val, y_pred, average='weighted')
        
        return f1
    
    def train_single_model(self, model_name, X_train, y_train, X_test, y_test, 
                          optimize_hyperparams=True):
        """Train a single model with optional hyperparameter optimization"""
        
        print(f"\n{'='*50}")
        print(f"Training {model_name.upper()}")
        print(f"{'='*50}")
        
        with mlflow.start_run(run_name=f"{model_name}_training"):
            # Log basic info
            mlflow.log_param("model_type", model_name)
            mlflow.log_param("train_samples", len(X_train))
            mlflow.log_param("test_samples", len(X_test))
            mlflow.log_param("features", len(X_train.columns))
            
            if optimize_hyperparams and self.config['model']['hyperparameter_tuning']['enable']:
                print("Optimizing hyperparameters with Optuna...")
                
                # Split training data for validation during optimization
                split_idx = int(0.8 * len(X_train))
                X_train_opt = X_train.iloc[:split_idx]
                y_train_opt = y_train.iloc[:split_idx]
                X_val_opt = X_train.iloc[split_idx:]
                y_val_opt = y_train.iloc[split_idx:]
                
                # Create Optuna study
                study = optuna.create_study(direction='maximize')
                study.optimize(
                    lambda trial: self.objective_function(
                        trial, model_name, X_train_opt, y_train_opt, X_val_opt, y_val_opt
                    ),
                    n_trials=self.config['model']['hyperparameter_tuning']['n_trials']
                )
                
                best_params = study.best_params
                print(f"Best parameters: {best_params}")
                
                # Log hyperparameters
                for key, value in best_params.items():
                    mlflow.log_param(f"best_{key}", value)
                
                # Train final model with best parameters
                model = self.get_model(model_name, best_params)
                
            else:
                print("Using default parameters...")
                model = self.get_model(model_name)
            
            # Train the model
            print("Training final model...")
            model.fit(X_train, y_train)
            
            # Make predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # Calculate metrics
            train_metrics = self.calculate_metrics(y_train, y_train_pred, model, X_train)
            test_metrics = self.calculate_metrics(y_test, y_test_pred, model, X_test)
            
            # Log metrics
            for metric_name, value in train_metrics.items():
                mlflow.log_metric(f"train_{metric_name}", value)
            
            for metric_name, value in test_metrics.items():
                mlflow.log_metric(f"test_{metric_name}", value)
            
            # Log model
            if model_name == 'xgboost':
                mlflow.xgboost.log_model(model, "model")
            elif model_name == 'lightgbm':
                mlflow.lightgbm.log_model(model, "model")
            elif model_name == 'catboost':
                mlflow.catboost.log_model(model, "model")
            else:
                mlflow.sklearn.log_model(model, "model")
            
            # Print results
            print(f"\\nTraining Results for {model_name}:")
            print(f"Train F1-Score: {train_metrics['f1_score']:.4f}")
            print(f"Test F1-Score: {test_metrics['f1_score']:.4f}")
            print(f"Test ROC-AUC: {test_metrics['roc_auc']:.4f}")
            
            # Update best model if this one is better
            if test_metrics['f1_score'] > self.best_score:
                self.best_score = test_metrics['f1_score']
                self.best_model = {
                    'name': model_name,
                    'model': model,
                    'metrics': test_metrics,
                    'run_id': mlflow.active_run().info.run_id
                }
                print(f"*** New best model: {model_name} with F1-Score: {self.best_score:.4f} ***")
            
            # Store model
            self.models[model_name] = {
                'model': model,
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'run_id': mlflow.active_run().info.run_id
            }
            
        return model
    
    def calculate_metrics(self, y_true, y_pred, model, X):
        """Calculate comprehensive evaluation metrics"""
        
        # Get prediction probabilities for ROC-AUC
        try:
            y_pred_proba = model.predict_proba(X)[:, 1]
        except:
            y_pred_proba = y_pred
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted'),
            'roc_auc': roc_auc_score(y_true, y_pred_proba) if len(np.unique(y_true)) > 1 else 0.0
        }
        
        return metrics
    
    def train_all_models(self, X_train, y_train, X_test, y_test):
        """Train all configured models"""
        
        print("Starting model training pipeline...")
        print(f"Training {len(self.config['model']['models_to_train'])} models")
        
        models_to_train = self.config['model']['models_to_train']
        
        for model_name in models_to_train:
            try:
                self.train_single_model(model_name, X_train, y_train, X_test, y_test)
            except Exception as e:
                print(f"Error training {model_name}: {str(e)}")
                continue
        
        # Print summary
        print(f"\\n{'='*60}")
        print("MODEL TRAINING SUMMARY")
        print(f"{'='*60}")
        
        results_df = pd.DataFrame([
            {
                'Model': name,
                'Train F1': data['train_metrics']['f1_score'],
                'Test F1': data['test_metrics']['f1_score'],
                'Test ROC-AUC': data['test_metrics']['roc_auc'],
                'MLflow Run ID': data['run_id']
            }
            for name, data in self.models.items()
        ])
        
        print(results_df.round(4))
        
        if self.best_model:
            print(f"\\nBest Model: {self.best_model['name']} with Test F1-Score: {self.best_score:.4f}")
        
        return results_df
    
    def save_best_model(self, output_dir='./models/'):
        """Save the best performing model"""
        if not self.best_model:
            print("No best model found. Train models first.")
            return
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = Path(output_dir) / 'best_model.pkl'
        joblib.dump(self.best_model['model'], model_path)
        
        # Save model info
        model_info = {
            'model_name': self.best_model['name'],
            'metrics': self.best_model['metrics'],
            'mlflow_run_id': self.best_model['run_id'],
            'model_path': str(model_path)
        }
        
        info_path = Path(output_dir) / 'best_model_info.yaml'
        with open(info_path, 'w') as file:
            yaml.dump(model_info, file)
        
        print(f"Best model saved to: {model_path}")
        print(f"Model info saved to: {info_path}")
    
    def load_best_model(self, model_dir='./models/'):
        """Load the best saved model"""
        model_path = Path(model_dir) / 'best_model.pkl'
        info_path = Path(model_dir) / 'best_model_info.yaml'
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model = joblib.load(model_path)
        
        if info_path.exists():
            with open(info_path, 'r') as file:
                model_info = yaml.safe_load(file)
            print(f"Loaded model: {model_info['model_name']}")
            print(f"Test F1-Score: {model_info['metrics']['f1_score']:.4f}")
        
        return model
    
    def generate_model_comparison_report(self):
        """Generate a comprehensive model comparison report"""
        if not self.models:
            print("No models trained yet.")
            return None
        
        # Create comparison DataFrame
        comparison_data = []
        for name, data in self.models.items():
            comparison_data.append({
                'Model': name,
                'Train Accuracy': data['train_metrics']['accuracy'],
                'Test Accuracy': data['test_metrics']['accuracy'],
                'Train Precision': data['train_metrics']['precision'],
                'Test Precision': data['test_metrics']['precision'],
                'Train Recall': data['train_metrics']['recall'],
                'Test Recall': data['test_metrics']['recall'],
                'Train F1': data['train_metrics']['f1_score'],
                'Test F1': data['test_metrics']['f1_score'],
                'Test ROC-AUC': data['test_metrics']['roc_auc'],
                'Overfitting': abs(data['train_metrics']['f1_score'] - data['test_metrics']['f1_score'])
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.round(4)
        
        # Save report
        report_path = Path('./models/model_comparison_report.csv')
        comparison_df.to_csv(report_path, index=False)
        print(f"Model comparison report saved to: {report_path}")
        
        return comparison_df

if __name__ == "__main__":
    # Example usage - this would normally be called from main training script
    print("Model Trainer module loaded successfully!")
    print("Use this module with preprocessed data to train models.")
