"""
Main Training Pipeline for EdTech Student Churn Prediction
Orchestrates data generation, preprocessing, model training, and evaluation
"""

import os
import sys
from pathlib import Path
import argparse
import yaml
import logging
from datetime import datetime

# Add src directory to path
sys.path.append(str(Path(__file__).parent / 'src'))

# We're trying to access the classes from these respective python files as a package/library
from src.data_generator import EdTechDataGenerator
from src.data_preprocessing import DataPreprocessor
from src.model_trainer import ModelTrainer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TrainingPipeline:
    def __init__(self, config_path='./config/config.yaml'):
        self.config_path = config_path
        self.load_config()
        self.setup_directories()
        
    def load_config(self):
        """Load configuration file"""
        with open(self.config_path, 'r') as file:
            self.config = yaml.safe_load(file, Loader=yaml.UnsafeLoader)
        logger.info(f"Configuration loaded from: {self.config_path}")
    
    def setup_directories(self):
        """Create necessary directories"""
        directories = [
            './data/raw',
            './data/processed',
            './models',
            './models/preprocessors',
            './mlruns',
            './logs'
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
        
        logger.info("Directory structure created")
    
    def generate_data(self, n_samples=15000):
        """Generate synthetic EdTech student data"""
        logger.info("Starting data generation...")
        
        generator = EdTechDataGenerator(
            n_samples=n_samples, 
            random_state=self.config['data']['random_state']
        )
        
        df = generator.save_data(output_dir=self.config['data']['raw_data_path'])
        
        logger.info(f"Data generation completed. Generated {len(df)} samples")
        return df
    
    def preprocess_data(self):
        """Preprocess the raw data"""
        logger.info("Starting data preprocessing...")
        
        preprocessor = DataPreprocessor(config_path=self.config_path)
        
        # Load raw data
        df = preprocessor.load_data()
        
        # Preprocess data
        processed_data = preprocessor.preprocess_data(
            df, 
            target_col=self.config['model']['target_column'],
            test_size=1 - self.config['data']['train_test_split_ratio'],
            apply_smote=True,
            feature_selection=self.config['model']['feature_engineering']['feature_selection'],
            n_features=20
        )
        
        # Save preprocessors
        preprocessor.save_preprocessors()
        
        logger.info("Data preprocessing completed")
        return processed_data, preprocessor
    
    def train_models(self, processed_data):
        """Train all configured models"""
        logger.info("Starting model training...")
        
        trainer = ModelTrainer(config_path=self.config_path)
        
        # Extract data
        X_train = processed_data['X_train']
        X_test = processed_data['X_test']
        y_train = processed_data['y_train']
        y_test = processed_data['y_test']
        
        # Train all models
        results_df = trainer.train_all_models(X_train, y_train, X_test, y_test)
        
        # Save best model
        trainer.save_best_model()
        
        # Generate comparison report
        trainer.generate_model_comparison_report()
        
        logger.info("Model training completed")
        return trainer, results_df
    
    def run_full_pipeline(self, generate_new_data=True, n_samples=15000):
        """Run the complete training pipeline"""
        logger.info("="*60)
        logger.info("STARTING EDTECH CHURN PREDICTION TRAINING PIPELINE")
        logger.info("="*60)
        
        start_time = datetime.now()
        
        try:
            # Step 1: Generate data (optional)
            if generate_new_data:
                logger.info("STEP 1: Generating synthetic data...")
                self.generate_data(n_samples=n_samples)
            else:
                logger.info("STEP 1: Skipping data generation (using existing data)")
            
            # Step 2: Preprocess data
            logger.info("STEP 2: Preprocessing data...")
            processed_data, preprocessor = self.preprocess_data()
            
            # Step 3: Train models
            logger.info("STEP 3: Training models...")
            trainer, results_df = self.train_models(processed_data)
            
            # Step 4: Summary
            end_time = datetime.now()
            duration = end_time - start_time
            
            logger.info("="*60)
            logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("="*60)
            logger.info(f"Total duration: {duration}")
            logger.info(f"Best model: {trainer.best_model['name']}")
            logger.info(f"Best F1-Score: {trainer.best_score:.4f}")
            
            # Print final results
            print("\\n" + "="*60)
            print("FINAL RESULTS")
            print("="*60)
            print(results_df)
            
            return {
                'trainer': trainer,
                'preprocessor': preprocessor,
                'results': results_df,
                'processed_data': processed_data
            }
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise e

def main():
    """Main function with command line arguments"""
    parser = argparse.ArgumentParser(description='EdTech Churn Prediction Training Pipeline')
    
    parser.add_argument('--config', type=str, default='./config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--generate-data', action='store_true',
                       help='Generate new synthetic data')
    parser.add_argument('--samples', type=int, default=15000,
                       help='Number of samples to generate')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip model training (only preprocess data)')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = TrainingPipeline(config_path=args.config)
    
    if args.skip_training:
        # Only run preprocessing
        logger.info("Running preprocessing only...")
        processed_data, preprocessor = pipeline.preprocess_data()
        logger.info("Preprocessing completed!")
    else:
        # Run full pipeline
        results = pipeline.run_full_pipeline(
            generate_new_data=args.generate_data,
            n_samples=args.samples
        )
        
        print("\\nPipeline completed successfully!")
        print("You can now run the Streamlit app with: streamlit run src/streamlit_app.py")

if __name__ == "__main__":
    main()
