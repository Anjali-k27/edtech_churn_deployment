#!/usr/bin/env python3
"""
Quick Start Script for EdTech Churn Prediction System
Runs the complete pipeline from data generation to model training and Streamlit app
"""

import os
import sys
import subprocess
import time
import argparse
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{'='*50}")
    print(f"🔄 {description}")
    print(f"{'='*50}")
    print(f"Running: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stderr:
            print(f"Error details: {e.stderr}")
        return False

def check_requirements():
    """Check if required software is installed"""
    print("🔍 Checking requirements...")
    
    requirements = {
        'python': 'python --version',
        'pip': 'pip --version',
        'git': 'git --version'
    }
    
    missing = []
    for name, command in requirements.items():
        try:
            subprocess.run(command, shell=True, check=True, capture_output=True)
            print(f"✅ {name} is installed")
        except subprocess.CalledProcessError:
            print(f"❌ {name} is not installed or not in PATH")
            missing.append(name)
    
    if missing:
        print(f"\nPlease install missing requirements: {', '.join(missing)}")
        return False
    
    return True

def setup_environment():
    """Set up Python virtual environment and install dependencies"""
    print("\n🐍 Setting up Python environment...")
    
    # Check if virtual environment exists
    venv_path = Path("venv")
    if not venv_path.exists():
        print("Creating virtual environment...")
        if not run_command("python -m venv venv", "Creating virtual environment"):
            return False
    
    # Determine activation command based on OS
    if os.name == 'nt':  # Windows
        activate_cmd = "venv\\Scripts\\activate"
        pip_cmd = "venv\\Scripts\\pip"
        python_cmd = "venv\\Scripts\\python"
    else:  # Unix/Linux/MacOS
        activate_cmd = "source venv/bin/activate"
        pip_cmd = "venv/bin/pip"
        python_cmd = "venv/bin/python"
    
    # Install dependencies
    print("Installing Python dependencies...")
    if not run_command(f"{pip_cmd} install --upgrade pip", "Upgrading pip"):
        return False
    
    if not run_command(f"{pip_cmd} install -r requirements.txt", "Installing dependencies"):
        return False
    
    return True, python_cmd

def generate_data_and_train(python_cmd, samples=10000):
    """Generate synthetic data and train models"""
    print(f"\n📊 Generating {samples} samples of synthetic data and training models...")
    
    # Run the training pipeline
    command = f"{python_cmd} train_pipeline.py --generate-data --samples {samples}"
    return run_command(command, f"Generating data and training models ({samples} samples)")

def start_streamlit(python_cmd, background=False):
    """Start the Streamlit application"""
    print("\n🚀 Starting Streamlit application...")
    
    command = f"{python_cmd} -m streamlit run src/streamlit_app.py --server.port=8501"
    
    if background:
        # Start in background (non-blocking)
        try:
            subprocess.Popen(command, shell=True)
            time.sleep(5)  # Give it time to start
            print("✅ Streamlit app started in background")
            print("🌐 Access the app at: http://localhost:8501")
            return True
        except Exception as e:
            print(f"❌ Failed to start Streamlit: {e}")
            return False
    else:
        # Start in foreground (blocking)
        print("🌐 Starting Streamlit app at: http://localhost:8501")
        print("Press Ctrl+C to stop the application")
        return run_command(command, "Starting Streamlit application")

def run_docker_setup():
    """Set up and run Docker containers"""
    print("\n🐳 Setting up Docker environment...")
    
    # Check if Docker is installed
    try:
        subprocess.run("docker --version", shell=True, check=True, capture_output=True)
        print("✅ Docker is installed")
    except subprocess.CalledProcessError:
        print("❌ Docker is not installed. Please install Docker first.")
        return False
    
    # Build Docker image
    if not run_command("docker build -t edtech-churn-prediction .", "Building Docker image"):
        return False
    
    # Run with Docker Compose
    if not run_command("docker-compose up -d", "Starting Docker containers"):
        return False
    
    print("✅ Docker containers are running!")
    print("🌐 Streamlit app: http://localhost:8501")
    print("📊 MLflow UI: http://localhost:5000")
    print("🔧 Nginx: http://localhost:80")
    
    return True

def cleanup():
    """Clean up temporary files"""
    print("\n🧹 Cleaning up...")
    
    # Remove cache files
    cache_dirs = [
        "src/__pycache__",
        "tests/__pycache__",
        ".pytest_cache"
    ]
    
    for cache_dir in cache_dirs:
        if Path(cache_dir).exists():
            run_command(f"rm -rf {cache_dir}", f"Removing {cache_dir}")

def print_summary():
    """Print project summary and next steps"""
    print("\n" + "="*60)
    print("🎉 EDTECH CHURN PREDICTION SYSTEM - QUICK START COMPLETE")
    print("="*60)
    
    print("""
📊 What was accomplished:
✅ Environment setup and dependencies installed
✅ Synthetic EdTech student data generated
✅ Multiple ML models trained and evaluated
✅ Best model selected and saved
✅ Streamlit web application ready

🌐 Access the Application:
• Streamlit App: http://localhost:8501
• MLflow UI: http://localhost:5000 (if using Docker)

📋 Available Features:
• 🏠 Home Dashboard - Dataset overview and statistics
• 📊 Data Explorer - Interactive data analysis and visualization
• 🔮 Churn Prediction - Individual student risk assessment
• 📈 Model Insights - Model performance and feature importance

🚀 Next Steps:
1. Explore the Streamlit interface
2. Try predicting churn for different student profiles
3. Analyze feature importance and model performance
4. Deploy to AWS using the provided Terraform scripts
5. Set up Jenkins CI/CD pipeline for automated deployments

📖 Documentation:
• README.md - Comprehensive project documentation
• config/config.yaml - Configuration settings
• Dockerfile & docker-compose.yml - Container deployment

🤝 Get Involved:
• Star the project on GitHub
• Report issues and contribute improvements
• Share your experience with the EdTech community

💡 Pro Tips:
• Adjust model parameters in config/config.yaml
• Add new EdTech companies by modifying src/data_generator.py
• Scale up by deploying to AWS EC2 using deployment/terraform/

🎓 Built for improving educational outcomes and student success!
""")

def main():
    parser = argparse.ArgumentParser(description='EdTech Churn Prediction Quick Start')
    parser.add_argument('--samples', type=int, default=10000, help='Number of samples to generate (default: 10000)')
    parser.add_argument('--docker', action='store_true', help='Use Docker setup instead of local Python')
    parser.add_argument('--skip-training', action='store_true', help='Skip model training (useful for testing)')
    parser.add_argument('--background', action='store_true', help='Start Streamlit in background')
    parser.add_argument('--no-cleanup', action='store_true', help='Skip cleanup step')
    
    args = parser.parse_args()
    
    print("🎓 EdTech Churn Prediction System - Quick Start")
    print("=" * 50)
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    try:
        if args.docker:
            # Docker setup path
            print("🐳 Using Docker setup...")
            if not run_docker_setup():
                print("❌ Docker setup failed")
                sys.exit(1)
        else:
            # Local Python setup path
            print("🐍 Using local Python setup...")
            
            # Setup environment
            result = setup_environment()
            if not result:
                print("❌ Environment setup failed")
                sys.exit(1)
            
            success, python_cmd = result
            
            if not args.skip_training:
                # Generate data and train models
                if not generate_data_and_train(python_cmd, args.samples):
                    print("❌ Data generation and training failed")
                    sys.exit(1)
            
            # Start Streamlit app
            if not start_streamlit(python_cmd, args.background):
                print("❌ Failed to start Streamlit app")
                sys.exit(1)
        
        # Cleanup
        if not args.no_cleanup:
            cleanup()
        
        # Print summary
        print_summary()
        
    except KeyboardInterrupt:
        print("\n🛑 Setup interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
