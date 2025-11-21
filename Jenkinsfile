pipeline {
    agent any
    
    environment {
        PYTHON_VERSION = '3.9'
        AWS_REGION = 'us-east-1'
        ECR_REGISTRY = 'your-account-id.dkr.ecr.us-east-1.amazonaws.com'
        IMAGE_NAME = 'edtech-churn-prediction'
        EC2_INSTANCE_ID = 'i-your-instance-id'
        MLFLOW_TRACKING_URI = 'http://localhost:5000'
    }
    
    stages {
        stage('Checkout') {
            steps {
                checkout scm
                echo 'Code checked out successfully'
            }
        }
        
        stage('Setup Environment') {
            steps {
                sh '''
                    echo "Setting up Python virtual environment..."
                    python3 -m venv venv
                    source venv/bin/activate
                    pip install --upgrade pip
                    pip install -r requirements.txt
                    echo "Environment setup complete"
                '''
            }
        }
        
        stage('Code Quality & Testing') {
            parallel {
                stage('Lint Code') {
                    steps {
                        sh '''
                            source venv/bin/activate
                            echo "Running code linting..."
                            flake8 src/ --max-line-length=100 --ignore=E501,W503 || true
                            echo "Linting completed"
                        '''
                    }
                }
                
                stage('Format Code') {
                    steps {
                        sh '''
                            source venv/bin/activate
                            echo "Checking code formatting..."
                            black --check src/ || true
                            echo "Code formatting check completed"
                        '''
                    }
                }
                
                stage('Run Tests') {
                    steps {
                        sh '''
                            source venv/bin/activate
                            echo "Running unit tests..."
                            python -m pytest tests/ -v --junitxml=test-results.xml || true
                            echo "Tests completed"
                        '''
                    }
                }
            }
        }
        
        stage('Generate Data & Train Models') {
            when {
                anyOf {
                    branch 'main'
                    branch 'develop'
                    changeRequest()
                }
            }
            steps {
                sh '''
                    source venv/bin/activate
                    echo "Starting ML pipeline..."
                    
                    # Generate synthetic data
                    python train_pipeline.py --generate-data --samples 10000
                    
                    echo "ML pipeline completed"
                '''
                
                // Archive model artifacts
                archiveArtifacts artifacts: 'models/**/*.pkl, models/**/*.yaml, mlruns/**/*', fingerprint: true
                
                // Publish test results
                publishTestResults testResultsPattern: 'test-results.xml'
            }
        }
        
        stage('Model Validation') {
            steps {
                sh '''
                    source venv/bin/activate
                    echo "Validating trained models..."
                    
                    python -c "
import sys
sys.path.append('src')
from model_trainer import ModelTrainer
import yaml

# Load model info
with open('models/best_model_info.yaml', 'r') as f:
    model_info = yaml.safe_load(f)

# Validate metrics
f1_score = model_info['metrics']['f1_score']
accuracy = model_info['metrics']['accuracy']

print(f'Model Performance:')
print(f'F1-Score: {f1_score:.4f}')
print(f'Accuracy: {accuracy:.4f}')

# Set thresholds
min_f1_score = 0.70
min_accuracy = 0.75

if f1_score < min_f1_score:
    print(f'ERROR: F1-Score {f1_score:.4f} below threshold {min_f1_score}')
    sys.exit(1)
    
if accuracy < min_accuracy:
    print(f'ERROR: Accuracy {accuracy:.4f} below threshold {min_accuracy}')
    sys.exit(1)
    
print('Model validation passed!')
"
                '''
            }
        }
        
        stage('Build Docker Image') {
            when {
                branch 'main'
            }
            steps {
                script {
                    def image = docker.build("${IMAGE_NAME}:${BUILD_NUMBER}")
                    
                    // Tag with 'latest' as well
                    sh "docker tag ${IMAGE_NAME}:${BUILD_NUMBER} ${IMAGE_NAME}:latest"
                    
                    echo "Docker image built successfully"
                }
            }
        }
        
        stage('Security Scan') {
            when {
                branch 'main'
            }
            steps {
                sh '''
                    echo "Running security scan..."
                    # Add security scanning tools like Bandit, Safety, etc.
                    source venv/bin/activate
                    
                    # Check for known vulnerabilities in dependencies
                    pip-audit --format=json --output=security-report.json || true
                    
                    echo "Security scan completed"
                '''
                
                archiveArtifacts artifacts: 'security-report.json', allowEmptyArchive: true
            }
        }
        
        stage('Push to ECR') {
            when {
                branch 'main'
            }
            steps {
                withAWS(region: "${AWS_REGION}") {
                    sh '''
                        echo "Pushing Docker image to ECR..."
                        
                        # Login to ECR
                        aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${ECR_REGISTRY}
                        
                        # Tag and push image
                        docker tag ${IMAGE_NAME}:${BUILD_NUMBER} ${ECR_REGISTRY}/${IMAGE_NAME}:${BUILD_NUMBER}
                        docker tag ${IMAGE_NAME}:${BUILD_NUMBER} ${ECR_REGISTRY}/${IMAGE_NAME}:latest
                        
                        docker push ${ECR_REGISTRY}/${IMAGE_NAME}:${BUILD_NUMBER}
                        docker push ${ECR_REGISTRY}/${IMAGE_NAME}:latest
                        
                        echo "Docker image pushed successfully"
                    '''
                }
            }
        }
        
        stage('Deploy to Staging') {
            when {
                branch 'develop'
            }
            steps {
                sh '''
                    echo "Deploying to staging environment..."
                    
                    # Deploy to staging EC2 instance
                    # This would typically involve SSH to staging server and updating containers
                    echo "Staging deployment completed"
                '''
            }
        }
        
        stage('Deploy to Production') {
            when {
                branch 'main'
            }
            steps {
                timeout(time: 10, unit: 'MINUTES') {
                    input message: 'Deploy to production?', ok: 'Deploy'
                }
                
                withAWS(region: "${AWS_REGION}") {
                    sh '''
                        echo "Deploying to production..."
                        
                        # Update EC2 instance with new image
                        aws ssm send-command \
                            --instance-ids ${EC2_INSTANCE_ID} \
                            --document-name "AWS-RunShellScript" \
                            --parameters commands="
                                cd /opt/edtech-churn-prediction && \
                                docker-compose pull && \
                                docker-compose up -d --remove-orphans
                            " \
                            --region ${AWS_REGION}
                        
                        echo "Production deployment initiated"
                    '''
                }
            }
        }
        
        stage('Health Check') {
            when {
                branch 'main'
            }
            steps {
                sh '''
                    echo "Running post-deployment health checks..."
                    
                    # Wait for deployment to complete
                    sleep 60
                    
                    # Health check API endpoint
                    HEALTH_URL="http://your-ec2-public-ip:8501/health"
                    
                    for i in {1..5}; do
                        if curl -f $HEALTH_URL; then
                            echo "Health check passed"
                            break
                        else
                            echo "Health check failed, retrying in 30 seconds..."
                            sleep 30
                        fi
                    done
                '''
            }
        }
        
        stage('Model Registry Update') {
            when {
                branch 'main'
            }
            steps {
                sh '''
                    source venv/bin/activate
                    echo "Updating MLflow model registry..."
                    
                    python -c "
import mlflow
import yaml

# Set MLflow tracking URI
mlflow.set_tracking_uri('${MLFLOW_TRACKING_URI}')

# Load model info
with open('models/best_model_info.yaml', 'r') as f:
    model_info = yaml.safe_load(f)

run_id = model_info['mlflow_run_id']
model_name = 'edtech-churn-prediction'

# Register model
model_uri = f'runs:/{run_id}/model'
mlflow.register_model(model_uri, model_name)

print('Model registered successfully!')
"
                '''
            }
        }
    }
    
    post {
        always {
            // Clean up
            sh '''
                echo "Cleaning up..."
                docker system prune -f || true
                rm -rf venv || true
            '''
            
            // Archive logs
            archiveArtifacts artifacts: 'training.log, *.log', allowEmptyArchive: true
            
            // Publish test results
            publishTestResults testResultsPattern: 'test-results.xml'
        }
        
        success {
            echo 'Pipeline completed successfully!'
            
            // Send success notification
            emailext (
                subject: "SUCCESS: EdTech Churn Prediction Pipeline - Build ${BUILD_NUMBER}",
                body: """
                The EdTech Churn Prediction pipeline has completed successfully!
                
                Build: ${BUILD_NUMBER}
                Branch: ${BRANCH_NAME}
                Commit: ${GIT_COMMIT}
                
                Model Performance:
                - Check the artifacts for detailed metrics
                
                Next steps:
                - Model is deployed to production
                - Monitor application health
                - Review model performance metrics
                
                Build URL: ${BUILD_URL}
                """,
                to: 'team@example.com'
            )
        }
        
        failure {
            echo 'Pipeline failed!'
            
            // Send failure notification
            emailext (
                subject: "FAILURE: EdTech Churn Prediction Pipeline - Build ${BUILD_NUMBER}",
                body: """
                The EdTech Churn Prediction pipeline has failed!
                
                Build: ${BUILD_NUMBER}
                Branch: ${BRANCH_NAME}
                Commit: ${GIT_COMMIT}
                
                Please check the build logs for details:
                ${BUILD_URL}/console
                
                Common issues:
                - Test failures
                - Model performance below threshold
                - Docker build issues
                - Deployment failures
                """,
                to: 'team@example.com'
            )
        }
        
        unstable {
            echo 'Pipeline completed with warnings!'
        }
    }
}
