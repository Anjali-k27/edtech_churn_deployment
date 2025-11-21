#!/bin/bash

# EdTech Churn Prediction EC2 User Data Script
# This script sets up the EC2 instance with all necessary dependencies

set -e  # Exit on any error

# Variables
PROJECT_NAME="${project_name}"
APP_DIR="/opt/$PROJECT_NAME"
LOG_FILE="/var/log/user_data.log"

# Logging function
log() {
    echo "$(date): $1" | tee -a $LOG_FILE
}

log "Starting user data script for $PROJECT_NAME"

# Update system
log "Updating system packages..."
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get upgrade -y

# Install essential packages
log "Installing essential packages..."
apt-get install -y \
    curl \
    wget \
    git \
    unzip \
    software-properties-common \
    apt-transport-https \
    ca-certificates \
    gnupg \
    lsb-release \
    build-essential \
    python3-dev \
    python3-pip \
    python3-venv \
    awscli

# Install Docker
log "Installing Docker..."
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null

apt-get update -y
apt-get install -y docker-ce docker-ce-cli containerd.io

# Start and enable Docker
systemctl start docker
systemctl enable docker

# Add ubuntu user to docker group
usermod -aG docker ubuntu

# Install Docker Compose
log "Installing Docker Compose..."
curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

# Install AWS CLI v2
log "Installing AWS CLI v2..."
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
./aws/install
rm -rf awscliv2.zip aws

# Install SSM Agent
log "Installing SSM Agent..."
systemctl enable snap.amazon-ssm-agent.amazon-ssm-agent.service
systemctl start snap.amazon-ssm-agent.amazon-ssm-agent.service

# Create application directory
log "Creating application directory..."
mkdir -p $APP_DIR
cd $APP_DIR

# Create docker-compose.yml for the application
log "Creating Docker Compose configuration..."
cat > docker-compose.yml << 'EOL'
version: '3.8'

services:
  streamlit-app:
    image: ${ECR_REGISTRY}/${PROJECT_NAME}:latest
    ports:
      - "8501:8501"
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./mlruns:/app/mlruns
      - ./logs:/app/logs
    environment:
      - STREAMLIT_SERVER_PORT=8501
      - STREAMLIT_SERVER_ADDRESS=0.0.0.0
      - MLFLOW_TRACKING_URI=http://mlflow:5000
      - AWS_DEFAULT_REGION=us-east-1
    depends_on:
      - mlflow
    restart: unless-stopped
    networks:
      - edtech-network

  mlflow:
    image: python:3.9-slim
    ports:
      - "5000:5000"
    volumes:
      - ./mlruns:/mlruns
    command: >
      bash -c "
        pip install mlflow boto3 psycopg2-binary &&
        mlflow server 
        --backend-store-uri sqlite:///mlruns/mlflow.db 
        --default-artifact-root ./mlruns 
        --host 0.0.0.0 
        --port 5000
      "
    restart: unless-stopped
    networks:
      - edtech-network

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - streamlit-app
    restart: unless-stopped
    networks:
      - edtech-network

networks:
  edtech-network:
    driver: bridge

volumes:
  mlruns:
  data:
  models:
  logs:
EOL

# Create nginx configuration
log "Creating Nginx configuration..."
cat > nginx.conf << 'EOL'
events {
    worker_connections 1024;
}

http {
    upstream streamlit {
        server streamlit-app:8501;
    }

    upstream mlflow {
        server mlflow:5000;
    }

    server {
        listen 80;
        server_name _;

        # Streamlit app
        location / {
            proxy_pass http://streamlit;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_buffering off;
        }

        # MLflow UI
        location /mlflow/ {
            proxy_pass http://mlflow/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        # Health check endpoint
        location /health {
            access_log off;
            return 200 "healthy\n";
            add_header Content-Type text/plain;
        }
    }
}
EOL

# Create data directories
log "Creating data directories..."
mkdir -p data/raw data/processed models/preprocessors mlruns logs

# Set proper permissions
chown -R ubuntu:ubuntu $APP_DIR

# Install CloudWatch Agent
log "Installing CloudWatch Agent..."
wget https://s3.amazonaws.com/amazoncloudwatch-agent/ubuntu/amd64/latest/amazon-cloudwatch-agent.deb
dpkg -i -E ./amazon-cloudwatch-agent.deb

# Create CloudWatch Agent configuration
cat > /opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json << 'EOL'
{
    "logs": {
        "logs_collected": {
            "files": {
                "collect_list": [
                    {
                        "file_path": "/var/log/user_data.log",
                        "log_group_name": "/aws/ec2/edtech-churn-prediction",
                        "log_stream_name": "user-data"
                    },
                    {
                        "file_path": "/opt/edtech-churn-prediction/logs/*.log",
                        "log_group_name": "/aws/ec2/edtech-churn-prediction",
                        "log_stream_name": "application"
                    }
                ]
            }
        }
    },
    "metrics": {
        "namespace": "EdTech/ChurnPrediction",
        "metrics_collected": {
            "cpu": {
                "measurement": [
                    "cpu_usage_idle",
                    "cpu_usage_iowait",
                    "cpu_usage_user",
                    "cpu_usage_system"
                ],
                "metrics_collection_interval": 60
            },
            "disk": {
                "measurement": [
                    "used_percent"
                ],
                "metrics_collection_interval": 60,
                "resources": [
                    "*"
                ]
            },
            "mem": {
                "measurement": [
                    "mem_used_percent"
                ],
                "metrics_collection_interval": 60
            }
        }
    }
}
EOL

# Start CloudWatch Agent
/opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl \
    -a fetch-config \
    -m ec2 \
    -c file:/opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json \
    -s

# Create startup script
log "Creating startup script..."
cat > /home/ubuntu/start_application.sh << 'EOL'
#!/bin/bash

# Script to start the EdTech Churn Prediction application

cd /opt/edtech-churn-prediction

# Get AWS account ID for ECR
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
AWS_REGION=$(curl -s http://169.254.169.254/latest/meta-data/placement/region)

# Update environment variables in docker-compose
export ECR_REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
export PROJECT_NAME="edtech-churn-prediction"

# Login to ECR
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $ECR_REGISTRY

# Pull latest image
docker-compose pull

# Start services
docker-compose up -d

echo "Application started successfully!"
echo "Streamlit URL: http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):8501"
echo "MLflow URL: http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):5000"
EOL

chmod +x /home/ubuntu/start_application.sh
chown ubuntu:ubuntu /home/ubuntu/start_application.sh

# Create systemd service for automatic startup
log "Creating systemd service..."
cat > /etc/systemd/system/edtech-churn-prediction.service << 'EOL'
[Unit]
Description=EdTech Churn Prediction Application
After=docker.service
Requires=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
User=ubuntu
Group=ubuntu
WorkingDirectory=/opt/edtech-churn-prediction
ExecStart=/home/ubuntu/start_application.sh
TimeoutStartSec=300

[Install]
WantedBy=multi-user.target
EOL

# Enable the service
systemctl enable edtech-churn-prediction.service

# Install monitoring tools
log "Installing monitoring tools..."
apt-get install -y htop iotop

# Setup log rotation
log "Setting up log rotation..."
cat > /etc/logrotate.d/edtech-churn-prediction << 'EOL'
/opt/edtech-churn-prediction/logs/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    copytruncate
}
EOL

# Create initial sample data and model (optional)
log "Setting up initial application data..."
cd $APP_DIR

# Download a simple Python script to generate initial data
cat > generate_initial_data.py << 'EOL'
#!/usr/bin/env python3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# Create sample data for demonstration
np.random.seed(42)
n_samples = 1000

# Generate sample student data
data = {
    'student_id': [f'STU_{i+1:06d}' for i in range(n_samples)],
    'company': np.random.choice(['Newton School', 'Scaler', 'Upgrad', 'Simplilearn'], n_samples),
    'course': np.random.choice(['Data Science', 'Full Stack Development', 'Machine Learning'], n_samples),
    'age': np.random.normal(26, 4, n_samples).astype(int).clip(18, 45),
    'login_frequency_per_week': np.random.exponential(3, n_samples),
    'assignment_completion_rate': np.random.beta(2, 1, n_samples),
    'quiz_avg_score': np.random.normal(75, 15, n_samples).clip(0, 100),
    'churn': np.random.binomial(1, 0.3, n_samples)
}

df = pd.DataFrame(data)

# Save to CSV
os.makedirs('data/raw', exist_ok=True)
df.to_csv('data/raw/edtech_student_data.csv', index=False)

print(f"Generated {len(df)} sample records")
print(f"Churn rate: {df['churn'].mean():.2%}")
EOL

python3 generate_initial_data.py

# Set final permissions
chown -R ubuntu:ubuntu $APP_DIR

# Final system configuration
log "Final system configuration..."
# Increase open file limits
echo "ubuntu soft nofile 65536" >> /etc/security/limits.conf
echo "ubuntu hard nofile 65536" >> /etc/security/limits.conf

# Configure swap (for memory management)
fallocate -l 2G /swapfile
chmod 600 /swapfile
mkswap /swapfile
swapon /swapfile
echo '/swapfile none swap sw 0 0' | tee -a /etc/fstab

# Final cleanup
apt-get autoremove -y
apt-get autoclean

log "User data script completed successfully!"
log "Instance is ready for the EdTech Churn Prediction application"
log "Next steps:"
log "1. Push Docker image to ECR"
log "2. Run: sudo systemctl start edtech-churn-prediction.service"
log "3. Access application at http://<public-ip>:8501"

# Signal that user data has completed
/opt/aws/bin/cfn-signal -e $? --stack ${AWS::StackName} --resource AutoScalingGroup --region ${AWS::Region} || true

echo "Setup complete!" > /tmp/setup_complete
