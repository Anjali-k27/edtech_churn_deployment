# 🚀 Quick Setup Guide

## **Fastest Way to Get Started**

### **Option 1: Windows Batch Script**
```bash
# Just double-click or run:
install.bat
```

### **Option 2: Manual Setup (3 minutes)**
```bash
# Step 1: Create environment
python -m venv venv

# Step 2: Activate environment
venv\Scripts\activate

# Step 3: Install packages
pip install -r requirements.txt

# Step 4: Generate data and train models (5 minutes)
python train_pipeline.py --generate-data --samples 200

# Step 5: Run the app
streamlit run src/streamlit_app.py
```

### **Option 3: Docker (if you have Docker)**
```bash
docker-compose up -d
```

## **What You'll Get**

✅ **Web Application** at http://localhost:8501
- Interactive dashboard
- Real-time churn predictions
- Data visualization and insights

✅ **ML Models Trained** 
- 5 different algorithms
- Hyperparameter optimization
- Performance comparison

✅ **Sample Data Generated**
- 5,000+ synthetic EdTech student records
- Realistic features and distributions
- Ready for experimentation

## **Troubleshooting**

### **Common Issues:**

1. **Python not found**
   - Install Python 3.9+ from python.org
   - Make sure it's in your PATH

2. **Permission errors**
   - Run Command Prompt as Administrator
   - Or use: `python -m pip install --user -r requirements.txt`

3. **Memory issues**
   - Reduce sample size: `--samples 1000`
   - Close other applications

4. **Streamlit not starting**
   - Check if port 8501 is free
   - Try: `streamlit run src/streamlit_app.py --server.port 8502`

### **Quick Fixes:**
```bash
# If pip upgrade fails:
python -m pip install --upgrade pip --user

# If streamlit fails:
pip install streamlit --upgrade

# If training fails:
python train_pipeline.py --generate-data --samples 1000
```

## **Next Steps**

1. **Explore the App** - Try different student profiles
2. **Analyze Results** - Check model performance and insights
3. **Customize** - Modify configurations in `config/config.yaml`
4. **Deploy** - Use AWS deployment scripts in `deployment/`

## **Support**

- 📖 Full documentation: `README.md`
- 🧪 Run tests: `python -m pytest tests/`
- 🐳 Docker alternative: `docker-compose up`
- ☁️ AWS deployment: `deployment/terraform/`

**Time to complete**: 5-10 minutes
**Requirements**: Python 3.9+, 4GB RAM, 2GB disk space
