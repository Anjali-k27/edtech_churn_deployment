@echo off
echo =========================================
echo EdTech Churn Prediction System Setup
echo =========================================

echo.
echo Creating virtual environment...
python -m venv venv

echo.
echo Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Installing dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt

echo.
echo Setup complete!
echo.
echo To run the project:
echo 1. Activate environment: venv\Scripts\activate
echo 2. Generate data and train: python train_pipeline.py --generate-data --samples 5000
echo 3. Run Streamlit: streamlit run src/streamlit_app.py
echo.
echo Or use: python quick_start.py
echo.
pause
