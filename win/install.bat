@echo off
REM install.bat - Windows installation script
REM Usage: double-click this file

echo ========================================
echo The Street - Reckoning
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed. Please install Python 3.8 or higher.
    echo Visit: https://www.python.org/downloads/
    pause
    exit /b 1
)

REM Check Python version
for /f "tokens=2" %%I in ('python --version 2^>^&1') do set PYTHON_VERSION=%%I
echo ✓ Python %PYTHON_VERSION% detected
echo.

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv

if not exist "venv\Scripts\activate.bat" (
    echo ❌ Failed to create virtual environment
    pause
    exit /b 1
)

echo ✓ Virtual environment created at: %CD%\venv
echo.

REM Activate virtual environment and install requirements
echo Installing dependencies...
call venv\Scripts\activate.bat
python -m pip install --upgrade pip

if exist "requirements.txt" (
    pip install -r requirements.txt
) else (
    echo Flask>=2.0.0 > requirements.txt
    pip install Flask
)

echo ✓ Dependencies installed
echo.

REM Create necessary directories
echo Creating necessary directories...
mkdir uploads 2>nul
mkdir logs 2>nul
mkdir exports 2>nul
mkdir backups 2>nul
echo ✓ Directories created
echo.

REM Create a default .env file if it doesn't exist
if not exist ".env" (
    echo Creating .env file...
    (
        echo # Application Settings
        echo FLASK_APP=app.py
        echo FLASK_ENV=development
        echo SECRET_KEY=your-secret-key-change-this-in-production
        echo MAX_CONTENT_LENGTH=16777216
        echo.
        echo # Security Settings
        echo SESSION_TIMEOUT=9999
        echo MAX_EXECUTION_TIME=99999
        echo LOG_LEVEL=INFO
        echo.
        echo # Paths (relative to project root)
        echo UPLOAD_FOLDER=uploads
        echo BACKUP_FOLDER=backups
        echo EXPORT_FOLDER=exports
        echo LOG_FOLDER=logs
    ) > .env
    echo ✓ Created .env file
    echo.
    echo ⚠ Please edit .env and change SECRET_KEY!
)

echo ========================================
echo ✅ Installation complete!
echo ========================================
echo.
echo To start the application:
echo   1. Double-click start_windows.bat
echo   2. Or run in Command Prompt: start_windows.bat
echo.
echo The application will be available at:
echo   http://localhost:5000
echo.
echo Default credentials (if applicable):
echo   Username: admin
echo   Password: admin
echo.
echo ⚠  Please change the default credentials in production!
echo.
pause