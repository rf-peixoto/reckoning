@echo off
REM start_windows.bat - Windows start script for Red Team Workflow Automation Platform
REM Usage: double-click this file or run from command prompt

echo ========================================
echo Starting The Street - Reckoning
echo ========================================
echo.

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo ❌ Virtual environment not found!
    echo Please run install.bat or install.sh first
    pause
    exit /b 1
)

REM Check if app.py exists
if not exist "app.py" (
    echo ❌ app.py not found!
    echo Please make sure you're in the correct directory
    pause
    exit /b 1
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Check Python version
python --version
echo.

REM Check if Flask is installed
python -c "import flask" >nul 2>&1
if errorlevel 1 (
    echo ❌ Flask not installed in virtual environment
    echo Please run install.bat again
    pause
    exit /b 1
)

REM Create necessary directories if they don't exist
mkdir uploads 2>nul
mkdir logs 2>nul
mkdir exports 2>nul
mkdir backups 2>nul

REM Set default values
set PORT=5000
set HOST=0.0.0.0

REM Check for command line arguments
if not "%1"=="" set PORT=%1
if not "%2"=="" set HOST=%2

REM Set Flask environment variables
set FLASK_APP=app.py
set FLASK_ENV=development

echo.
echo Application Information:
echo   Host: %HOST%
echo   Port: %PORT%
echo   Environment: %FLASK_ENV%
echo.
echo Directories:
echo   Uploads: %CD%\uploads
echo   Logs: %CD%\logs
echo   Exports: %CD%\exports
echo   Backups: %CD%\backups
echo.
echo ========================================
echo 🚀 Starting Flask application...
echo Press Ctrl+C to stop
echo ========================================
echo.
echo Access the application at:
echo   http://localhost:%PORT%
echo.
echo Or on your network at:
echo   http://%COMPUTERNAME%:%PORT%
echo.

REM Start the Flask application
if "%FLASK_ENV%"=="production" (
    echo Starting in PRODUCTION mode...
    python -c "import waitress; from app import app; waitress.serve(app, host='%HOST%', port=%PORT%)"
) else (
    echo Starting in DEVELOPMENT mode...
    flask run --host=%HOST% --port=%PORT% --debug
)

pause