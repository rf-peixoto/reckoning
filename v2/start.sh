#!/bin/bash

# start.sh - Start script for Red Team Workflow Automation Platform
# Usage: ./start.sh [port] [host]

set -e  # Exit on any error

# Default values
PORT=${1:-5000}
HOST=${2:-0.0.0.0}

echo "========================================"
echo "Starting The Street - Reckoning"
echo "========================================"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please run ./install.sh first"
    exit 1
fi

# Check if app.py exists
if [ ! -f "app.py" ]; then
    echo "❌ app.py not found!"
    echo "Please make sure you're in the correct directory"
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Check Python version in venv
echo "Python version: $(python --version 2>&1)"
echo ""

# Check if Flask is installed
if ! python -c "import flask" &> /dev/null; then
    echo "❌ Flask not installed in virtual environment"
    echo "Please run ./install.sh again"
    exit 1
fi

# Load environment variables if .env exists
#if [ -f ".env" ]; then
#   echo "Loading environment variables from .env..."
#   # Simple .env loader (doesn't handle multiline or comments perfectly)
#   export $(grep -v '^#' .env | xargs)
#   echo "✓ Environment variables loaded"
#fi

# Create necessary directories if they don't exist
mkdir -p uploads logs exports backups tools

# Set Flask environment variables
export FLASK_APP=app.py
export FLASK_ENV=${FLASK_ENV:-development}

echo ""
echo "Application Information:"
echo "  Host: $HOST"
echo "  Port: $PORT"
echo "  Environment: $FLASK_ENV"
echo ""
echo "Directories:"
echo "  Uploads: $(pwd)/uploads"
echo "  Logs: $(pwd)/logs"
echo "  Exports: $(pwd)/exports"
echo "  Backups: $(pwd)/backups"
echo ""
echo "========================================"
echo "Starting Flask application..."
echo "Press Ctrl+C to stop"
echo "========================================"
echo ""
echo "Access the application at:"
echo "  http://localhost:$PORT"
echo ""
echo "Or on your network at:"
echo "  http://$(hostname -I | awk '{print $1}' 2>/dev/null || echo "127.0.0.1"):$PORT"
echo ""

# Start the Flask application
if [ "$FLASK_ENV" = "production" ]; then
    # Production mode (using waitress for Windows compatibility)
    echo "Starting in PRODUCTION mode..."
    if python -c "import waitress" &> /dev/null; then
        python -c "import waitress; from app import app; waitress.serve(app, host='$HOST', port=$PORT)"
    else
        echo "⚠ Waitress not installed, using Flask development server instead"
        echo "Install production server: pip install waitress"
        flask run --host=$HOST --port=$PORT
    fi
else
    # Development mode
    echo "Starting in DEVELOPMENT mode..."
    flask run --host=$HOST --port=$PORT --debug
fi
