# install.sh - Setup script for Red Team Workflow Automation Platform
# Usage: ./install.sh

set -e  # Exit on any error

echo "========================================"
echo "The Street - Reckoning"
echo "========================================"
echo ""

# Check if Python3 is installed
if ! command -v python &> /dev/null; then
    echo "❌ Python is not installed. Please install Python 3.8 or higher."
    echo "Visit: https://www.python.org/downloads/"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "✓ Python $PYTHON_VERSION detected"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv

if [ ! -d "venv" ]; then
    echo "❌ Failed to create virtual environment"
    exit 1
fi

echo "✓ Virtual environment created at: $(pwd)/venv"

# Activate virtual environment and install requirements
echo ""
echo "Installing dependencies..."
if [ -f "requirements.txt" ]; then
    # Use the existing requirements.txt
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
else
    # requirements.txt is missing
    echo "❌ requirements.txt is missing"
fi

echo "✓ Dependencies installed"

# Check for missing requirements file
if [ ! -f "requirements.txt" ]; then
    echo "⚠ No requirements.txt found"
fi

# Create necessary directories
echo ""
echo "Creating necessary directories..."
mkdir -p uploads logs exports backups tools
echo "✓ Directories created: uploads/, logs/, exports/, backups/, tools/"

# Set permissions for shell scripts
echo ""
echo "Setting permissions..."
chmod +x start.sh 2>/dev/null || true
chmod +x start_windows.bat 2>/dev/null || true
echo "✓ Permissions set"

# Create a default .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo ""
    echo "Creating .env file..."
    cat > .env << EOF
# Application Settings
FLASK_APP=app.py
FLASK_ENV=development
SECRET_KEY=$(openssl rand -hex 24 2>/dev/null || python3 -c "import secrets; print(secrets.token_hex(24))")
MAX_CONTENT_LENGTH=33554432  # 32MB

# Security Settings
SESSION_TIMEOUT=9999
MAX_EXECUTION_TIME=99999
LOG_LEVEL=INFO

# Paths (relative to project root)
UPLOAD_FOLDER=uploads
BACKUP_FOLDER=backups
EXPORT_FOLDER=exports
LOG_FOLDER=logs
TOOLS_FOLDER=tools
EOF
    echo "✓ Created .env file"
fi

echo ""
echo "========================================"
echo "✅ Installation complete!"
echo "========================================"
echo ""
echo "To start the application, run:"
echo "  ./start.sh"
echo ""
echo "Or on Windows:"
echo "  start_windows.bat"
echo ""
echo "The application will be available at:"
echo "  http://localhost:5000"
echo ""
echo "Default credentials (if applicable):"
echo "  Username: admin"
echo "  Password: admin"
echo ""
echo "⚠  Please change the default credentials in production!"
