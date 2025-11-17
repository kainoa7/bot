#!/bin/bash

echo "================================================"
echo "  Stock Options Trading Assistant - Setup"
echo "================================================"

# Check Python version
echo -e "\n1. Checking Python version..."
if command -v python3 &> /dev/null; then
    python_version=$(python3 --version)
    echo "   ✓ Found: $python_version"
else
    echo "   ✗ Python 3 not found. Please install Python 3.8 or higher."
    exit 1
fi

# Create virtual environment
echo -e "\n2. Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "   ✓ Virtual environment created"
else
    echo "   ✓ Virtual environment already exists"
fi

# Activate virtual environment
echo -e "\n3. Activating virtual environment..."
source venv/bin/activate
echo "   ✓ Virtual environment activated"

# Install dependencies
echo -e "\n4. Installing dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo "   ✓ Dependencies installed"

# Set up environment file
echo -e "\n5. Setting up environment variables..."
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "   ✓ Created .env file from template"
    echo "   ⚠  Please edit .env and add your API keys:"
    echo "      - Alpha Vantage: https://www.alphavantage.co/support/#api-key"
    echo "      - NewsAPI: https://newsapi.org/register"
else
    echo "   ✓ .env file already exists"
fi

# Make main.py executable
chmod +x main.py

echo -e "\n================================================"
echo "  Setup Complete!"
echo "================================================"
echo ""
echo "Next steps:"
echo "  1. Edit .env file and add your API keys"
echo "  2. Activate the virtual environment:"
echo "     source venv/bin/activate"
echo "  3. Run the assistant:"
echo "     python main.py AAPL"
echo ""
echo "For detailed analysis, use:"
echo "  python main.py AAPL --verbose"
echo ""
echo "================================================"
