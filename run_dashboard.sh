#!/bin/bash
# Launch Streamlit Dashboard for Thesis Allocation System

# Get script directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

# Check if venv exists
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate venv
source .venv/bin/activate

# Install requirements if needed
if ! python -c "import streamlit" 2>/dev/null; then
    echo "Installing dependencies..."
    pip install -q -r requirements.txt
fi

# Launch Streamlit
echo ""
echo "🎓 Launching Thesis Allocation Dashboard..."
echo "Dashboard will open at: http://localhost:8501"
echo "Pages will be automatically discovered from: streamlit_dashboard_pages/pages/"
echo "Press Ctrl+C to stop the server"
echo ""

# Run from streamlit_dashboard_pages directory so pages are auto-discovered
cd streamlit_dashboard_pages
streamlit run Home.py
