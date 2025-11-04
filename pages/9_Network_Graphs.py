import streamlit as st
import sys
from pathlib import Path

# Get absolute path to the actual page script
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
actual_page = project_root / "streamlit_dashboard_pages" / "pages" / "9_Network_Graphs.py"

# Add project root to path for imports
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Read and execute the actual page script
with open(actual_page, 'r', encoding='utf-8') as f:
    code = f.read()
exec(compile(code, str(actual_page), 'exec'), {'__file__': str(actual_page), '__name__': '__main__'})
