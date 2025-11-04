import sys
from pathlib import Path

# Ensure project root on sys.path for package imports  
project_root = Path(__file__).parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import and execute Home module - this allows Streamlit to detect multipage structure
# Importing the module executes its top-level code
from streamlit_dashboard_pages import Home


