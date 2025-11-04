import importlib
import sys
from pathlib import Path
import runpy

# Ensure project root on sys.path for package imports
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Delegate to the actual multipage app entry in streamlit_dashboard_pages/Home.py
try:
    importlib.import_module("streamlit_dashboard_pages.Home")
except Exception:
    # Fallback: execute the Home.py script directly
    runpy.run_path(str(project_root / "streamlit_dashboard_pages/Home.py"), run_name="__main__")


