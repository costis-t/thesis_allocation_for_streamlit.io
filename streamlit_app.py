import streamlit as st
import sys
from pathlib import Path
import runpy
import traceback

# Ensure project root on sys.path for package imports
project_root = Path(__file__).parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Execute the Home.py script directly - this ensures all code runs
home_script = project_root / "streamlit_dashboard_pages" / "Home.py"
try:
    if home_script.exists():
        # Use runpy to execute the script - this is the most reliable method
        runpy.run_path(str(home_script), run_name="__main__")
    else:
        st.error(f"❌ Could not find Home.py at {home_script}")
        st.info(f"Project root: {project_root}")
        st.info(f"Looking for: {home_script}")
except Exception as e:
    # If execution fails, show error details
    st.error(f"❌ Error loading Home.py: {e}")
    with st.expander("Error Details"):
        st.code(traceback.format_exc())
    st.info(f"Project root: {project_root}")
    st.info(f"Home script: {home_script}")
    st.info(f"Home exists: {home_script.exists()}")


