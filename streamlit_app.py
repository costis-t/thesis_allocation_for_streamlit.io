import streamlit as st
import sys
from pathlib import Path

# Ensure project root on sys.path for package imports  
project_root = Path(__file__).parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Execute Home.py directly - this ensures all Streamlit code runs
home_script = project_root / "streamlit_dashboard_pages" / "Home.py"
if home_script.exists():
    # Read and execute the Home script
    with open(home_script, 'r', encoding='utf-8') as f:
        code = f.read()
    exec(compile(code, str(home_script), 'exec'), {'__file__': str(home_script), '__name__': '__main__'})
else:
    st.error(f"❌ Could not find Home.py at {home_script}")


