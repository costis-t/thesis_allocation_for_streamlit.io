import runpy
from pathlib import Path

# Get absolute path to the actual page script
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
actual_page = project_root / "streamlit_dashboard_pages" / "pages" / "1_Grid_Search.py"

runpy.run_path(str(actual_page), run_name="__main__")


