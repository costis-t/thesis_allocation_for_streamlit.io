"""
Thesis Allocation System - Main Dashboard Entry Point
"""
import streamlit as st
from pathlib import Path
import sys

# Add parent and project root to path for imports
try:
    current_dir = Path(__file__).parent
    project_root = current_dir.parent  # project root
    sys.path.insert(0, str(project_root))
except NameError:
    # Fallback for testing
    project_root = Path.cwd()
    sys.path.insert(0, str(project_root))

from streamlit_dashboard_pages.shared import initialize_session_state, safe_set_page_config

# Initialize session state
initialize_session_state()

# Set page config
safe_set_page_config(
    page_title="Thesis Allocation Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
<style>
    .main-header {
        font-size: 3em;
        color: #0066cc;
        font-weight: bold;
        margin-bottom: 1em;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5em;
        border-radius: 0.5em;
        border-left: 5px solid #0066cc;
    }
    .success-color { color: #00cc00; }
    .warning-color { color: #ff9900; }
    .error-color { color: #cc0000; }

    /* Ensure Streamlit sidebar navigation items are visible */
    [data-testid="stSidebar"] { color: #222; }
    [data-testid="stSidebarNav"] a { color: #222 !important; }
    [data-testid="stSidebarNav"] ul { color: #222 !important; }
    /* If using dark mode backgrounds elsewhere, uncomment the following to force light text
    [data-testid="stSidebarNav"] a { color: #e0e0e0 !important; }
    [data-testid="stSidebarNav"] ul { color: #e0e0e0 !important; }
    */
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🎓  Thesis Allocation Dashboard</div>', unsafe_allow_html=True)

# Main page content
st.header("Welcome to Thesis Allocation System")
st.write("""
This dashboard allows you to configure, run, and analyze thesis allocations for students.

### 📋 Navigation
Use the sidebar navigation to access different sections:

- **🏠 Home** - Overview and dashboard information
- **🔍 Data Explorer** - Explore and validate input data
- **⚙️ Configuration** - Configure allocation settings and preferences
- **🚀 Run Allocation** - Execute thesis allocation with live progress
- **📊 Results Analysis** - View results, charts, and metrics

### 🚀 Getting Started

1. **Explore Data** (Data Explorer) - Review your input files
2. **Configure** (Configuration) - Set your allocation preferences
3. **Run** (Run Allocation) - Execute the allocation algorithm
4. **Analyze** (Results Analysis) - View results and visualizations

### 💡 Quick Tips

- Use **Configuration** to adjust preferences, penalties, and solver settings
- **Run Allocation** provides real-time progress and detailed diagnostics
- **Results Analysis** includes interactive charts and download options
- Session state is preserved across pages for seamless workflow
""")

# Display cache status
if st.session_state.last_allocation is not None:
    st.divider()
    st.success("✅ Cached Results Available")
    st.write(f"**Timestamp:** {st.session_state.last_allocation_timestamp}")
    st.write("You can access these results in the **Results Analysis** page.")
    
    if st.button("🗑️ Clear Cache"):
        st.session_state.last_allocation = None
        st.session_state.last_summary = None
        st.session_state.last_allocation_rows = None
        st.session_state.last_repos = None
        st.session_state.last_allocation_timestamp = None
        st.rerun()

# File status
st.divider()
st.subheader("📂 Input Files")
col1, col2 = st.columns(2)

with col1:
    students_file = st.session_state.students_file
    if students_file.exists():
        st.success(f"✅ Students file ready: `{students_file}`")
    else:
        st.warning(f"⚠️ Students file not found: `{students_file}`")

with col2:
    capacities_file = st.session_state.capacities_file
    if capacities_file.exists():
        st.success(f"✅ Capacities file ready: `{capacities_file}`")
    else:
        st.warning(f"⚠️ Capacities file not found: `{capacities_file}`")

# Add license in sidebar
st.sidebar.markdown("---")
# st.sidebar.markdown("""
# <div style='text-align: center; font-size: 11px; color: #666; padding: 10px;'>
#     <a href='https://www.gnu.org/licenses/gpl-3.0.en.html' target='_blank' style='color: #666; text-decoration: none;'>
#         📄 Licensed under GNU GPLv3
#     </a>
# </div>
# """, unsafe_allow_html=True)
