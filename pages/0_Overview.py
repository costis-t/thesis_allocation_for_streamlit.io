"""
Home page for Thesis Allocation System
"""
import streamlit as st
from pathlib import Path
import sys

# Add parent and project root to path for imports
try:
    current_dir = Path(__file__).parent
    parent_dir = current_dir.parent  # streamlit_dashboard_pages/
    project_root = parent_dir.parent  # project root
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(parent_dir))
except NameError:
    # Fallback for testing
    project_root = Path.cwd()
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(project_root / "streamlit_dashboard_pages"))


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
</style>
""", unsafe_allow_html=True)

def main():
    st.markdown('<div class="main-header">🎓 Thesis Allocation Dashboard</div>', unsafe_allow_html=True)
    
    # Display cache status in sidebar
    if st.session_state.last_allocation is not None:
        st.sidebar.success("✅ Cached Results Available")
        st.sidebar.write(f"From: {st.session_state.last_allocation_timestamp}")
        if st.sidebar.button("🗑️ Clear Cache"):
            st.session_state.last_allocation = None
            st.session_state.last_summary = None
            st.session_state.last_allocation_rows = None
            st.session_state.last_repos = None
            st.session_state.last_allocation_timestamp = None
            st.rerun()
    
    st.header("Welcome to the Thesis Allocation System")
    
    st.markdown("""
    ## 🎓 Overview
    
    This dashboard provides a comprehensive interface for managing and analyzing thesis allocations for students.
    
    ### Key Features:
    
    - **⚙️ Configuration**: Set up allocation parameters and preferences
    - **🚀 Run Allocation**: Execute allocation algorithms (ILP, Flow, Hybrid)
    - **📊 Results Analysis**: View allocation results and metrics
    - **🔍 Data Explorer**: Explore input data and allocations
    - **📈 Advanced Charts**: Interactive visualizations of allocation data
    
    ### Quick Start:
    
    1. Navigate to **⚙️ Configuration** in the sidebar to set up your allocation parameters
    2. Go to **🚀 Run Allocation** in the sidebar to execute the allocation
    3. View results in **📊 Results Analysis** in the sidebar
    
    ### Navigation:
    
    Use the sidebar to navigate between different sections of the dashboard.
    """)
    
    # System status
    st.subheader("📊 System Status")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Status", "🟢 Operational", delta=None)
    
    with col2:
        students_file = Path("data/input/students.csv")
        if students_file.exists():
            st.metric("Students File", "✅ Available")
        else:
            st.metric("Students File", "❌ Missing")
    
    with col3:
        capacities_file = Path("data/input/capacities.csv")
        if capacities_file.exists():
            st.metric("Capacities File", "✅ Available")
        else:
            st.metric("Capacities File", "❌ Missing")
    
    with col4:
        if st.session_state.get('last_allocation') is not None:
            st.metric("Cached Results", "✅ Available")
        else:
            st.metric("Cached Results", "❌ None")

if __name__ == "__main__":
    main()

