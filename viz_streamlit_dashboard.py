"""
Thesis Allocation System!!! - Interactive Streamlit Dashboard
Provides real-time visualization and configuration of thesis allocations, only for MiM students!
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from pathlib import Path
import json
from io import StringIO
from datetime import datetime
import tempfile
import shutil

# Import from allocator
from allocator.data_repository import DataRepository
from allocator.preference_model import PreferenceModel
from allocator.allocation_model_ilp import AllocationModelILP, AllocationConfig as LegacyAllocationConfig
from allocator.allocation_model_flow import AllocationModelFlow
from allocator.config import AllocationConfig, PreferenceConfig, CapacityConfig, SolverConfig
from allocator.validation import InputValidator
from allocator.logging_config import setup_logging
from allocator.outputs import write_summary_txt
from viz_sankey_enhanced import create_sankey_html


# Set page config
st.set_page_config(
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
</style>
""", unsafe_allow_html=True)


# Initialize session state for file persistence and results caching
if 'uploaded_students' not in st.session_state:
    st.session_state.uploaded_students = None
if 'uploaded_capacities' not in st.session_state:
    st.session_state.uploaded_capacities = None
if 'uploaded_overrides' not in st.session_state:
    st.session_state.uploaded_overrides = None
if 'last_allocation' not in st.session_state:
    st.session_state.last_allocation = None
if 'last_summary' not in st.session_state:
    st.session_state.last_summary = None
if 'last_allocation_timestamp' not in st.session_state:
    st.session_state.last_allocation_timestamp = None
if 'last_allocation_rows' not in st.session_state:
    st.session_state.last_allocation_rows = None
if 'last_repos' not in st.session_state:
    st.session_state.last_repos = None

# Initialize default file paths
if 'students_file' not in st.session_state:
    st.session_state.students_file = Path("data/input/students.csv")
if 'capacities_file' not in st.session_state:
    st.session_state.capacities_file = Path("data/input/capacities.csv")

# Load configuration from file if it exists, otherwise use defaults
config_file = Path("config_streamlit.json")
if config_file.exists():
    try:
        config = AllocationConfig.load_json(str(config_file))
        # Use config values but ensure capacity settings are False by default
        st.session_state.config_allow_unranked = config.preference.allow_unranked
        st.session_state.config_tier2_cost = config.preference.tier2_cost
        st.session_state.config_tier3_cost = config.preference.tier3_cost
        st.session_state.config_unranked_cost = config.preference.unranked_cost
        st.session_state.config_top2_bias = config.preference.top2_bias
        st.session_state.config_rank1_cost = getattr(config.preference, 'rank1_cost', 0)
        st.session_state.config_rank2_cost = getattr(config.preference, 'rank2_cost', 1)
        st.session_state.config_rank3_cost = getattr(config.preference, 'rank3_cost', 100)
        st.session_state.config_rank4_cost = getattr(config.preference, 'rank4_cost', 101)
        st.session_state.config_rank5_cost = getattr(config.preference, 'rank5_cost', 102)
        st.session_state.config_min_pref = config.preference.min_acceptable_preference_rank
        st.session_state.config_max_pref = config.preference.max_acceptable_preference_rank
        st.session_state.config_excluded_prefs = config.preference.excluded_preference_ranks or []
        # Force capacity settings to False (new default)
        st.session_state.config_enable_topic_overflow = False
        st.session_state.config_enable_coach_overflow = False
        st.session_state.config_dept_min_mode = config.capacity.dept_min_mode
        st.session_state.config_dept_max_mode = config.capacity.dept_max_mode
        st.session_state.config_P_dept_shortfall = config.capacity.P_dept_shortfall
        st.session_state.config_P_dept_overflow = config.capacity.P_dept_overflow
        st.session_state.config_P_topic = config.capacity.P_topic
        st.session_state.config_P_coach = config.capacity.P_coach
        st.session_state.config_algorithm = config.solver.algorithm
        st.session_state.config_time_limit = config.solver.time_limit_sec or 60
        st.session_state.config_random_seed = config.solver.random_seed
        st.session_state.config_epsilon = config.solver.epsilon_suboptimal or 0.0
    except Exception as e:
        st.error(f"Error loading config file: {e}. Using defaults.")
        # Fall through to default initialization
        config_file = None

# Initialize configuration settings with defaults (only if not loaded from file)
if not config_file or not config_file.exists():
    if 'config_allow_unranked' not in st.session_state:
        st.session_state.config_allow_unranked = False
    if 'config_tier2_cost' not in st.session_state:
        st.session_state.config_tier2_cost = 1
    if 'config_tier3_cost' not in st.session_state:
        st.session_state.config_tier3_cost = 5
    if 'config_unranked_cost' not in st.session_state:
        st.session_state.config_unranked_cost = 200
    if 'config_top2_bias' not in st.session_state:
        st.session_state.config_top2_bias = False
    if 'config_min_pref' not in st.session_state:
        st.session_state.config_min_pref = None
    if 'config_max_pref' not in st.session_state:
        st.session_state.config_max_pref = None
    if 'config_excluded_prefs' not in st.session_state:
        st.session_state.config_excluded_prefs = []
    if 'config_rank1_cost' not in st.session_state:
        st.session_state.config_rank1_cost = 0
    if 'config_rank2_cost' not in st.session_state:
        st.session_state.config_rank2_cost = 1
    if 'config_rank3_cost' not in st.session_state:
        st.session_state.config_rank3_cost = 100
    if 'config_rank4_cost' not in st.session_state:
        st.session_state.config_rank4_cost = 101
    if 'config_rank5_cost' not in st.session_state:
        st.session_state.config_rank5_cost = 102
    if 'config_enable_topic_overflow' not in st.session_state:
        st.session_state.config_enable_topic_overflow = False
    if 'config_enable_coach_overflow' not in st.session_state:
        st.session_state.config_enable_coach_overflow = False
    if 'config_dept_min_mode' not in st.session_state:
        st.session_state.config_dept_min_mode = "soft"
    if 'config_dept_max_mode' not in st.session_state:
        st.session_state.config_dept_max_mode = "soft"
    if 'config_P_dept_shortfall' not in st.session_state:
        st.session_state.config_P_dept_shortfall = 1000
    if 'config_P_dept_overflow' not in st.session_state:
        st.session_state.config_P_dept_overflow = 1200
    if 'config_P_topic' not in st.session_state:
        st.session_state.config_P_topic = 800
    if 'config_P_coach' not in st.session_state:
        st.session_state.config_P_coach = 600
    if 'config_algorithm' not in st.session_state:
        st.session_state.config_algorithm = "ilp"
    if 'config_time_limit' not in st.session_state:
        st.session_state.config_time_limit = 60
    if 'config_random_seed' not in st.session_state:
        st.session_state.config_random_seed = None
    if 'config_epsilon' not in st.session_state:
        st.session_state.config_epsilon = 0.0


# Ensure visualizations folder exists
VISUALIZATIONS_DIR = Path("visualisations")
VISUALIZATIONS_DIR.mkdir(exist_ok=True)


def save_visualization(html_content, filename):
    """Save visualization HTML to visualizations folder and return path."""
    filepath = VISUALIZATIONS_DIR / filename
    filepath.write_text(html_content)
    return filepath


def download_combined_results(allocation_df, summary_text):
    """Create download options for allocation results."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        csv_data = allocation_df.to_csv(index=False)
        st.download_button(
            "📥 Allocation CSV",
            csv_data,
            "allocation_result.csv",
            "text/csv"
        )
    
    with col2:
        st.download_button(
            "📥 Summary TXT",
            summary_text,
            "allocation_summary.txt",
            "text/plain"
        )
    
    with col3:
        # Combined download as both CSV + TXT (separate files)
        import zipfile
        import io
        
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            zip_file.writestr("allocation.csv", allocation_df.to_csv(index=False))
            zip_file.writestr("summary.txt", summary_text)
        
        zip_buffer.seek(0)
        st.download_button(
            "📦 Both as ZIP",
            zip_buffer.getvalue(),
            "allocation_results.zip",
            "application/zip"
        )
    
    with col4:
        st.write("")  # Placeholder for alignment


def create_preference_satisfaction_chart(summary_data):
    """Create preference satisfaction visualization."""
    if "Ranked choice satisfaction:" not in summary_data:
        return None
    
    # Parse ranked satisfaction
    lines = summary_data.split('\n')
    satisfaction = {}
    in_ranked = False
    for line in lines:
        if "Ranked choice satisfaction:" in line:
            in_ranked = True
        elif in_ranked and ':' in line and not line.startswith('  Unranked'):
            parts = line.strip().split(':')
            if len(parts) == 2:
                key = parts[0].strip()
                try:
                    value = int(parts[1].strip())
                    satisfaction[key] = value
                except:
                    pass
        elif in_ranked and line.strip() == "":
            break
    
    if satisfaction:
        fig = px.bar(
            x=list(satisfaction.keys()),
            y=list(satisfaction.values()),
            title="Preference Satisfaction (Ranked Choices)",
            labels={"x": "Choice Rank", "y": "Number of Students"},
            color=list(satisfaction.values()),
            color_continuous_scale="Viridis"
        )
        fig.update_layout(height=400, showlegend=False)
        return fig
    return None


def create_capacity_utilization_chart(summary_data):
    """Create topic capacity utilization visualization."""
    if "Topic utilization:" not in summary_data:
        return None
    
    # Parse topic utilization
    lines = summary_data.split('\n')
    topics = {}
    in_util = False
    for line in lines:
        if "Topic utilization:" in line:
            in_util = True
        elif in_util and ':' in line and line.startswith('  topic'):
            parts = line.strip().split(':')[1].split('/')
            if len(parts) == 2:
                topic_name = line.split(':')[0].strip()
                used = int(parts[0].strip())
                total = int(parts[1].strip())
                topics[topic_name] = {'used': used, 'total': total, 'pct': (used/total*100) if total > 0 else 0}
        elif in_util and line.strip() == "" and topics:
            break
    
    if topics:
        df = pd.DataFrame(topics).T
        fig = px.bar(
            df,
            x=df.index,
            y=['used', 'total'],
            title="Topic Capacity Utilization",
            labels={"index": "Topic", "value": "Number of Students"},
            barmode="overlay",
            color_discrete_map={"used": "#0066cc", "total": "#cccccc"}
        )
        fig.update_layout(height=400, xaxis_tickangle=-45)
        return fig
    return None


def create_department_distribution_chart(summary_data):
    """Create department distribution visualization."""
    if "Department totals:" not in summary_data:
        return None
    
    lines = summary_data.split('\n')
    departments = {}
    in_dept = False
    for line in lines:
        if "Department totals:" in line:
            in_dept = True
        elif in_dept and ':' in line and line.startswith('  department'):
            parts = line.split(':')
            if len(parts) >= 2:
                dept_name = parts[0].strip()
                try:
                    count = int(parts[1].split('(')[0].strip())
                    departments[dept_name] = count
                except:
                    pass
        elif in_dept and line.strip() == "" and departments:
            break
    
    if departments:
        fig = px.pie(
            values=list(departments.values()),
            names=list(departments.keys()),
            title="Student Distribution by Department",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig.update_layout(height=400)
        return fig
    return None


def create_allocation_summary_metrics(allocation_df, summary_data):
    """Create summary metrics from allocation results."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_students = len(allocation_df)
        st.metric(
            "Total Students Assigned",
            total_students,
            delta=None,
            delta_color="off"
        )
    
    with col2:
        first_choice = len(allocation_df[allocation_df['preference_rank'].between(10, 14)])
        pct = (first_choice / total_students * 100) if total_students > 0 else 0
        st.metric(
            "Got Ranked Choice",
            f"{first_choice}",
            delta=f"{pct:.1f}%"
        )
    
    with col3:
        if "Objective:" in summary_data:
            objective = summary_data.split("Objective:")[1].split('\n')[0].strip()
            st.metric(
                "Optimal Cost",
                objective,
                delta=None,
                delta_color="off"
            )
    
    with col4:
        unassigned = summary_data.count("Unassigned after solve: 0")
        status = "✓ All Assigned" if unassigned > 0 else "⚠ Some Unassigned"
        st.metric(
            "Assignment Status",
            status,
            delta=None,
            delta_color="off"
        )


# Helper function to calculate Gini coefficient
def calculate_gini(values):
    """Calculate Gini coefficient (0=perfect equality, 1=perfect inequality)"""
    if len(values) == 0:
        return 0
    sorted_vals = np.sort(values)
    n = len(sorted_vals)
    # Gini = (2 * sum(i * x_i)) / (n * sum(x_i)) - (n + 1) / n
    cumsum = np.cumsum(sorted_vals)
    gini = (2 * np.sum(np.arange(1, n + 1) * sorted_vals)) / (n * np.sum(sorted_vals)) - (n + 1) / n
    return max(0, gini)  # Ensure non-negative


# Helper function to calculate fairness score
def calculate_fairness_score(allocation_df):
    """
    Calculate multiple fairness metrics
    Returns dict with various fairness indicators
    """
    metrics = {}
    
    if 'effective_cost' in allocation_df.columns:
        costs = allocation_df['effective_cost'].values
        metrics['gini_cost'] = calculate_gini(costs)
        metrics['cost_std'] = np.std(costs)
        metrics['cost_cv'] = metrics['cost_std'] / (np.mean(costs) + 1e-6)  # Coefficient of variation
        metrics['cost_mean'] = np.mean(costs)
        metrics['cost_median'] = np.median(costs)
    
    if 'preference_rank' in allocation_df.columns:
        # Count students by satisfaction level
        ranked_students = len(allocation_df[allocation_df['preference_rank'].between(10, 14)])
        tier_students = len(allocation_df[allocation_df['preference_rank'].between(0, 2)])
        total_students = len(allocation_df)
        
        metrics['ranked_satisfaction'] = ranked_students / total_students if total_students > 0 else 0
        metrics['tier_satisfaction'] = tier_students / total_students if total_students > 0 else 0
    
    if 'assigned_topic' in allocation_df.columns:
        topic_counts = allocation_df['assigned_topic'].value_counts().values
        metrics['gini_topics'] = calculate_gini(topic_counts)
        metrics['topic_balance'] = 1 - metrics['gini_topics']
    
    if 'assigned_coach' in allocation_df.columns:
        coach_counts = allocation_df['assigned_coach'].value_counts().values
        metrics['gini_coaches'] = calculate_gini(coach_counts)
        metrics['coach_balance'] = 1 - metrics['gini_coaches']
    
    if 'department_id' in allocation_df.columns:
        dept_counts = allocation_df['department_id'].value_counts().values
        metrics['gini_departments'] = calculate_gini(dept_counts)
        metrics['dept_balance'] = 1 - metrics['gini_departments']
    
    return metrics


def main():
    """Main Streamlit app."""
    st.markdown('<div class="main-header">Thesis Allocation Dashboard</div>', unsafe_allow_html=True)
    
    # Sidebar for navigation
    st.sidebar.title("📋 Navigation")
    page = st.sidebar.radio(
        "Select View",
        ["🏠 Home", "⚙️ Configuration", "🚀 Run Allocation", "📊 Results Analysis", "🔍 Data Explorer", "📈 Advanced Charts", "🚀 Really Advanced Charts", "⚖️ Compare Allocations"]
    )
    
    # Display cache status in sidebar
    if st.session_state.last_allocation is not None:
        st.sidebar.divider()
        st.sidebar.success("✅ Cached Results Available")
        st.sidebar.write(f"From: {st.session_state.last_allocation_timestamp}")
        if st.sidebar.button("🗑️ Clear Cache"):
            st.session_state.last_allocation = None
            st.session_state.last_summary = None
            st.session_state.last_allocation_rows = None
            st.session_state.last_repos = None
            st.session_state.last_allocation_timestamp = None
            st.rerun()
    
    # License information in sidebar
    st.sidebar.divider()
    st.sidebar.markdown(
        """
        <div style='position: fixed; bottom: 10px; left: 10px; font-size: 12px; color: #666;'>
            <a href='https://www.gnu.org/licenses/gpl-3.0.en.html' target='_blank' style='color: #666; text-decoration: none;'>
                📄 GPLv3 License
            </a>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    # Home Page
    if page == "🏠 Home":
        st.header("Welcome to the Thesis Allocation System")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📖 About")
            st.write("""
            This interactive dashboard allows you to:
            - Configure thesis allocation parameters
            - Upload student preferences and topic data
            - Run allocations with different algorithms
            - Analyze results with rich visualizations
            - Export allocation results
            """)
        
        with col2:
            st.subheader("🚀 Quick Start")
            st.write("""
            1. Go to **Configuration** to set parameters
            2. Upload CSV files with students and capacities
            3. Click **Run Allocation**
            4. View results in **Results Analysis**
            5. Explore data in **Data Explorer**
            """)
        
        st.divider()
        
        st.subheader("📂 File Management")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Default Data:**")
            st.success("✅ Default files preloaded:")
            st.write(f"📄 Students: `{st.session_state.students_file}`")
            st.write(f"📄 Capacities: `{st.session_state.capacities_file}`")
            st.info("💡 These files are automatically loaded. Upload custom files below to override.")
        
        with col2:
            st.write("**Upload Custom Data:**")
            students_upload = st.file_uploader("Students CSV", type=['csv'], key="students_upload")
            capacities_upload = st.file_uploader("Capacities CSV", type=['csv'], key="capacities_upload")
        
        # License information at bottom of Home page
        st.divider()
        st.markdown(
            """
            <div style='text-align: left; font-size: 12px; color: #666; margin-top: 20px;'>
                <a href='https://www.gnu.org/licenses/gpl-3.0.en.html' target='_blank' style='color: #666; text-decoration: none;'>
                    📄 Licensed under GNU GPLv3
                </a>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    # Configuration Page
    elif page == "⚙️ Configuration":
        st.header("⚙️ Configuration")
        st.info("""
        📋 **Configuration Guide:**
        These settings control how the allocation algorithm behaves. 
        Adjust them to prioritize different outcomes (fairness, speed, preference satisfaction, etc.)
        """)
        
        # Explanation of Tiers and Penalties
        with st.expander("📚 What are Tiers and Penalties?", expanded=False):
            st.markdown("""
            ### 🎯 **What are Tiers?**
            
            **Tiers** represent **indifference groups** - topics that students consider equally desirable:
            - **Tier 1**: Topics the student likes most (all equally preferred)
            - **Tier 2**: Topics the student likes moderately (all equally preferred)  
            - **Tier 3**: Topics the student likes least (all equally preferred)
            - **Unranked**: Topics the student didn't rank (didn't care about)
            
            **Example**: If a student puts topics A, B, C in Tier 1, they're saying "I like A, B, and C equally - any of them would be great!"
            
            ### ⚖️ **What are Penalties?**
            
            **Penalties** are "costs" the algorithm pays when assigning students to topics:
            - **Lower penalty** = Algorithm prefers this assignment
            - **Higher penalty** = Algorithm tries to avoid this assignment
            
            **Examples:**
            - Tier 2 Cost = 1 → Small penalty for Tier 2 topics (algorithm is OK with this)
            - Tier 3 Cost = 5 → Medium penalty for Tier 3 topics (algorithm prefers Tier 1/2)
            - Unranked Cost = 200 → Huge penalty (algorithm really tries to avoid this)
            
            ### 🎮 **How it Works:**
            The algorithm tries to minimize total penalties across all students. 
            So if you set Tier 2 Cost = 1 and Tier 3 Cost = 5, the algorithm will 
            strongly prefer giving students their Tier 2 topics over their Tier 3 topics.
        """)
        
        st.subheader("Preference Settings")
        st.markdown("*How to value different preference levels*")
        col1, col2 = st.columns(2)
        
        with col1:
            allow_unranked = st.checkbox(
                "Allow Unranked Topics", 
                value=st.session_state.config_allow_unranked,
                help="If OFF: Students MUST get a ranked preference. If ON: Can be assigned to any topic."
            )
            tier2_cost = st.slider(
                "Tier 2 Cost", 
                0, 10, st.session_state.config_tier2_cost,
                help="Penalty for Tier 2 topics (moderately preferred). Lower = algorithm prefers Tier 2. Higher = algorithm avoids Tier 2. Default: 1"
            )
            tier3_cost = st.slider(
                "Tier 3 Cost", 
                0, 20, st.session_state.config_tier3_cost,
                help="Penalty for Tier 3 topics (least preferred). Lower = algorithm prefers Tier 3. Higher = algorithm avoids Tier 3. Default: 5"
            )
        
        with col2:
            unranked_cost = st.slider(
                "Unranked Cost", 
                0, 500, st.session_state.config_unranked_cost,
                help="Penalty for unranked topics (topics student didn't rank). Very high = algorithm strongly avoids unranked assignments. Default: 200"
            )
            top2_bias = st.checkbox(
                "Apply Top-2 Bias", 
                value=st.session_state.config_top2_bias,
                help="If ON: Strongly prefer 1st & 2nd choices. If OFF: All ranks treated equally. This setting only affects default values, not your custom ranked choice costs."
            )
        
        st.divider()
        st.subheader("Ranked Choice Costs")
        st.markdown("*Configure penalties for each ranked choice level*")
        st.info("💡 **Tip**: Lower costs = algorithm prefers this choice. Higher costs = algorithm avoids this choice.")
        
        # Debug: Show current session state values
        with st.expander("🔍 Debug - Current Session State Values"):
            st.write("**Ranked Choice Costs in Session State:**")
            st.write(f"- rank1_cost: {st.session_state.config_rank1_cost}")
            st.write(f"- rank2_cost: {st.session_state.config_rank2_cost}")
            st.write(f"- rank3_cost: {st.session_state.config_rank3_cost}")
            st.write(f"- rank4_cost: {st.session_state.config_rank4_cost}")
            st.write(f"- rank5_cost: {st.session_state.config_rank5_cost}")
            st.write(f"- top2_bias: {st.session_state.config_top2_bias}")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            rank1_cost = st.slider(
                "1st Choice Cost", 
                0, 50, st.session_state.config_rank1_cost,
                help="Penalty for 1st choice (student's #1 ranked topic). Usually 0 (best)."
            )
            rank2_cost = st.slider(
                "2nd Choice Cost", 
                0, 50, st.session_state.config_rank2_cost,
                help="Penalty for 2nd choice (student's #2 ranked topic). Usually 1."
            )
        
        with col2:
            rank3_cost = st.slider(
                "3rd Choice Cost", 
                0, 200, st.session_state.config_rank3_cost,
                help="Penalty for 3rd choice (student's #3 ranked topic). Usually 100."
            )
            rank4_cost = st.slider(
                "4th Choice Cost", 
                0, 200, st.session_state.config_rank4_cost,
                help="Penalty for 4th choice (student's #4 ranked topic). Usually 101."
            )
        
        with col3:
            rank5_cost = st.slider(
                "5th Choice Cost", 
                0, 200, st.session_state.config_rank5_cost,
                help="Penalty for 5th choice (student's #5 ranked topic). Usually 102."
            )
        
        st.divider()
        st.subheader("Preference Satisfaction Constraints")
        st.markdown("*Force minimum/maximum satisfaction levels*")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            min_pref = st.selectbox(
                "Minimum Preference", 
                [None, 10, 11, 12, 13, 14],
                format_func=lambda x: "None" if x is None else f"{'1st' if x==10 else '2nd' if x==11 else '3rd' if x==12 else '4th' if x==13 else '5th'} choice (rank {x})",
                help="Force NO ONE to get WORSE than this level. Example: 11 = everyone gets 1st or 2nd choice minimum."
            )
        
        with col2:
            max_pref = st.selectbox(
                "Maximum Preference", 
                [None, 10, 11, 12, 13, 14],
                format_func=lambda x: "None" if x is None else f"{'1st' if x==10 else '2nd' if x==11 else '3rd' if x==12 else '4th' if x==13 else '5th'} choice (rank {x})",
                help="Force NO ONE to get BETTER than this level. Example: 12 = no one better than 3rd choice."
            )
        
        with col3:
            excluded_str = st.text_input(
                "Excluded Preferences", 
                "",
                help="Comma-separated ranks to exclude (e.g., '10,14' = no 1st or 5th choice). Leave empty for none."
            )
            excluded_prefs = []
            if excluded_str.strip():
                try:
                    excluded_prefs = [int(x.strip()) for x in excluded_str.split(",")]
                except ValueError:
                    st.error("❌ Invalid format. Use comma-separated numbers (e.g., '10,14')")
                    excluded_prefs = []
        
        st.divider()
        st.subheader("Capacity Settings")
        st.markdown("*How to handle capacity constraints*")
        col1, col2 = st.columns(2)
        
        with col1:
            enable_topic_overflow = st.checkbox(
                "Enable Topic Overflow", 
                value=st.session_state.config_enable_topic_overflow,
                help="If ON: Topics can exceed capacity (with penalty). If OFF: Hard cap on topics."
            )
            enable_coach_overflow = st.checkbox(
                "Enable Coach Overflow", 
                value=st.session_state.config_enable_coach_overflow,
                help="If ON: Coaches can exceed capacity (with penalty). If OFF: Hard cap on coaches."
            )
            dept_min_mode = st.selectbox(
                "Department Min Mode", 
                ["soft", "hard"],
                index=0 if st.session_state.config_dept_min_mode == "soft" else 1,
                help="'soft' = Try but don't require minimums. 'hard' = Enforce department minimums strictly."
            )
            dept_max_mode = st.selectbox(
                "Department Max Mode", 
                ["soft", "hard"],
                index=0 if st.session_state.config_dept_max_mode == "soft" else 1,
                help="'soft' = Try but allow exceeding maximums (with penalty). 'hard' = Enforce department maximums strictly."
            )
        
        with col2:
            P_dept_shortfall = st.slider(
                "Dept Shortfall Penalty", 
                0, 5000, st.session_state.config_P_dept_shortfall,
                help="Penalty when department minimum not met (higher = stricter enforcement). Default: 1000"
            )
            P_dept_overflow = st.slider(
                "Dept Overflow Penalty", 
                0, 5000, st.session_state.config_P_dept_overflow,
                help="Penalty when department maximum exceeded (higher = stricter enforcement). Default: 1200"
            )
            P_topic = st.slider(
                "Topic Overflow Penalty", 
                0, 2000, st.session_state.config_P_topic,
                help="Penalty when topic exceeds capacity (higher = stricter). Default: 800"
            )
            P_coach = st.slider(
                "Coach Overflow Penalty", 
                0, 2000, st.session_state.config_P_coach,
                help="Penalty when coach exceeds capacity (higher = stricter). Default: 600"
            )
        
        st.markdown("💡 **Tip**: Higher penalties = stricter constraints = slower solving but fairer results")
        
        st.divider()
        st.subheader("Solver Settings")
        st.markdown("*Algorithm selection and optimization parameters*")
        col1, col2 = st.columns(2)
        
        with col1:
            algorithm = st.selectbox(
                "Algorithm", 
                ["ilp", "flow", "hybrid"],
                index=["ilp", "flow", "hybrid"].index(st.session_state.config_algorithm),
                help="""
                • 'ilp' = Optimal solution (slow, up to 2 min)
                • 'flow' = Fast approximate solution (seconds)
                • 'hybrid' = ILP verified with flow (balanced)
                """
            )
            time_limit = st.slider(
                "Time Limit (seconds)", 
                0, 600, st.session_state.config_time_limit,
                help="Max time solver can spend. 0 = no limit. Higher = better results but slower."
            )
        
        with col2:
            random_seed = st.number_input(
                "Random Seed", 
                value=st.session_state.config_random_seed, 
                min_value=0,
                help="Same seed = same results (for reproducibility). Leave empty for random."
            )
            epsilon = st.slider(
                "Epsilon Suboptimal", 
                0.0, 1.0, st.session_state.config_epsilon, 0.05,
                help="Allow solutions within X% of optimal (e.g., 0.05 = 5% worse but faster). Default: 0 (optimal only)"
            )
        
        # Store configuration in session state
        st.session_state.config_allow_unranked = allow_unranked
        st.session_state.config_tier2_cost = tier2_cost
        st.session_state.config_tier3_cost = tier3_cost
        st.session_state.config_unranked_cost = unranked_cost
        st.session_state.config_top2_bias = top2_bias
        st.session_state.config_rank1_cost = rank1_cost
        st.session_state.config_rank2_cost = rank2_cost
        st.session_state.config_rank3_cost = rank3_cost
        st.session_state.config_rank4_cost = rank4_cost
        st.session_state.config_rank5_cost = rank5_cost
        st.session_state.config_min_pref = min_pref
        st.session_state.config_max_pref = max_pref
        st.session_state.config_excluded_prefs = excluded_prefs
        st.session_state.config_enable_topic_overflow = enable_topic_overflow
        st.session_state.config_enable_coach_overflow = enable_coach_overflow
        st.session_state.config_dept_min_mode = dept_min_mode
        st.session_state.config_dept_max_mode = dept_max_mode
        st.session_state.config_P_dept_shortfall = P_dept_shortfall
        st.session_state.config_P_dept_overflow = P_dept_overflow
        st.session_state.config_P_topic = P_topic
        st.session_state.config_P_coach = P_coach
        st.session_state.config_algorithm = algorithm
        st.session_state.config_time_limit = time_limit
        st.session_state.config_random_seed = random_seed
        st.session_state.config_epsilon = epsilon
        
        st.markdown("💡 **Quick presets:**")
        st.markdown("""
        - **Fast**: flow, 10-30 sec, results in seconds
        - **Balanced**: hybrid, 60 sec, good quality & speed
        - **Optimal**: ilp, 300 sec, best results
        """)
        
        # Save config
        st.divider()
        if st.button("💾 Save Configuration"):
            # Ensure CapacityConfig is available in this scope
            from allocator.config import CapacityConfig, SolverConfig, PreferenceConfig, AllocationConfig
            config = AllocationConfig(
                preference=PreferenceConfig(
                    allow_unranked=st.session_state.config_allow_unranked,
                    tier2_cost=st.session_state.config_tier2_cost,
                    tier3_cost=st.session_state.config_tier3_cost,
                    unranked_cost=st.session_state.config_unranked_cost,
                    top2_bias=st.session_state.config_top2_bias,
                    rank1_cost=st.session_state.config_rank1_cost,
                    rank2_cost=st.session_state.config_rank2_cost,
                    rank3_cost=st.session_state.config_rank3_cost,
                    rank4_cost=st.session_state.config_rank4_cost,
                    rank5_cost=st.session_state.config_rank5_cost,
                    min_acceptable_preference_rank=st.session_state.config_min_pref,
                    max_acceptable_preference_rank=st.session_state.config_max_pref,
                    excluded_preference_ranks=st.session_state.config_excluded_prefs if st.session_state.config_excluded_prefs else None
                ),
                capacity=CapacityConfig(
                    enable_topic_overflow=st.session_state.config_enable_topic_overflow,
                    enable_coach_overflow=st.session_state.config_enable_coach_overflow,
                    dept_min_mode=st.session_state.config_dept_min_mode,
                    dept_max_mode=st.session_state.config_dept_max_mode,
                    P_dept_shortfall=st.session_state.config_P_dept_shortfall,
                    P_dept_overflow=st.session_state.config_P_dept_overflow,
                    P_topic=st.session_state.config_P_topic,
                    P_coach=st.session_state.config_P_coach
                ),
                solver=SolverConfig(
                    algorithm=st.session_state.config_algorithm,
                    time_limit_sec=st.session_state.config_time_limit if st.session_state.config_time_limit > 0 else None,
                    random_seed=st.session_state.config_random_seed if st.session_state.config_random_seed and st.session_state.config_random_seed > 0 else None,
                    epsilon_suboptimal=st.session_state.config_epsilon if st.session_state.config_epsilon > 0 else None
                )
            )
            config.save_json("config_streamlit.json")
            st.success("✓ Configuration saved to config_streamlit.json")
            st.info("💡 Go to 🚀 Run Allocation to use this configuration!")
    
    # Run Allocation Page
    elif page == "🚀 Run Allocation":
        st.header("🚀 Run Allocation")
        st.write("Run thesis allocation directly from the dashboard with live progress tracking.")
        
        # Show if configuration is being used
        if hasattr(st.session_state, 'config_algorithm') and config_file.exists():
            st.success("⚙️ Using saved configuration from ⚙️ Configuration page!")
        
        col1, col2 = st.columns(2)
        
        # File uploads
        with col1:
            st.subheader("📥 Input Files")
            
            # Show preloaded default files
            st.info("📄 **Default files preloaded:**")
            st.write(f"• Students: `{st.session_state.students_file}`")
            st.write(f"• Capacities: `{st.session_state.capacities_file}`")
            st.write("💡 Upload custom files below to override defaults")
            
            st.divider()
            
            students_file = st.file_uploader(
                "Students CSV (Override Default)",
                type=['csv'],
                key="run_students",
                help="CSV with student preferences. Leave empty to use default file."
            )
            capacities_file = st.file_uploader(
                "Capacities CSV (Override Default)",
                type=['csv'],
                key="run_capacities",
                help="CSV with topic/coach capacities. Leave empty to use default file."
            )
            overrides_file = st.file_uploader(
                "Overrides CSV (Optional)",
                type=['csv'],
                key="run_overrides",
                help="Optional: CSV with manual cost overrides"
            )
        
        with col2:
            st.subheader("⚙️ Algorithm Settings")
            # Use saved algorithm from config, default to "ilp" if not set
            default_algorithm = st.session_state.config_algorithm if hasattr(st.session_state, 'config_algorithm') else "ilp"
            run_algorithm = st.selectbox(
                "Algorithm",
                ["ilp", "flow", "hybrid"],
                index=["ilp", "flow", "hybrid"].index(default_algorithm) if default_algorithm in ["ilp", "flow", "hybrid"] else 0,
                key="run_algorithm"
            )
            # Use saved time limit from config, default to 60 if not set
            default_time_limit = st.session_state.config_time_limit if hasattr(st.session_state, 'config_time_limit') else 60
            run_time_limit = st.slider(
                "Time Limit (seconds)",
                0, 600, default_time_limit,
                key="run_time_limit"
            )
            # Use saved random seed from config
            default_seed = st.session_state.config_random_seed if hasattr(st.session_state, 'config_random_seed') else None
            run_seed = st.number_input(
                "Random Seed (optional)",
                value=default_seed,
                min_value=0,
                key="run_seed"
            )
            
            # Algorithm explanation
            with st.expander("📖 Algorithm Selection Guide"):
                st.write("""
                **Choose the right algorithm for your needs:**
                
                **ILP (Integer Linear Programming)** 🎯
                - **Quality**: ⭐⭐⭐⭐⭐ Optimal (best possible)
                - **Speed**: ⭐☆☆☆☆ Very slow
                - **Time**: 60-300 seconds typical
                - **Use when**: You want the absolute best solution (final production run)
                - **Best for**: Final allocations, when time is not critical
                
                **Flow (Network Flow)** ⚡
                - **Quality**: ⭐⭐⭐⭐☆ Very good
                - **Speed**: ⭐⭐⭐⭐⭐ Very fast
                - **Time**: 1-10 seconds typical
                - **Use when**: You need fast results (testing, iterations)
                - **Best for**: Experimentation, parameter tuning, quick feedback
                
                **Hybrid** 🔄
                - **Quality**: ⭐⭐⭐⭐⭐ Near-optimal
                - **Speed**: ⭐⭐⭐☆☆ Medium
                - **Time**: 10-60 seconds typical
                - **Use when**: You want balance between quality and speed
                - **Best for**: When you have moderate time available
                
                **Time Limit Guidance:**
                - **< 10 sec**: Quick preview (Flow only)
                - **10-60 sec**: Balanced run (Hybrid or Flow)
                - **60-300 sec**: High quality (ILP or Hybrid)
                - **> 300 sec**: Maximum quality (ILP only)
                
                **Tips:**
                - Start with Flow for testing
                - Use Hybrid for final checks
                - Use ILP for production runs
                - Random seed = reproducible results (same seed = same result)
                """)
        
        st.divider()
        
        # Show active configuration settings from Configuration page
        with st.expander("⚙️ Active Configuration Settings (from Configuration Page)"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**Preference Settings:**")
                st.write(f"- Allow Unranked: {st.session_state.config_allow_unranked}")
                st.write(f"- Tier 2 Cost: {st.session_state.config_tier2_cost}")
                st.write(f"- Tier 3 Cost: {st.session_state.config_tier3_cost}")
                st.write(f"- Unranked Cost: {st.session_state.config_unranked_cost}")
                st.write(f"- Top-2 Bias: {st.session_state.config_top2_bias}")
                
                st.write("**Ranked Choice Costs:**")
                st.write(f"- 1st Choice: {st.session_state.config_rank1_cost}")
                st.write(f"- 2nd Choice: {st.session_state.config_rank2_cost}")
                st.write(f"- 3rd Choice: {st.session_state.config_rank3_cost}")
                st.write(f"- 4th Choice: {st.session_state.config_rank4_cost}")
                st.write(f"- 5th Choice: {st.session_state.config_rank5_cost}")
            
            with col2:
                st.write("**Preference Constraints:**")
                min_pref_text = "None" if st.session_state.config_min_pref is None else f"Rank {st.session_state.config_min_pref}"
                max_pref_text = "None" if st.session_state.config_max_pref is None else f"Rank {st.session_state.config_max_pref}"
                excluded_text = "None" if not st.session_state.config_excluded_prefs else f"{st.session_state.config_excluded_prefs}"
                st.write(f"- Min Preference: {min_pref_text}")
                st.write(f"- Max Preference: {max_pref_text}")
                st.write(f"- Excluded Ranks: {excluded_text}")
                
                st.write("**Capacity Settings:**")
                st.write(f"- Topic Overflow: {st.session_state.config_enable_topic_overflow}")
                st.write(f"- Coach Overflow: {st.session_state.config_enable_coach_overflow}")
                st.write(f"- Dept Min Mode: {st.session_state.config_dept_min_mode}")
                st.write(f"- Dept Max Mode: {st.session_state.config_dept_max_mode}")
                st.write(f"- Dept Shortfall Penalty: {st.session_state.config_P_dept_shortfall}")
                st.write(f"- Dept Overflow Penalty: {st.session_state.config_P_dept_overflow}")
                st.write(f"- Topic Penalty: {st.session_state.config_P_topic}")
                st.write(f"- Coach Penalty: {st.session_state.config_P_coach}")
            
            with col3:
                st.write("**Solver Settings:**")
                st.write(f"- Algorithm: {st.session_state.config_algorithm}")
                time_limit_text = "No limit" if st.session_state.config_time_limit == 0 else f"{st.session_state.config_time_limit}s"
                st.write(f"- Time Limit: {time_limit_text}")
                seed_text = "Random" if st.session_state.config_random_seed is None else str(st.session_state.config_random_seed)
                st.write(f"- Random Seed: {seed_text}")
                epsilon_text = "Optimal only" if st.session_state.config_epsilon == 0.0 else f"{st.session_state.config_epsilon:.1%}"
                st.write(f"- Epsilon Suboptimal: {epsilon_text}")
            
            st.write("💡 **Tip**: Go to ⚙️ Configuration page to change these settings, then return here to run allocation with new settings.")
        
        st.divider()
        
        # Validation section
        if (students_file and capacities_file) or (st.session_state.students_file.exists() and st.session_state.capacities_file.exists()):
            st.subheader("✓ Validation")
            col1, col2 = st.columns(2)
            
            with col1:
                if students_file:
                    students_df = pd.read_csv(students_file)
                    st.write(f"**Students:** {len(students_df)} records (uploaded file)")
                    st.write(f"**Columns:** {', '.join(students_df.columns.tolist()[:5])}")
                else:
                    # Use default file
                    students_df = pd.read_csv(st.session_state.students_file)
                    st.write(f"**Students:** {len(students_df)} records (default file)")
                st.write(f"**Columns:** {', '.join(students_df.columns.tolist()[:5])}")
            
            with col2:
                if capacities_file:
                    capacities_df = pd.read_csv(capacities_file)
                    st.write(f"**Capacities:** {len(capacities_df)} records (uploaded file)")
                    st.write(f"**Columns:** {', '.join(capacities_df.columns.tolist()[:5])}")
                else:
                    # Use default file
                    capacities_df = pd.read_csv(st.session_state.capacities_file)
                    st.write(f"**Capacities:** {len(capacities_df)} records (default file)")
                st.write(f"**Columns:** {', '.join(capacities_df.columns.tolist()[:5])}")
            
            st.divider()
            
            # Run button with status
            if st.button("▶️ Run Allocation", key="run_btn", type="primary"):
                with st.spinner("🔄 Running allocation..."):
                    try:
                        # Save uploaded files temporarily
                        import tempfile
                        from pathlib import Path
                        
                        with tempfile.TemporaryDirectory() as tmpdir:
                            students_path = Path(tmpdir) / "students.csv"
                            capacities_path = Path(tmpdir) / "capacities.csv"
                            output_path = Path(tmpdir) / "allocation.csv"
                            summary_path = Path(tmpdir) / "summary.txt"
                            
                            # Use uploaded files or default files
                            if students_file is not None:
                                students_path.write_text(students_file.getvalue().decode())
                                st.info(f"📄 Using uploaded students file")
                            else:
                                # Copy default students file
                                import shutil
                                shutil.copy2(st.session_state.students_file, students_path)
                                st.info(f"📄 Using default students file: {st.session_state.students_file}")
                            
                            if capacities_file is not None:
                                capacities_path.write_text(capacities_file.getvalue().decode())
                                st.info(f"📄 Using uploaded capacities file")
                            else:
                                # Copy default capacities file
                                import shutil
                                shutil.copy2(st.session_state.capacities_file, capacities_path)
                                st.info(f"📄 Using default capacities file: {st.session_state.capacities_file}")
                            
                            # Load data
                            st.info("📂 Loading data...")
                            # Ensure DataRepository is available in this scope
                            from allocator.data_repository import DataRepository
                            repo = DataRepository(
                                str(students_path),
                                str(capacities_path),
                                str(Path(tmpdir) / "overrides.csv") if overrides_file else None
                            )
                            if overrides_file:
                                (Path(tmpdir) / "overrides.csv").write_text(overrides_file.getvalue().decode())
                            repo.load()
                            
                            st.info(f"✓ Loaded {len(repo.students)} students, {len(repo.topics)} topics")
                            
                            # Validate
                            st.info("🔍 Validating data...")
                            validator = InputValidator()
                            is_valid, validation_results = validator.validate_all(
                                repo.students, repo.topics, repo.coaches, repo.departments
                            )
                            
                            if not is_valid:
                                st.error("❌ Validation failed!")
                                for result in validation_results:
                                    if result.severity == "error":
                                        st.error(str(result))
                            else:
                                st.success("✓ Validation passed")
                                
                                # Build preference model
                                st.info("🎯 Building preference model...")
                                from allocator.preference_model import PreferenceModel, PreferenceModelConfig
                                pref_model = PreferenceModel(
                                    topics=repo.topics,
                                    overrides=repo.overrides,
                                    cfg=PreferenceModelConfig(
                                        allow_unranked=st.session_state.config_allow_unranked,
                                        tier2_cost=st.session_state.config_tier2_cost,
                                        tier3_cost=st.session_state.config_tier3_cost,
                                        unranked_cost=st.session_state.config_unranked_cost,
                                        top2_bias=st.session_state.config_top2_bias,
                                        rank1_cost=st.session_state.config_rank1_cost,
                                        rank2_cost=st.session_state.config_rank2_cost,
                                        rank3_cost=st.session_state.config_rank3_cost,
                                        rank4_cost=st.session_state.config_rank4_cost,
                                        rank5_cost=st.session_state.config_rank5_cost
                                    )
                                )
                                
                                # Debug: Show actual ranked choice costs being used
                                st.info(f"🔍 **Debug - Ranked Choice Costs:**")
                                st.write(f"- 1st Choice: {st.session_state.config_rank1_cost}")
                                st.write(f"- 2nd Choice: {st.session_state.config_rank2_cost}")
                                st.write(f"- 3rd Choice: {st.session_state.config_rank3_cost}")
                                st.write(f"- 4th Choice: {st.session_state.config_rank4_cost}")
                                st.write(f"- 5th Choice: {st.session_state.config_rank5_cost}")
                                st.write(f"- Top-2 Bias: {st.session_state.config_top2_bias}")
                                
                                # Debug: Test the PreferenceModel's _rank_cost method
                                st.info(f"🧪 **Debug - Testing _rank_cost method:**")
                                test_costs = []
                                for rank in [1, 2, 3, 4, 5]:
                                    cost = pref_model._rank_cost(rank)
                                    test_costs.append(f"Rank {rank}: {cost}")
                                st.write(" | ".join(test_costs))
                                
                                # Create allocation config using settings from Configuration page
                                legacy_cfg = LegacyAllocationConfig(
                                    pref_cfg=PreferenceModelConfig(
                                        allow_unranked=st.session_state.config_allow_unranked,
                                        tier2_cost=st.session_state.config_tier2_cost,
                                        tier3_cost=st.session_state.config_tier3_cost,
                                        unranked_cost=st.session_state.config_unranked_cost,
                                        top2_bias=st.session_state.config_top2_bias,
                                        rank1_cost=st.session_state.config_rank1_cost,
                                        rank2_cost=st.session_state.config_rank2_cost,
                                        rank3_cost=st.session_state.config_rank3_cost,
                                        rank4_cost=st.session_state.config_rank4_cost,
                                        rank5_cost=st.session_state.config_rank5_cost
                                    ),
                                    dept_min_mode=st.session_state.config_dept_min_mode,
                                    dept_max_mode=st.session_state.config_dept_max_mode,
                                    enable_topic_overflow=st.session_state.config_enable_topic_overflow,
                                    enable_coach_overflow=st.session_state.config_enable_coach_overflow,
                                    P_dept_shortfall=st.session_state.config_P_dept_shortfall,
                                    P_dept_overflow=st.session_state.config_P_dept_overflow,
                                    P_topic=st.session_state.config_P_topic,
                                    P_coach=st.session_state.config_P_coach,
                                    time_limit_sec=run_time_limit if run_time_limit > 0 else None,
                                    random_seed=run_seed if run_seed and run_seed > 0 else None,
                                    epsilon_suboptimal=None,
                                    # ✅ CRITICAL: Pass preference rank constraints to allocation config!
                                    min_acceptable_preference_rank=st.session_state.config_min_pref,
                                    max_acceptable_preference_rank=st.session_state.config_max_pref,
                                    excluded_preference_ranks=st.session_state.config_excluded_prefs if st.session_state.config_excluded_prefs else None
                                )
                                
                                # Build model
                                st.info(f"🔨 Building {run_algorithm.upper()} model...")
                                if run_algorithm == "ilp":
                                    model = AllocationModelILP(
                                        students=repo.students,
                                        topics=repo.topics,
                                        coaches=repo.coaches,
                                        departments=repo.departments,
                                        pref_model=pref_model,
                                        cfg=legacy_cfg
                                    )
                                elif run_algorithm == "flow":
                                    model = AllocationModelFlow(
                                        students=repo.students,
                                        topics=repo.topics,
                                        coaches=repo.coaches,
                                        departments=repo.departments,
                                        pref_model=pref_model,
                                        cfg=legacy_cfg
                                    )
                                else:  # hybrid
                                    model = AllocationModelILP(
                                        students=repo.students,
                                        topics=repo.topics,
                                        coaches=repo.coaches,
                                        departments=repo.departments,
                                        pref_model=pref_model,
                                        cfg=legacy_cfg
                                    )
                                
                                # Solve
                                st.info("⚡ Solving...")
                                model.build()
                                rows, diagnostics = model.solve()
                                
                                # Results
                                st.success("✅ Allocation complete!")
                                
                                # Store repo in session state FIRST (needed for summary generation)
                                st.session_state.last_repos = repo
                                st.session_state.last_allocation_rows = rows
                                
                                # Display results
                                st.divider()
                                st.subheader("📊 Results")
                                
                                # Metrics
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("Students Assigned", len(rows))
                                with col2:
                                    obj_value = diagnostics.get("objective_value", "N/A")
                                    st.metric("Optimal Cost", obj_value)
                                with col3:
                                    first_choice = len([r for r in rows if 10 <= r.preference_rank <= 14])
                                    pct = (first_choice / len(rows) * 100) if rows else 0
                                    st.metric("Got Choice %", f"{pct:.1f}%")
                                with col4:
                                    status = "✓ Success" if diagnostics.get("unassigned_after_solve", 1) == 0 else "⚠ Partial"
                                    st.metric("Status", status)
                                
                                # Explain status
                                st.info("""
                                📌 **Status Explanation:**
                                • **✓ Success**: All students were successfully assigned to a topic
                                • **⚠ Partial**: Some students could NOT be assigned (constraints too tight)
                                  - Check if topic/coach capacity is exceeded
                                  - Try enabling "Topic Overflow" or "Coach Overflow"
                                  - Or relax "Department Min Mode" to "soft"
                                """)
                                
                                # Allocation table
                                st.divider()
                                st.subheader("📋 Allocation Details")
                                allocation_df = pd.DataFrame([
                                    {
                                        'student': row.student,
                                        'assigned_topic': row.assigned_topic,
                                        'assigned_coach': row.assigned_coach,
                                        'department_id': row.department_id,
                                        'preference_rank': row.preference_rank,
                                        'effective_cost': row.effective_cost
                                    }
                                    for row in rows
                                ])
                                st.dataframe(allocation_df, use_container_width=True)
                                
                                # Add column explanations
                                with st.expander("📖 Column Explanations"):
                                    st.write("""
                                    **Student**: The student ID/name
                                    
                                    **Assigned Topic**: The thesis topic this student was allocated to
                                    
                                    **Assigned Coach**: The coach/supervisor for this topic
                                    
                                    **Department ID**: The department this topic belongs to
                                    
                                    **Preference Rank**: How much the student wanted this topic
                                    - **10-14**: Ranked choice (1st-5th choice in preference list)
                                    - **0-2**: Tier preference (general category preference)
                                    - **999**: Unranked (topic not in preferences)
                                    - **-1**: Forced assignment (assigned by system constraint)
                                    - **Lower number = Better preference match**
                                    
                                    **Effective Cost**: The numerical cost of this assignment
                                    - **Lower cost = Student is happier** ✅
                                    - **Higher cost = Student is less satisfied** ❌
                                    - Calculated based on how much student wanted this topic
                                    - Cost of 10 = Got 1st choice (excellent!)
                                    - Cost of 500+ = Got a non-preferred topic (poor)
                                    """)
                                
                                # Create summary text using the proper format from outputs.py
                                import io
                                summary_buffer = io.StringIO()
                                
                                # Write summary to a string buffer using the real write_summary_txt logic
                                from collections import Counter
                                total_assigned = len(rows)
                                pref_counts = Counter(r.preference_rank for r in rows)
                                
                                used_per_topic = Counter(r.assigned_topic for r in rows)
                                used_per_coach = Counter(r.assigned_coach for r in rows)
                                used_per_dept = Counter(r.department_id for r in rows)
                                
                                # Build proper summary
                                summary_text = f"Solver status: {diagnostics.get('status', 'Unknown')}\n"
                                summary_text += f"Objective: {diagnostics.get('objective_value', 'N/A')}\n\n"
                                
                                unassignable = diagnostics.get("unassignable_students", [])
                                unassigned_after = diagnostics.get("unassigned_after_solve", [])
                                summary_text += f"Unassignable students (no admissible topics): {len(unassignable)}\n"
                                if unassignable:
                                    for e in unassignable:
                                        summary_text += f"  - {e}\n"
                                summary_text += f"\nUnassigned after solve: {len(unassigned_after)}\n"
                                if unassigned_after:
                                    for e in unassigned_after:
                                        summary_text += f"  - {e}\n"
                                
                                # Uniqueness check
                                tied = diagnostics.get("tied_students", [])
                                summary_text += f"\n--- SOLUTION UNIQUENESS ---\n"
                                if not tied:
                                    summary_text += "✓ Solution appears UNIQUE (no ties in costs).\n"
                                else:
                                    summary_text += f"⚠ Solution may NOT be unique: {len(tied)} student(s) have equally-good alternatives:\n"
                                
                                # Preference satisfaction
                                summary_text += "\nPreference satisfaction:\n"
                                summary_text += f"  Tier1: {pref_counts.get(0, 0)}\n"
                                summary_text += f"  Tier2: {pref_counts.get(1, 0)}\n"
                                summary_text += f"  Tier3: {pref_counts.get(2, 0)}\n"
                                
                                summary_text += "\nRanked choice satisfaction:\n"
                                summary_text += f"  1st choice: {pref_counts.get(10, 0)}\n"
                                summary_text += f"  2nd choice: {pref_counts.get(11, 0)}\n"
                                summary_text += f"  3rd choice: {pref_counts.get(12, 0)}\n"
                                summary_text += f"  4th choice: {pref_counts.get(13, 0)}\n"
                                summary_text += f"  5th choice: {pref_counts.get(14, 0)}\n"
                                summary_text += f"  Unranked : {pref_counts.get(999, 0)}\n"
                                
                                # Topic utilization
                                summary_text += "\nTopic utilization:\n"
                                topic_over = diagnostics.get("topic_overflow", {})
                                for tid in sorted(st.session_state.last_repos.topics.keys()):
                                    t = st.session_state.last_repos.topics[tid]
                                    used = used_per_topic.get(tid, 0)
                                    ov = topic_over.get(tid, 0)
                                    summary_text += f"  {tid}: {used} / {t.topic_cap}" + (f"  (overflow={ov})" if ov else "") + "\n"
                                
                                # Coach utilization
                                summary_text += "\nCoach utilization:\n"
                                coach_over = diagnostics.get("coach_overflow", {})
                                for cid in sorted(st.session_state.last_repos.coaches.keys()):
                                    c = st.session_state.last_repos.coaches[cid]
                                    used = used_per_coach.get(cid, 0)
                                    ov = coach_over.get(cid, 0)
                                    summary_text += f"  {cid}: {used} / {c.coach_cap}" + (f"  (overflow={ov})" if ov else "") + "\n"
                                
                                # Department totals
                                summary_text += "\nDepartment totals:\n"
                                dept_short = diagnostics.get("department_shortfall", {})
                                for did in sorted(st.session_state.last_repos.departments.keys()):
                                    d = st.session_state.last_repos.departments[did]
                                    used = used_per_dept.get(did, 0)
                                    line = f"  {did}: {used}"
                                    if d.desired_min:
                                        line += f" (desired_min={d.desired_min}"
                                        if dept_short:
                                            line += f", shortfall={dept_short.get(did, 0)}"
                                        line += ")"
                                    summary_text += line + "\n"
                                
                                # Download buttons
                                st.divider()
                                download_combined_results(allocation_df, summary_text)
                                
                                # Store in session for Results Analysis
                                st.session_state.last_allocation = allocation_df
                                st.session_state.last_summary = summary_text
                                st.session_state.last_allocation_timestamp = datetime.now().isoformat()
                                
                                st.info("💡 Go to 📊 Results Analysis to view visualizations!")
                        
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                        import traceback
                        st.error(traceback.format_exc())
        else:
            if not st.session_state.students_file.exists() or not st.session_state.capacities_file.exists():
                st.warning("👆 Default files not found. Please upload students.csv and capacities.csv files to begin")
            else:
                st.warning("👆 Default files are available but validation failed. Please check the file formats or upload custom files.")
    
    # Results Analysis Page
    elif page == "📊 Results Analysis":
        st.header("📊 Results Analysis")
        st.info("""
        📈 **How to interpret these visualizations:**
        - **Key Metrics**: Quick overview of allocation quality
        - **Preference Chart**: Shows how many students got their ranked choices
        - **Department Pie**: Shows student distribution across departments
        - **Capacity Bars**: Shows how full each topic is vs its capacity
        """)
        
        # Check if cached data is available
        if st.session_state.last_allocation is not None and st.session_state.last_summary is not None:
            st.success("✅ Using cached results from recent allocation")
            allocation_df = st.session_state.last_allocation
            summary_text = st.session_state.last_summary
            
            # Display summary metrics
            st.divider()
            st.subheader("📊 Key Metrics")
            create_allocation_summary_metrics(allocation_df, summary_text)
            
            # Display charts
            st.divider()
            st.subheader("📈 Visualizations")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = create_preference_satisfaction_chart(summary_text)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = create_department_distribution_chart(summary_text)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            
            fig = create_capacity_utilization_chart(summary_text)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # Allocation details table
            st.divider()
            st.subheader("📋 Allocation Details")
            st.dataframe(allocation_df, use_container_width=True)
            
            # Download results
            st.divider()
            st.subheader("📥 Download Results")
            download_combined_results(allocation_df, summary_text)
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📥 Input Files")
                students_file = st.file_uploader("Students CSV", type=['csv'], key="results_students")
                capacities_file = st.file_uploader("Capacities CSV", type=['csv'], key="results_capacities")
            
            with col2:
                st.subheader("📤 Allocation Results")
                allocation_file = st.file_uploader("Allocation CSV", type=['csv'], key="results_allocation")
                summary_file = st.file_uploader("Summary TXT", type=['txt'], key="results_summary")
            
                if not (allocation_file and summary_file):
                    st.warning("👆 Upload allocation CSV and summary TXT files, or run allocation first")
                    st.stop()
            
            # Load data
            allocation_df = pd.read_csv(allocation_file)
            summary_text = summary_file.read().decode("utf-8")
            
            # Display summary metrics
            st.divider()
            st.subheader("📊 Key Metrics")
            create_allocation_summary_metrics(allocation_df, summary_text)
            
            # Display charts
            st.divider()
            st.subheader("📈 Visualizations")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = create_preference_satisfaction_chart(summary_text)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = create_department_distribution_chart(summary_text)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            
            fig = create_capacity_utilization_chart(summary_text)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # Allocation details table
            st.divider()
            st.subheader("📋 Allocation Details")
            st.dataframe(allocation_df, use_container_width=True)
            
            # Download results
            st.divider()
            st.subheader("📥 Download Results")
            download_combined_results(allocation_df, summary_text)
    
    # Data Explorer Page
    elif page == "🔍 Data Explorer":
        st.header("🔍 Data Explorer")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("📊 Students")
            students_file = st.file_uploader("Upload Students CSV", type=['csv'], key="explorer_students")
            if students_file:
                df = pd.read_csv(students_file)
                st.write(f"**Total Students:** {len(df)}")
                st.dataframe(df.head(10), use_container_width=True)
        
        with col2:
            st.subheader("🎯 Topics")
            capacities_file = st.file_uploader("Upload Capacities CSV", type=['csv'], key="explorer_capacities")
            if capacities_file:
                df = pd.read_csv(capacities_file)
                st.write(f"**Total Topics:** {len(df)}")
                st.dataframe(df.head(10), use_container_width=True)
        
        with col3:
            st.subheader("👥 Coaches")
            if capacities_file:
                df = pd.read_csv(capacities_file)
                coaches_df = df[['coach_id', 'maximum students per coach']].drop_duplicates('coach_id')
                st.write(f"**Total Coaches:** {len(coaches_df)}")
                st.dataframe(coaches_df.head(10), use_container_width=True)
    
    # Advanced Charts Page
    elif page == "📈 Advanced Charts":
        st.header("📈 Advanced Charts")
        
        st.info("""
        🔍 **Advanced Visualizations:**
        - **Sankey Diagram**: Flow from Students → Topics → Coaches → Departments (colored by preference rank)
        - **Network Flow**: Network graph showing allocations as connections
        - **Cost Matrix**: Heatmap of costs for student-topic pairs
        - **Statistics**: Histograms of costs and preference ranks
        """)
        
        # Explanation of Preference Rank
        with st.expander("📚 What is Preference Rank?", expanded=False):
            st.markdown("""
            ### 🎯 **Preference Rank Explained**
            
            **Preference Rank** is a numeric value that indicates how satisfied a student is with their assigned topic. It's calculated based on how the student ranked that topic in their preferences.
            
            ### 📊 **Preference Rank Values:**
            
            **🟢 Excellent Satisfaction (Tiers):**
            - **0** = Tier 1 (most preferred topics - indifference group)
            - **1** = Tier 2 (moderately preferred topics - indifference group)  
            - **2** = Tier 3 (least preferred topics - indifference group)
            
            **🟡 Good Satisfaction (Ranked Preferences):**
            - **10** = 1st choice (student's #1 ranked topic)
            - **11** = 2nd choice (student's #2 ranked topic)
            - **12** = 3rd choice (student's #3 ranked topic)
            - **13** = 4th choice (student's #4 ranked topic)
            - **14** = 5th choice (student's #5 ranked topic)
            
            **🔴 Poor Satisfaction:**
            - **999** = Unranked topic (student didn't rank this topic at all)
            
            **⚡ Special Cases:**
            - **-1** = Forced assignment (student was manually assigned to this topic)
            
            ### 🎨 **Color Coding in Visualizations:**
            - **Green** = Low preference rank (0-2, 10-11) = Happy students
            - **Yellow** = Medium preference rank (12-13) = Moderately satisfied
            - **Red** = High preference rank (14, 999) = Unhappy students
            
            ### 💡 **How to Interpret:**
            - **Lower numbers = Better satisfaction** (0 is best, 999 is worst)
            - **Tiers (0-2) are always considered good** regardless of configuration
            - **Ranked preferences (10-14) depend on student's actual ranking**
            - **Unranked (999) means student didn't care about this topic**
        """)
        
        # Sankey Visualization
        st.divider()
        st.subheader("🌊 Sankey Diagram (Student → Topic → Coach → Department)")
        st.write("Shows the flow of allocations with colors representing preference satisfaction (green=good, red=bad). **Preference Rank Values**: 0-2=Tiers (excellent), 10-14=Ranked choices (10=1st, 11=2nd, etc.), 999=Unranked (poor), -1=Forced.")
        
        # Check if cached data is available
        if st.session_state.last_allocation_rows is not None and st.session_state.last_repos is not None:
            st.info("✅ Using cached Sankey data from recent allocation.")
            try:
                rows = st.session_state.last_allocation_rows
                
                # Convert namedtuple rows to dictionaries for create_sankey_html
                rows_dicts = [
                    {
                        'student': row.student,
                        'assigned_topic': row.assigned_topic,
                        'assigned_coach': row.assigned_coach,
                        'department_id': row.department_id,
                        'preference_rank': str(row.preference_rank),
                        'effective_cost': str(row.effective_cost)
                    }
                    for row in rows
                ]
                
                # Create Sankey HTML - save to temp file first
                from viz_sankey_enhanced import create_sankey_html as create_sankey
                import tempfile
                
                with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as tmp:
                    temp_sankey_path = tmp.name
                
                # create_sankey_html saves to file and returns path
                _ = create_sankey(rows_dicts, temp_sankey_path)
                
                # Read the generated HTML
                with open(temp_sankey_path, 'r') as f:
                    sankey_html = f.read()
                
                # Save to visualizations folder
                sankey_path = save_visualization(sankey_html, "sankey_diagram.html")
                st.success(f"✓ Sankey saved to: {sankey_path}")
                
                # Create clickable link to open file
                col1, col2 = st.columns([3, 1])
                with col1:
                    file_url = sankey_path.resolve()
                    st.markdown(f"""
                    <a href="file://{file_url}" target="_blank" style="text-decoration: none; color: #1f77b4;">
                        📊 <strong>Open Sankey Diagram in Browser (opens in new tab)</strong>
                    </a>
                    """, unsafe_allow_html=True)
                with col2:
                    if st.button("🔍 View", key="view_sankey_graph"):
                        st.info(f"File location: `{file_url}`")
                
                # Display as HTML
                st.components.v1.html(sankey_html, height=800, scrolling=True)
                
                # Clean up temp file
                import os
                try:
                    os.remove(temp_sankey_path)
                except:
                    pass
                    
            except Exception as e:
                st.warning(f"⚠️ Could not generate Sankey diagram: {str(e)}")
                import traceback
                st.error(traceback.format_exc())
        else:
            sankey_file = st.file_uploader("Allocation CSV for Sankey", type=['csv'], key="sankey_alloc")
            if sankey_file:
                try:
                    import tempfile
                    from pathlib import Path
                    
                    # Write uploaded file to temp location
                    with tempfile.TemporaryDirectory() as tmpdir:
                        temp_csv = Path(tmpdir) / "temp_allocation.csv"
                        temp_csv.write_text(sankey_file.getvalue().decode())
                        
                        # Import and run Sankey generator
                        from viz_sankey_enhanced import load_allocation, create_sankey_html as create_sankey
                        
                        rows = load_allocation(str(temp_csv))
                        
                        # Create temp output file
                        temp_sankey_path = Path(tmpdir) / "sankey_temp.html"
                        _ = create_sankey(rows, str(temp_sankey_path))
                            
                            # Read generated HTML
                        sankey_html = temp_sankey_path.read_text()
                        
                        # Save to visualizations folder
                        sankey_path = save_visualization(sankey_html, "sankey_diagram.html")
                        st.success(f"✓ Sankey saved to: {sankey_path}")
                        
                        # Create clickable link to open file
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            file_url = sankey_path.resolve()
                            st.markdown(f"""
                            <a href="file://{file_url}" target="_blank" style="text-decoration: none; color: #1f77b4;">
                                📊 <strong>Open Sankey Diagram in Browser (opens in new tab)</strong>
                            </a>
                            """, unsafe_allow_html=True)
                        with col2:
                            if st.button("🔍 View", key="view_sankey_upload"):
                                st.info(f"File location: `{file_url}`")
                        
                        # Display as HTML
                        st.components.v1.html(sankey_html, height=800, scrolling=True)
                        
                except Exception as e:
                    st.warning(f"⚠️ Could not generate Sankey diagram: {str(e)}")
                    import traceback
                    st.error(traceback.format_exc())
                else:
                    st.warning("Please upload an Allocation CSV file or run an allocation first to generate a Sankey diagram.")
        
        st.divider()
        
        # Network Flow Visualization
        st.subheader("🕸️ Network Flow Graph")
        st.write("Shows the network structure of allocations (Students ↔ Topics ↔ Coaches ↔ Departments). Edge colors represent preference satisfaction. **Preference Rank Values**: 0-2=Tiers (excellent), 10-14=Ranked choices (10=1st, 11=2nd, etc.), 999=Unranked (poor), -1=Forced.")
        
        # Check if cached data is available
        if st.session_state.last_allocation_rows is not None and st.session_state.last_repos is not None:
            st.info("✅ Using cached Network data from recent allocation.")
            try:
                rows = st.session_state.last_allocation_rows
                
                # Convert namedtuple rows to dictionaries
                rows_dicts = [
                    {
                        'student': row.student,
                        'assigned_topic': row.assigned_topic,
                        'assigned_coach': row.assigned_coach,
                        'department_id': row.department_id,
                        'preference_rank': str(row.preference_rank),
                        'effective_cost': str(row.effective_cost)
                    }
                    for row in rows
                ]
                
                # Import and run Network visualization
                from viz_network_flow import create_network_visualization
                import tempfile
                
                # Create temp file
                with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as tmp:
                    temp_network_path = tmp.name
                
                # create_network_visualization saves to file and returns path
                _ = create_network_visualization(rows_dicts, temp_network_path)
                
                # Read the generated HTML
                with open(temp_network_path, 'r') as f:
                    network_html = f.read()
                
                # Save to visualizations folder
                network_path = save_visualization(network_html, "network_flow.html")
                st.success(f"✓ Network graph saved to: {network_path}")
                
                # Create clickable link to open file
                col1, col2 = st.columns([3, 1])
                with col1:
                    file_url = network_path.resolve()
                    st.markdown(f"""
                    <a href="file://{file_url}" target="_blank" style="text-decoration: none; color: #1f77b4;">
                        🕸️ <strong>Open Network Flow in Browser (opens in new tab)</strong>
                    </a>
                    """, unsafe_allow_html=True)
                with col2:
                    if st.button("🔍 View", key="view_network_graph"):
                        st.info(f"File location: `{file_url}`")
                
                # Display as HTML
                st.components.v1.html(network_html, height=800, scrolling=True)
                
                # Clean up temp file
                import os
                try:
                    os.remove(temp_network_path)
                except:
                    pass
                    
            except Exception as e:
                st.warning(f"⚠️ Could not generate network diagram: {str(e)}")
                import traceback
                st.error(traceback.format_exc())
        else:
            network_file = st.file_uploader("Allocation CSV for Network", type=['csv'], key="network_alloc")
            if network_file:
                try:
                    import tempfile
                    from pathlib import Path
                    
                    # Write uploaded file to temp location
                    with tempfile.TemporaryDirectory() as tmpdir:
                        temp_csv = Path(tmpdir) / "temp_allocation.csv"
                        temp_csv.write_text(network_file.getvalue().decode())
                        
                        # Import and run Network visualization
                        from viz_network_flow import load_allocation, create_network_visualization
                        
                        rows = load_allocation(str(temp_csv))
                            
                            # Create temp output file
                        temp_network_path = Path(tmpdir) / "network_temp.html"
                        _ = create_network_visualization(rows, str(temp_network_path))
                        
                        # Read generated HTML
                        network_html = temp_network_path.read_text()
                        
                        # Save to visualizations folder
                        network_path = save_visualization(network_html, "network_flow.html")
                        st.success(f"✓ Network graph saved to: {network_path}")
                        
                        # Create clickable link to open file
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            file_url = network_path.resolve()
                            st.markdown(f"🕸️ **[Open Network Flow in Browser](file://{file_url})**", unsafe_allow_html=True)
                        with col2:
                            if st.button("🔍 View", key="view_network_upload"):
                                st.info(f"File location: `{file_url}`")
                        
                        # Display as HTML
                        st.components.v1.html(network_html, height=800, scrolling=True)
                            
                except Exception as e:
                    st.warning(f"⚠️ Could not generate network diagram: {str(e)}")
                    import traceback
                    st.error(traceback.format_exc())
                else:
                    st.warning("Please upload an Allocation CSV file or run an allocation first to generate a Network Flow diagram.")
            
            st.divider()
            
            # Cost Heatmap
        st.subheader("🔥 Cost Matrix Heatmaps")
        
        # Explanation of effective cost
        with st.expander("ℹ️ What is Effective Cost?"):
            st.write("""
            **Effective Cost** is a numerical measure of how well each student-topic assignment matches the student's preferences:
            
            - **Lower cost = Better match** ✅ Student got their preference
            - **Higher cost = Worse match** ❌ Student got a less preferred topic
            
            **Cost calculation:**
            - 1st ranked choice: Low cost (student wants this most)
            - 2nd ranked choice: Slightly higher cost
            - 3rd, 4th, 5th ranked choices: Increasing cost
            - Unranked topics: Very high cost (not in student's preferences)
            - Tier preferences: Intermediate costs
            
            **In the heatmap:**
            - 🟢 Green = Low cost (good assignment)
            - 🟡 Yellow = Medium cost (acceptable)
            - 🔴 Red = High cost (poor assignment)
            
            The allocation algorithm tries to **minimize total cost** across all students.
            """)
        
        st.write("Shows effective cost for each student-topic pair (darker red = higher cost/worse fit)")
        
        # Check if we have cached allocation first
        if st.session_state.last_allocation is not None:
            st.info("✅ Using cached allocation data")
            df = st.session_state.last_allocation
        else:
            allocation_file = st.file_uploader("Allocation CSV for Heatmaps", type=['csv'], key="heatmap_alloc")
            df = None
        if allocation_file:
                df = pd.read_csv(allocation_file)
                
        if df is not None:
            try:
                # TAB 1: STUDENT × TOPIC HEATMAP (ALL STUDENTS)
                st.subheader("📊 Student × Topic Heatmap (All Students)")
                
                if 'effective_cost' in df.columns and 'student' in df.columns and 'assigned_topic' in df.columns:
                    # Use ALL students, not just first 50
                    cost_pivot = df.pivot_table(
                        values='effective_cost',
                        index='student',
                        columns='assigned_topic',
                        fill_value=0,
                        aggfunc='first'
                    )
                    
                    fig = go.Figure(data=go.Heatmap(
                        z=cost_pivot.values,
                        x=cost_pivot.columns,
                        y=cost_pivot.index,
                        colorscale='RdYlGn_r',
                        hovertemplate='Student: %{y}<br>Topic: %{x}<br>Cost: %{z}<extra></extra>'
                    ))
                    fig.update_layout(
                        title=f"Effective Cost Heatmap - All {len(df)} Students",
                        height=max(600, len(df) * 10),  # Dynamic height based on student count
                        xaxis_title="Topic",
                        yaxis_title="Student",
                        xaxis_tickangle=-45
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.write(f"*Showing all {len(df)} students and their assigned topics*")
                else:
                    st.warning("Required columns not found. Need: 'student', 'assigned_topic', 'effective_cost'")
                
                # TAB 2: COACH × TOPIC HEATMAP
                st.divider()
                st.subheader("👥 Coach × Topic Heatmap")
                
                if 'assigned_coach' in df.columns and 'assigned_topic' in df.columns and 'effective_cost' in df.columns:
                    st.write("Shows total cost impact of each coach-topic combination")
                    
                    # Create aggregated data: for each coach-topic pair, sum the costs
                    coach_topic_cost = df.groupby(['assigned_coach', 'assigned_topic'])['effective_cost'].agg(['sum', 'count']).reset_index()
                    
                    cost_pivot_coach = coach_topic_cost.pivot_table(
                        values='sum',
                        index='assigned_coach',
                        columns='assigned_topic',
                        fill_value=0
                    )
                    
                    fig = go.Figure(data=go.Heatmap(
                        z=cost_pivot_coach.values,
                        x=cost_pivot_coach.columns,
                        y=cost_pivot_coach.index,
                        colorscale='RdYlGn_r',
                        hovertemplate='Coach: %{y}<br>Topic: %{x}<br>Total Cost: %{z}<extra></extra>'
                    ))
                    fig.update_layout(
                        title="Coach × Topic Cost Distribution",
                        height=400,
                        xaxis_title="Topic",
                        yaxis_title="Coach",
                        xaxis_tickangle=-45
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Summary stats
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Coaches", len(coach_topic_cost['assigned_coach'].unique()))
                    with col2:
                        st.metric("Total Topics", len(coach_topic_cost['assigned_topic'].unique()))
                    with col3:
                        st.metric("Total Cost", f"{coach_topic_cost['sum'].sum():.0f}")
                else:
                    st.warning("Required columns not found. Need: 'assigned_coach', 'assigned_topic', 'effective_cost'")
                
                # TAB 3: DEPARTMENT × TOPIC HEATMAP
                st.divider()
                st.subheader("🏛️ Department × Topic Heatmap")
                
                if 'department_id' in df.columns and 'assigned_topic' in df.columns and 'effective_cost' in df.columns:
                    st.write("Shows total cost impact of each department-topic combination")
                    
                    # Create aggregated data: for each department-topic pair, sum the costs
                    dept_topic_cost = df.groupby(['department_id', 'assigned_topic'])['effective_cost'].agg(['sum', 'count']).reset_index()
                    
                    cost_pivot_dept = dept_topic_cost.pivot_table(
                        values='sum',
                        index='department_id',
                        columns='assigned_topic',
                        fill_value=0
                    )
                    
                    fig = go.Figure(data=go.Heatmap(
                        z=cost_pivot_dept.values,
                        x=cost_pivot_dept.columns,
                        y=cost_pivot_dept.index,
                        colorscale='RdYlGn_r',
                        hovertemplate='Department: %{y}<br>Topic: %{x}<br>Total Cost: %{z}<extra></extra>'
                    ))
                    fig.update_layout(
                        title="Department × Topic Cost Distribution",
                        height=400,
                        xaxis_title="Topic",
                        yaxis_title="Department",
                        xaxis_tickangle=-45
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Summary stats
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Departments", len(dept_topic_cost['department_id'].unique()))
                    with col2:
                        st.metric("Total Topics", len(dept_topic_cost['assigned_topic'].unique()))
                    with col3:
                        st.metric("Total Cost", f"{dept_topic_cost['sum'].sum():.0f}")
                else:
                    st.warning("Required columns not found. Need: 'department_id', 'assigned_topic', 'effective_cost'")
                    
            except Exception as e:
                st.warning(f"⚠️ Could not generate heatmaps: {str(e)}")
        elif st.session_state.last_allocation is None:
            st.warning("👆 Upload allocation CSV or run allocation first to generate heatmaps")
        
        st.divider()
        
        # Summary Statistics Section
        st.subheader("📊 Summary Statistics")
        st.write("Complete allocation results and diagnostics")
        
        if st.session_state.last_summary is not None:
            summary_text = st.session_state.last_summary
            
            # DEBUG: Show raw summary for troubleshooting
            with st.expander("🐛 DEBUG: Show Raw Summary (click to expand)"):
                st.text(summary_text)
            
            # Parse summary data
            lines = summary_text.split('\n')
            
            # Create tabs for different sections
            tab1, tab2, tab3, tab4, tab5 = st.tabs(
                ["📌 Overview", "🎯 Preferences", "📚 Topics", "👥 Coaches", "🏛️ Departments"]
            )
            
            # TAB 1: OVERVIEW
            with tab1:
                col1, col2, col3, col4 = st.columns(4)
                
                # Extract key metrics
                total_students = len(st.session_state.last_allocation) if st.session_state.last_allocation is not None else 0
                
                # Extract solver status and objective - more robust
                objective_value = "N/A"
                unassignable = 0
                unassigned = 0
                
                for i, line in enumerate(lines):
                    line_lower = line.lower()
                    
                    # Match the actual format from outputs.py: "Objective: 3061.0"
                    if line.startswith("Objective:"):
                        try:
                            val_str = line.split("Objective:")[-1].strip()
                            # Extract number (could be float or int)
                            import re
                            match = re.search(r'[\d.]+', val_str)
                            if match:
                                objective_value = match.group()
                        except Exception as e:
                            pass
                    
                    # Match: "Unassignable students (no admissible topics): 0"
                    if "Unassignable students" in line:
                        try:
                            val_str = line.split(":")[-1].strip()
                            unassignable = int(val_str)
                        except:
                            pass
                    
                    # Match: "Unassigned after solve: 0"
                    if "Unassigned after solve:" in line:
                        try:
                            val_str = line.split(":")[-1].strip()
                            unassigned = int(val_str)
                        except:
                            pass
                
                with col1:
                    st.metric("Total Students", total_students)
                with col2:
                    st.metric("Optimal Cost", objective_value)
                with col3:
                    st.metric("Unassignable", unassignable, delta_color="off")
                with col4:
                    st.metric("Unassigned", unassigned, delta_color="off")
                
                # Add explanations
                with st.expander("📖 What Do These Metrics Mean?"):
                    col_exp1, col_exp2 = st.columns(2)
                    
                    with col_exp1:
                        st.write("""
                        **Total Students**: Number of students in allocation
                        
                        **Optimal Cost**: Total "cost" of the allocation
                        - Lower is better ✅
                        - Measured in arbitrary units
                        - Represents how satisfied students are overall
                        - Example: Cost 3061 means students are on average fairly satisfied
                        - Cost includes preference mismatches + penalties + constraints
                        """)
                    
                    with col_exp2:
                        st.write("""
                        **Unassignable**: Students with NO possible topics
                        - These students cannot be assigned to ANY topic
                        - Usually 0 (everyone has some option)
                        - If > 0: Check if topic requirements are too strict
                        
                        **Unassigned**: Students not assigned after solving
                        - The solver couldn't find assignments for these students
                        - Means constraints were too restrictive
                        - If > 0: Try enabling Topic/Coach overflow or relaxing constraints
                        """)
                
                # Check for uniqueness
                st.divider()
                if "UNIQUE" in summary_text.upper():
                    st.success("✓ Solution appears UNIQUE (no ties in costs)")
                    with st.expander("📖 What Does Uniqueness Mean?"):
                        st.write("""
                        **Solution Uniqueness**: Whether this is the ONLY best allocation
                        
                        ✅ **UNIQUE**: This is the best solution. No other allocation has same cost.
                        - Most desirable
                        - You can be confident in this allocation
                        
                        ⚠️ **NOT UNIQUE**: Multiple allocations have equally-good costs
                        - The solver picked one arbitrarily
                        - Alternative allocations might work equally well
                        - Consider running again or tweaking parameters
                        """)
                else:
                    st.info("Solution uniqueness information not found")
            
            # TAB 2: PREFERENCES
            with tab2:
                st.subheader("🎯 Preference Satisfaction Analysis")
                
                # Parse preference satisfaction data from summary
                pref_data = {}
                lines = summary_text.split('\n')
                
                # Find preference satisfaction section
                pref_start = None
                for i, line in enumerate(lines):
                    if "Preference satisfaction:" in line:
                        pref_start = i
                        break
                
                if pref_start is not None:
                    # Parse tier satisfaction
                    for i in range(pref_start + 1, min(pref_start + 10, len(lines))):
                        line = lines[i].strip()
                        if line.startswith("Tier"):
                            if "Tier1:" in line:
                                pref_data["Tier 1"] = int(line.split(":")[1].strip())
                            elif "Tier2:" in line:
                                pref_data["Tier 2"] = int(line.split(":")[1].strip())
                            elif "Tier3:" in line:
                                pref_data["Tier 3"] = int(line.split(":")[1].strip())
                        elif line.startswith("Ranked choice satisfaction:"):
                            break
                    
                    # Parse ranked choice satisfaction
                    for i in range(pref_start + 1, min(pref_start + 15, len(lines))):
                        line = lines[i].strip()
                        if "1st choice:" in line:
                            pref_data["1st Choice"] = int(line.split(":")[1].strip())
                        elif "2nd choice:" in line:
                            pref_data["2nd Choice"] = int(line.split(":")[1].strip())
                        elif "3rd choice:" in line:
                            pref_data["3rd Choice"] = int(line.split(":")[1].strip())
                        elif "4th choice:" in line:
                            pref_data["4th Choice"] = int(line.split(":")[1].strip())
                        elif "5th choice:" in line:
                            pref_data["5th Choice"] = int(line.split(":")[1].strip())
                        elif "Unranked" in line:
                            pref_data["Unranked"] = int(line.split(":")[1].strip())
                
                if pref_data:
                    # Display preference satisfaction metrics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Tier 1 Satisfaction", pref_data.get("Tier 1", 0))
                        st.metric("Tier 2 Satisfaction", pref_data.get("Tier 2", 0))
                        st.metric("Tier 3 Satisfaction", pref_data.get("Tier 3", 0))
                    
                    with col2:
                        st.metric("1st Choice", pref_data.get("1st Choice", 0))
                        st.metric("2nd Choice", pref_data.get("2nd Choice", 0))
                        st.metric("3rd Choice", pref_data.get("3rd Choice", 0))
                    
                    with col3:
                        st.metric("4th Choice", pref_data.get("4th Choice", 0))
                        st.metric("5th Choice", pref_data.get("5th Choice", 0))
                        st.metric("Unranked", pref_data.get("Unranked", 0))
                    
                    # Calculate and display satisfaction percentages
                    total_students = sum(pref_data.values())
                    if total_students > 0:
                        st.divider()
                        st.subheader("📊 Satisfaction Breakdown")
                        
                        # Tier satisfaction
                        tier_total = pref_data.get("Tier 1", 0) + pref_data.get("Tier 2", 0) + pref_data.get("Tier 3", 0)
                        tier_percentage = (tier_total / total_students) * 100
                        
                        # Ranked satisfaction (1st-3rd choices)
                        ranked_total = pref_data.get("1st Choice", 0) + pref_data.get("2nd Choice", 0) + pref_data.get("3rd Choice", 0)
                        ranked_percentage = (ranked_total / total_students) * 100
                        
                        # Poor satisfaction (4th-5th choices + unranked)
                        poor_total = pref_data.get("4th Choice", 0) + pref_data.get("5th Choice", 0) + pref_data.get("Unranked", 0)
                        poor_percentage = (poor_total / total_students) * 100
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric(
                                "Excellent Satisfaction (Tiers)",
                                f"{tier_percentage:.1f}%",
                                f"{tier_total} students"
                            )
                        
                        with col2:
                            st.metric(
                                "Good Satisfaction (1st-3rd Choice)",
                                f"{ranked_percentage:.1f}%",
                                f"{ranked_total} students"
                            )
                        
                        with col3:
                            st.metric(
                                "Poor Satisfaction (4th+ Choice)",
                                f"{poor_percentage:.1f}%",
                                f"{poor_total} students"
                            )
                        
                        # Preference rank explanation
                        with st.expander("📚 Preference Rank Values Reference"):
                            st.markdown("""
                            **Preference Rank Values:**
                            - **0-2**: Tiers (excellent satisfaction)
                            - **10-14**: Ranked choices (10=1st, 11=2nd, etc.)
                            - **999**: Unranked (poor satisfaction)
                            - **-1**: Forced assignment
                            
                            **Lower numbers = Better satisfaction**
                            """)
                        
                        # Satisfaction analysis
                        st.divider()
                        st.subheader("🎯 Satisfaction Analysis")
                        
                        if tier_percentage > 50:
                            st.success(f"✅ Excellent! {tier_percentage:.1f}% of students got tier preferences (indifference groups)")
                        elif ranked_percentage > 70:
                            st.success(f"✅ Good! {ranked_percentage:.1f}% of students got their top 3 choices")
                        elif poor_percentage > 30:
                            st.warning(f"⚠️ Room for improvement: {poor_percentage:.1f}% of students got poor satisfaction (4th+ choice or unranked)")
                        else:
                            st.info(f"📊 Balanced allocation: {ranked_percentage:.1f}% good satisfaction, {poor_percentage:.1f}% poor satisfaction")
                
                else:
                    st.warning("Preference satisfaction data not found in summary")
                    st.info("This data will be available after running an allocation")

            # TAB 3: TOPICS
            with tab3:
                topics_data = {}
                topic_start = None
                
                for i, line in enumerate(lines):
                    if "topic utilization:" in line.lower():
                        topic_start = i
                        break
                
                if topic_start is not None:
                    for i in range(topic_start + 1, len(lines)):
                        line = lines[i]
                        if line.strip() == "" or not line.startswith('  '):
                            break
                        if "topic" in line.lower():
                            try:
                                parts = line.split(':')
                                if len(parts) == 2:
                                    topic_name = parts[0].strip()
                                    util_parts = parts[1].split('/')
                                    if len(util_parts) == 2:
                                        used = int(util_parts[0].strip())
                                        total = int(util_parts[1].strip())
                                        pct = (used / total * 100) if total > 0 else 0
                                        topics_data[topic_name] = {'used': used, 'total': total, 'pct': pct}
                            except:
                                pass
                
                if topics_data:
                    topic_names = list(topics_data.keys())
                    used_counts = [topics_data[t]['used'] for t in topic_names]
                    total_counts = [topics_data[t]['total'] for t in topic_names]
                    
                    fig = go.Figure(data=[
                        go.Bar(name='Used', x=topic_names, y=used_counts, marker_color='#27ae60', opacity=0.8),
                        go.Bar(name='Capacity', x=topic_names, y=total_counts, marker_color='#e74c3c', opacity=0.6)
                    ])
                    fig.update_layout(
                        title="Topic Utilization (Green=Used, Red=Capacity)",
                        xaxis_title="Topic",
                        yaxis_title="Students",
                        barmode='group',
                        height=400,
                        xaxis_tickangle=-45,
                        hovermode='x unified'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.divider()
                    st.write("**Utilization Details:**")
                    table_data = []
                    for topic, data in topics_data.items():
                        table_data.append({
                            'Topic': topic,
                            'Used': data['used'],
                            'Capacity': data['total'],
                            'Utilization %': f"{data['pct']:.1f}%"
                        })
                    st.dataframe(pd.DataFrame(table_data), use_container_width=True)
                else:
                    st.info(f"No topic utilization data found (looked around line {topic_start})")
            
            # TAB 4: COACHES
            with tab4:
                coaches_data = {}
                coach_start = None
                
                for i, line in enumerate(lines):
                    if "coach utilization:" in line.lower():
                        coach_start = i
                        break
                
                if coach_start is not None:
                    for i in range(coach_start + 1, len(lines)):
                        line = lines[i]
                        if line.strip() == "" or not line.startswith('  '):
                            break
                        if "coach" in line.lower():
                            try:
                                parts = line.split(':')
                                if len(parts) == 2:
                                    coach_name = parts[0].strip()
                                    util_parts = parts[1].split('/')
                                    if len(util_parts) == 2:
                                        used = int(util_parts[0].strip())
                                        total = int(util_parts[1].strip())
                                        pct = (used / total * 100) if total > 0 else 0
                                        coaches_data[coach_name] = {'used': used, 'total': total, 'pct': pct}
                            except:
                                pass
                
                if coaches_data:
                    coach_names = list(coaches_data.keys())
                    used_counts = [coaches_data[c]['used'] for c in coach_names]
                    total_counts = [coaches_data[c]['total'] for c in coach_names]
                    
                    fig = go.Figure(data=[
                        go.Bar(name='Used', x=coach_names, y=used_counts, marker_color='#3498db', opacity=0.8),
                        go.Bar(name='Capacity', x=coach_names, y=total_counts, marker_color='#e67e22', opacity=0.6)
                    ])
                    fig.update_layout(
                        title="Coach Utilization (Blue=Used, Orange=Capacity)",
                        xaxis_title="Coach",
                        yaxis_title="Students",
                        barmode='group',
                        height=400,
                        xaxis_tickangle=-45,
                        hovermode='x unified'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.divider()
                    st.write("**Utilization Details:**")
                    table_data = []
                    for coach, data in coaches_data.items():
                        table_data.append({
                            'Coach': coach,
                            'Used': data['used'],
                            'Capacity': data['total'],
                            'Utilization %': f"{data['pct']:.1f}%"
                        })
                    st.dataframe(pd.DataFrame(table_data), use_container_width=True)
                else:
                    st.info(f"No coach utilization data found (looked around line {coach_start})")
            
            # TAB 5: DEPARTMENTS
            with tab5:
                dept_data = {}
                dept_start = None
                
                for i, line in enumerate(lines):
                    if "department total" in line.lower():
                        dept_start = i
                        break
                
                if dept_start is not None:
                    for i in range(dept_start + 1, len(lines)):
                        line = lines[i]
                        if line.strip() == "" or not line.startswith('  '):
                            break
                        if "department" in line.lower():
                            try:
                                parts = line.split(':')
                                if len(parts) >= 2:
                                    dept_name = parts[0].strip()
                                    count_str = parts[1].split('(')[0].strip()
                                    count = int(count_str)
                                    dept_data[dept_name] = count
                            except:
                                pass
                
                if dept_data:
                    col1, col2 = st.columns([1.2, 1])
                    with col1:
                        fig = px.pie(
                            values=list(dept_data.values()),
                            names=list(dept_data.keys()),
                            title="Student Distribution by Department",
                            color_discrete_sequence=px.colors.qualitative.Set3
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        st.write("**Department Totals:**")
                        table_data = []
                        for dept, count in dept_data.items():
                            table_data.append({'Department': dept, 'Students': count})
                        st.dataframe(pd.DataFrame(table_data), use_container_width=True)
                else:
                    st.info(f"No department data found (looked around line {dept_start})")
        else:
            st.warning("👆 Run allocation first to view summary statistics")
        
        st.divider()
        
        # Fairness Metrics Section
        st.subheader("⚖️ Fairness Metrics")
        st.write("Analyze allocation fairness using statistical measures and ethical distribution metrics")
        
        # Comprehensive fairness explanation
        with st.expander("📖 Understanding Fairness Metrics (Click to Learn)"):
            tab_explain1, tab_explain2, tab_explain3 = st.tabs(
                ["🎯 What is Fairness?", "📊 Gini Coefficient", "⚖️ Fairness Score"]
            )
            
            with tab_explain1:
                st.write("""
                **Fairness in Allocation** means:
                - Every student gets considered fairly
                - No group is systematically disadvantaged
                - Satisfaction is distributed equitably
                - Load on coaches/topics is reasonable
                
                **Why It Matters:**
                - ✅ Ethical responsibility to treat all students equally
                - ✅ Reduces conflicts and complaints
                - ✅ Ensures legitimate allocations
                - ✅ Builds trust in the system
                
                **What We Measure:**
                1. **Cost Fairness**: Are some students getting much worse assignments than others?
                2. **Preference Satisfaction**: What % of students got choices they wanted?
                3. **Load Balance**: Are coaches/topics distributed fairly?
                4. **Overall Score**: Combined measure of all fairness dimensions
                """)
            
            with tab_explain2:
                st.write("""
                **Gini Coefficient** is a standard measure of inequality (from economics)
                
                **Scale: 0 to 1**
                - **0** = Perfect equality (everyone treated exactly the same)
                - **0.3** = Very fair (minor differences)
                - **0.5** = Moderately fair (some inequality)
                - **0.7** = Unfair (large differences)
                - **1** = Perfect inequality (one person has everything)
                
                **In Allocation Context:**
                
                **Example 1 - Fair Allocation** 🟢
                - All students get costs: 10, 12, 11, 13
                - Gini = 0.08 (very fair!)
                - Everyone is equally happy
                
                **Example 2 - Unfair Allocation** 🔴
                - Students get costs: 10, 10, 10, 1000
                - Gini = 0.62 (very unfair!)
                - Most are happy but one is very upset
                
                **Formula:**
                Gini = (2 * Σ(i * x_i)) / (n * Σx_i) - (n + 1) / n
                
                Where x_i are values in sorted order.
                """)
            
            with tab_explain3:
                st.write("""
                **Fairness Score** combines multiple metrics into 0-100 scale
                
                **Components & Weights:**
                - **40%** - Preference Satisfaction (most important!)
                  - How many students got their ranked choices?
                - **20%** - Cost Fairness (using Gini coefficient)
                  - Are some students treated much worse?
                - **15%** - Topic Load Balance
                  - Are topics equally filled?
                - **15%** - Coach Load Balance
                  - Are coaches equally loaded?
                - **10%** - Department Load Balance
                  - Are departments equally filled?
                
                **Score Interpretation:**
                - **80-100 ✅**: Excellent - Allocation is ethically sound
                  - Students are satisfied and treated fairly
                  - Recommend proceeding with this allocation
                  
                - **60-79 ⚠️**: Good - Improvements possible
                  - Acceptable but some issues identified
                  - Consider tweaking parameters
                  
                - **Below 60 ❌**: Poor - Needs review
                  - Significant fairness issues detected
                  - Suggest relaxing constraints or adjusting weights
                
                **Target**: Aim for 75+ for balanced allocations
                """)
        
        # Check if we have cached allocation
        if st.session_state.last_allocation is not None:
            st.info("✅ Using cached allocation data")
            df = st.session_state.last_allocation
            
            try:
                # Calculate fairness metrics
                metrics = calculate_fairness_score(df)
                
                # Display main fairness metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    gini_cost = metrics.get('gini_cost', 0)
                    st.metric(
                        "Cost Fairness (Gini)",
                        f"{gini_cost:.3f}",
                        delta="← Lower is fairer" if gini_cost < 0.5 else "← Higher = more unequal",
                        delta_color="inverse"
                    )
                    st.caption("0=Perfect equality, 1=Complete inequality")
                
                with col2:
                    topic_balance = metrics.get('topic_balance', 0)
                    st.metric(
                        "Topic Load Balance",
                        f"{topic_balance:.1%}",
                        delta="← Higher is more balanced" if topic_balance > 0.5 else "← Topics imbalanced",
                        delta_color="normal"
                    )
                    st.caption("How evenly topics are filled")
                
                with col3:
                    coach_balance = metrics.get('coach_balance', 0)
                    st.metric(
                        "Coach Load Balance",
                        f"{coach_balance:.1%}",
                        delta="← Higher is more balanced",
                        delta_color="normal"
                    )
                    st.caption("How evenly coaches are assigned")
                
                with col4:
                    ranked_sat = metrics.get('ranked_satisfaction', 0)
                    st.metric(
                        "Preference Satisfaction",
                        f"{ranked_sat:.1%}",
                        delta="← Students getting ranked choices",
                        delta_color="normal"
                    )
                    st.caption("Students satisfied with assignment")
                
                st.divider()
                
                # Detailed fairness analysis
                tab_f1, tab_f2, tab_f3, tab_f4 = st.tabs(
                    ["📊 Cost Distribution", "🎯 Preference Fairness", "⚖️ Load Balance", "📊 Normalized Load Balance"]
                )
                
                # TAB 1: Cost Distribution
                with tab_f1:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Cost Statistics:**")
                        cost_stats = {
                            'Mean Cost': f"{metrics.get('cost_mean', 0):.2f}",
                            'Median Cost': f"{metrics.get('cost_median', 0):.2f}",
                            'Std Dev': f"{metrics.get('cost_std', 0):.2f}",
                            'Coeff of Variation': f"{metrics.get('cost_cv', 0):.3f}",
                            'Gini Coefficient': f"{metrics.get('gini_cost', 0):.3f}",
                        }
                        stats_df = pd.DataFrame(list(cost_stats.items()), columns=['Metric', 'Value'])
                        st.dataframe(stats_df, use_container_width=True)
                        
                        st.write("""
                        **Interpretation:**
                        - **Gini < 0.3**: Very fair distribution of costs
                        - **Gini 0.3-0.5**: Moderately fair
                        - **Gini > 0.5**: Unequal distribution (some students get bad matches)
                        
                        **Coefficient of Variation:**
                        - **< 0.5**: Low variability (consistent fairness)
                        - **> 1.0**: High variability (unfair allocation)
                        """)
                    
                    with col2:
                        # Cost distribution histogram
                        costs = df['effective_cost'].values
                        fig = go.Figure()
                        fig.add_trace(go.Histogram(
                            x=costs,
                            nbinsx=30,
                            marker_color='#3498db',
                            name='Students',
                            opacity=0.7
                        ))
                        fig.add_vline(
                            x=np.mean(costs),
                            line_dash="dash",
                            line_color="red",
                            name=f"Mean: {np.mean(costs):.2f}"
                        )
                        fig.add_vline(
                            x=np.median(costs),
                            line_dash="dash",
                            line_color="green",
                            name=f"Median: {np.median(costs):.2f}"
                        )
                        fig.update_layout(
                            title="Cost Distribution Across Students",
                            xaxis_title="Effective Cost",
                            yaxis_title="Number of Students",
                            height=400,
                            showlegend=True
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                # TAB 2: Preference Fairness
                with tab_f2:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Preference rank distribution
                        pref_dist = df['preference_rank'].value_counts().sort_index()
                        
                        # Categorize preferences
                        ranked_students = len(df[df['preference_rank'].between(10, 14)])
                        tier_students = len(df[df['preference_rank'].between(0, 2)])
                        unranked_students = len(df[df['preference_rank'] == 999])
                        forced_students = len(df[df['preference_rank'] == -1])
                        other_students = len(df) - ranked_students - tier_students - unranked_students - forced_students
                        
                        pref_categories = {
                            '🎯 Got Ranked Choice': ranked_students,
                            '⭐ Got Tier Preference': tier_students,
                            '❌ Got Unranked': unranked_students,
                            '🔧 Forced Assignment': forced_students,
                            '❓ Other': other_students
                        }
                        
                        fig = px.pie(
                            values=list(pref_categories.values()),
                            names=list(pref_categories.keys()),
                            title="Student Satisfaction Distribution",
                            color_discrete_sequence=['#27ae60', '#3498db', '#e74c3c', '#f39c12', '#95a5a6']
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        st.write("**Preference Satisfaction Summary:**")
                        total = len(df)
                        pref_summary = {
                            'Metric': [
                                '✅ Got Ranked Choice (1st-5th)',
                                '⭐ Got Tier Preference',
                                '❌ Got Unranked Topic',
                                '🔧 Forced Assignment',
                                '📊 Total Students'
                            ],
                            'Count': [
                                ranked_students,
                                tier_students,
                                unranked_students,
                                forced_students,
                                total
                            ],
                            'Percentage': [
                                f"{ranked_students/total*100:.1f}%" if total > 0 else "0%",
                                f"{tier_students/total*100:.1f}%" if total > 0 else "0%",
                                f"{unranked_students/total*100:.1f}%" if total > 0 else "0%",
                                f"{forced_students/total*100:.1f}%" if total > 0 else "0%",
                                "100%"
                            ]
                        }
                        pref_df = pd.DataFrame(pref_summary)
                        st.dataframe(pref_df, use_container_width=True)
                        
                        st.write("""
                        **Fairness Interpretation:**
                        - **Got Ranked Choice > 70%**: Very fair ✅
                        - **Got Ranked Choice 50-70%**: Acceptable ⚠️
                        - **Got Ranked Choice < 50%**: Unfair ❌
                        """)
                
                # TAB 3: Load Balance
                with tab_f3:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Topic load balance
                        if 'assigned_topic' in df.columns:
                            topic_counts = df['assigned_topic'].value_counts().sort_index()
                            
                            fig = go.Figure()
                            fig.add_trace(go.Bar(
                                x=topic_counts.index,
                                y=topic_counts.values,
                                marker_color='#3498db',
                                name='Students'
                            ))
                            fig.add_hline(
                                y=topic_counts.mean(),
                                line_dash="dash",
                                line_color="red",
                                annotation_text=f"Avg: {topic_counts.mean():.1f}"
                            )
                            fig.update_layout(
                                title=f"Topic Load Balance (Gini: {metrics.get('gini_topics', 0):.3f})",
                                xaxis_title="Topic",
                                yaxis_title="Students Assigned",
                                height=400,
                                xaxis_tickangle=-45
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Coach load balance
                        if 'assigned_coach' in df.columns:
                            coach_counts = df['assigned_coach'].value_counts().sort_index()
                            
                            fig = go.Figure()
                            fig.add_trace(go.Bar(
                                x=coach_counts.index,
                                y=coach_counts.values,
                                marker_color='#e67e22',
                                name='Students'
                            ))
                            fig.add_hline(
                                y=coach_counts.mean(),
                                line_dash="dash",
                                line_color="red",
                                annotation_text=f"Avg: {coach_counts.mean():.1f}"
                            )
                            fig.update_layout(
                                title=f"Coach Load Balance (Gini: {metrics.get('gini_coaches', 0):.3f})",
                                xaxis_title="Coach",
                                yaxis_title="Students Assigned",
                                height=400,
                                xaxis_tickangle=-45
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    
                    st.divider()
                    st.write("**Load Balance Interpretation:**")
                    balance_col1, balance_col2, balance_col3 = st.columns(3)
                    
                    with balance_col1:
                        st.write(f"""
                        **Topic Balance: {metrics.get('topic_balance', 0):.1%}**
                        
                        Shows if topics are equally loaded.
                        
                        - **> 80%**: Topics well balanced ✅
                        - **60-80%**: Acceptable ⚠️
                        - **< 60%**: Imbalanced ❌
                        """)
                    
                    with balance_col2:
                        st.write(f"""
                        **Coach Balance: {metrics.get('coach_balance', 0):.1%}**
                        
                        Shows if coaches are equally loaded.
                        
                        - **> 80%**: Coaches well balanced ✅
                        - **60-80%**: Acceptable ⚠️
                        - **< 60%**: Imbalanced ❌
                        """)
                    
                    with balance_col3:
                        st.write(f"""
                        **Dept Balance: {metrics.get('dept_balance', 0):.1%}**
                        
                        Shows if departments are equally filled.
                        
                        - **> 80%**: Depts well balanced ✅
                        - **60-80%**: Acceptable ⚠️
                        - **< 60%**: Imbalanced ❌
                        """)
                
                # TAB 4: Normalized Load Balance
                with tab_f4:
                    st.write("**Normalized Load Balance** - Shows assignments as % of capacity")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Topic normalized balance (% of capacity)
                        if 'assigned_topic' in df.columns:
                            topic_counts = df['assigned_topic'].value_counts()
                            
                            # Get topic capacities from repo (stored in session state)
                            try:
                                if st.session_state.last_repos:
                                    repo = st.session_state.last_repos
                                    topic_data = []
                                    for topic_id in sorted(repo.topics.keys()):
                                        topic = repo.topics[topic_id]
                                        assigned = topic_counts.get(topic_id, 0)
                                        capacity = topic.topic_cap
                                        normalized = (assigned / capacity * 100) if capacity > 0 else 0
                                        topic_data.append({
                                            'Topic': topic_id,
                                            'Assigned': assigned,
                                            'Capacity': capacity,
                                            'Usage %': normalized
                                        })
                                    
                                    topic_norm_df = pd.DataFrame(topic_data)
                                    
                                    fig = px.bar(
                                        topic_norm_df,
                                        x='Topic',
                                        y='Usage %',
                                        title='Topic Utilization (% of Capacity)',
                                        color='Usage %',
                                        color_continuous_scale='RdYlGn_r',
                                        labels={'Usage %': 'Capacity Used (%)'},
                                        hover_data={'Assigned': True, 'Capacity': True}
                                    )
                                    fig.add_hline(y=100, line_dash="dash", line_color="red", annotation_text="100% Full")
                                    fig.add_hline(y=80, line_dash="dash", line_color="orange", annotation_text="80% Utilization")
                                    fig.update_layout(height=400, xaxis_tickangle=-45)
                                    st.plotly_chart(fig, use_container_width=True)
                                else:
                                    st.info("Repository data not available")
                            except Exception as e:
                                st.warning(f"Could not load topic capacities: {e}")
                    
                    with col2:
                        # Coach normalized balance (% of capacity)
                        if 'assigned_coach' in df.columns:
                            coach_counts = df['assigned_coach'].value_counts()
                            
                            try:
                                if st.session_state.last_repos:
                                    repo = st.session_state.last_repos
                                    coach_data = []
                                    for coach_id in sorted(repo.coaches.keys()):
                                        coach = repo.coaches[coach_id]
                                        assigned = coach_counts.get(coach_id, 0)
                                        capacity = coach.coach_cap
                                        normalized = (assigned / capacity * 100) if capacity > 0 else 0
                                        coach_data.append({
                                            'Coach': coach_id,
                                            'Assigned': assigned,
                                            'Capacity': capacity,
                                            'Usage %': normalized
                                        })
                                    
                                    coach_norm_df = pd.DataFrame(coach_data)
                                    
                                    fig = px.bar(
                                        coach_norm_df,
                                        x='Coach',
                                        y='Usage %',
                                        title='Coach Utilization (% of Capacity)',
                                        color='Usage %',
                                        color_continuous_scale='RdYlGn_r',
                                        labels={'Usage %': 'Capacity Used (%)'},
                                        hover_data={'Assigned': True, 'Capacity': True}
                                    )
                                    fig.add_hline(y=100, line_dash="dash", line_color="red", annotation_text="100% Full")
                                    fig.add_hline(y=80, line_dash="dash", line_color="orange", annotation_text="80% Utilization")
                                    fig.update_layout(height=400, xaxis_tickangle=-45)
                                    st.plotly_chart(fig, use_container_width=True)
                                else:
                                    st.info("Repository data not available")
                            except Exception as e:
                                st.warning(f"Could not load coach capacities: {e}")
                    
                    st.divider()
                    st.write("""
                    **Normalized Load Balance Interpretation:**
                    
                    This view shows how much each topic/coach is utilized **relative to its capacity**.
                    
                    - **100%+**: Topic/Coach is overbooked (if overflow enabled)
                    - **80-100%**: Well utilized ✅
                    - **50-80%**: Adequate utilization ⚠️
                    - **< 50%**: Underutilized capacity
                    """)
                
                # Overall Fairness Score
                st.divider()
                st.subheader("📈 Overall Fairness Score")
                
                # Calculate composite fairness score
                scores = []
                
                # Cost fairness (inverse Gini, 0-100)
                cost_fairness_score = (1 - metrics.get('gini_cost', 0)) * 100
                scores.append(('Cost Fairness', cost_fairness_score))
                
                # Preference satisfaction (already percentage)
                pref_fairness_score = metrics.get('ranked_satisfaction', 0) * 100
                scores.append(('Preference Satisfaction', pref_fairness_score))
                
                # Topic balance (already percentage)
                topic_score = metrics.get('topic_balance', 0) * 100
                scores.append(('Topic Load Balance', topic_score))
                
                # Coach balance
                coach_score = metrics.get('coach_balance', 0) * 100
                scores.append(('Coach Load Balance', coach_score))
                
                # Dept balance
                dept_score = metrics.get('dept_balance', 0) * 100
                scores.append(('Dept Load Balance', dept_score))
                
                # Overall score (weighted average)
                weights = [0.2, 0.4, 0.15, 0.15, 0.1]  # More weight on preferences and costs
                overall_score = sum(s * w for (_, s), w in zip(scores, weights))
                
                # Display scores
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    score_data = {
                        'Fairness Dimension': [s[0] for s in scores] + ['🏆 OVERALL SCORE'],
                        'Score': [f"{s[1]:.1f}/100" for s in scores] + [f"{overall_score:.1f}/100"],
                        'Status': [
                            '✅' if s[1] >= 80 else '⚠️' if s[1] >= 60 else '❌'
                            for s in scores
                        ] + ['✅' if overall_score >= 75 else '⚠️' if overall_score >= 60 else '❌']
                    }
                    score_df = pd.DataFrame(score_data)
                    st.dataframe(score_df, use_container_width=True)
                
                with col2:
                    # Gauge chart for overall score
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=overall_score,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Overall Fairness"},
                        delta={'reference': 75, 'prefix': 'vs Target'},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 60], 'color': "#f8d7da"},
                                {'range': [60, 80], 'color': "#fff3cd"},
                                {'range': [80, 100], 'color': "#d4edda"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 75
                            }
                        }
                    ))
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                st.write("""
                **Fairness Score Interpretation:**
                - **80-100**: Excellent fairness ✅ - Allocation is ethically sound
                - **60-79**: Good fairness ⚠️ - Some improvements possible
                - **Below 60**: Poor fairness ❌ - Consider revising allocation parameters
                """)
                
                # Add solutions for improving fairness
                st.divider()
                st.subheader("💡 How to Improve Fairness")
                
                gini_cost = metrics.get('gini_cost', 0)
                
                if gini_cost > 0.5:
                    st.error(f"⚠️ Cost fairness is low (Gini = {gini_cost:.3f}). High inequality detected!")
                    
                    with st.expander("🔧 Solutions to Improve Cost Fairness"):
                        st.write("""
                        **Your allocation has high cost inequality.** Here are evidence-based solutions:
                        
                        ### 1️⃣ **Increase Topic Capacity (Most Impactful)**
                        - **Problem**: Popular topics are full, forcing some students to unranked alternatives
                        - **Solution**: Add more slots to high-demand topics
                        - **Expected Impact**: ⬇️ Gini by 20-40%, major fairness improvement
                        - **Action**: Review which topics are over-subscribed and increase their capacity
                        
                        ### 2️⃣ **Relax Preference Satisfaction Constraints**
                        - **Problem**: Min/Max preference constraints may force unfair assignments
                        - **Solution**: Remove or loosen constraints to give solver more flexibility
                        - **Expected Impact**: ⬇️ Gini by 10-20%
                        - **Action**: In Configuration page, set Min Preference to "None" and try again
                        
                        ### 3️⃣ **Encourage Diverse Student Rankings**
                        - **Problem**: Many students rank only 1-2 topics, creating bottlenecks
                        - **Solution**: Ask students to rank at least 5-10 different topics
                        - **Expected Impact**: ⬇️ Gini by 15-25%
                        - **Action**: Communicate to students why diverse rankings help everyone
                        
                        ### 4️⃣ **Adjust Unranked Cost Parameter**
                        - **Problem**: Current unranked cost (100) creates huge jumps in fairness
                        - **Solution**: Reduce unranked cost (e.g., from 100 to 50-75)
                        - **Expected Impact**: ⬇️ Gini by 5-15%
                        - **Action**: In Configuration, change "Unranked Topic Cost" to lower value
                        - **Trade-off**: Students might get fewer preferred topics but fairness improves
                        
                        ### 5️⃣ **Increase Coach/Topic Load Penalties**
                        - **Problem**: Load isn't being balanced fairly
                        - **Solution**: Increase P_topic and P_coach penalties in Configuration
                        - **Expected Impact**: ⬇️ Gini by 5-10%
                        - **Action**: Try P_topic=1000 and P_coach=800 instead of current values
                        
                        ### 6️⃣ **Use Tier Preferences Instead of Rankings**
                        - **Problem**: Ranked preferences are fine-grained but inflexible
                        - **Solution**: Group topics into tiers (must-have, prefer, acceptable)
                        - **Expected Impact**: ⬇️ Gini by 10-20%
                        - **Action**: Modify student input to use tier system
                        
                        ### Priority Order for Maximum Impact:
                        1. **Increase capacity** (biggest impact, but requires resources)
                        2. **Diversify student rankings** (high impact, no resources needed)
                        3. **Adjust cost parameters** (medium impact, easy to configure)
                        4. **Relax constraints** (medium impact, may reduce other fairness aspects)
                        """)
                
                elif gini_cost > 0.3:
                    st.warning(f"⚠️ Cost fairness is moderate (Gini = {gini_cost:.3f}). Room for improvement.")
                    
                    with st.expander("🔧 Suggestions to Improve Cost Fairness"):
                        st.write(f"""
                        **Current Fairness Level**: Good but can be improved
                        
                        **Quick Wins:**
                        - Reduce unranked cost (Configuration → Unranked Topic Cost: try 50-75)
                        - Ask a few more students to rank additional topics
                        - Increase topic overflow penalty slightly (P_topic: +100-200)
                        
                        **Target**: Get Gini to < 0.2 for excellent fairness
                        """)
                
                else:
                    st.success(f"✅ Excellent cost fairness (Gini = {gini_cost:.3f})!")
                    st.write("Your allocation has fair cost distribution across students. Well done!")
                
                # Additional recommendations based on other metrics
                st.divider()
                st.subheader("📋 Overall Recommendations")
                
                issues = []
                
                if metrics.get('gini_cost', 0) > 0.3:
                    issues.append("❌ Cost fairness could be improved")
                if metrics.get('topic_balance', 0) < 0.7:
                    issues.append("❌ Topic load is imbalanced")
                if metrics.get('coach_balance', 0) < 0.7:
                    issues.append("❌ Coach load is imbalanced")
                if metrics.get('ranked_satisfaction', 0) < 0.5:
                    issues.append("❌ Less than 50% of students got ranked choices")
                
                if issues:
                    st.write("**Issues detected:**")
                    for issue in issues:
                        st.write(issue)
                    
                    st.write("""
                    **Suggested Next Steps:**
                    1. Review the specific metrics in tabs above to understand root causes
                    2. Try the recommended solutions in priority order
                    3. Re-run allocation with adjusted parameters
                    4. Compare fairness before/after by noting the scores
                    """)
                else:
                    st.success("✅ All fairness metrics are excellent! This is a high-quality allocation.")
                
                st.divider()
            except Exception as e:
                st.error(f"❌ Error calculating fairness metrics: {str(e)}")
                import traceback
                st.error(traceback.format_exc())
        else:
            st.warning("👆 Run allocation first to view fairness metrics")

    # Really Advanced Charts Page
    elif page == "🚀 Really Advanced Charts":
        st.header("🚀 Really Advanced Charts")
        st.write("""
        **Advanced analytical visualizations for comprehensive allocation insights:**
        """)
        
        # Verbose explanations for Really Advanced Charts
        with st.expander("📚 What are Really Advanced Charts?", expanded=False):
            st.markdown("""
            ### 🚀 **Really Advanced Charts Explained**
            
            These are sophisticated analytical visualizations that provide deep insights into your allocation results:
            
            **📊 Preference & Cost Analysis Tab:**
            - **Preference Satisfaction Funnel**: Shows how many students got each preference level (1st choice, 2nd choice, etc.)
            - **Cost Breakdown Pie Chart**: Visualizes the composition of your allocation by preference type
            - **Cost Distribution Violin Plot**: Statistical view showing cost clustering and outliers
            
            **⚖️ Fairness & Capacity Tab:**
            - **Fairness Comparison Radar**: Multi-dimensional fairness metrics compared to benchmarks
            - **Topic Demand vs Capacity**: Identifies bottleneck topics and capacity mismatches
            
            **🎯 Deep Dive Analysis Tab:**
            - **Student Satisfaction Scatter**: Each bubble is a student, showing cost vs preference satisfaction
            - **Coach Specialization Heatmap**: Which coaches handle which topics (workload distribution)
            - **Department Diversity Analysis**: Topic diversity within departments using Shannon Entropy
            
            ### 💡 **How to Use These Charts:**
            1. **Identify Patterns**: Look for clusters, outliers, and trends
            2. **Spot Issues**: Find bottlenecks, imbalances, or unfair distributions
            3. **Optimize**: Use insights to adjust configuration settings
            4. **Compare**: Run multiple allocations to see how changes affect results
            """)
        
        if st.session_state.last_allocation is not None and st.session_state.last_repos is not None:
            st.info("✅ Using cached allocation data for advanced analysis")
            
            try:
                from viz_really_advanced_charts import (
                    create_preference_funnel,
                    create_cost_violin_plot,
                    create_topic_demand_vs_capacity,
                    create_cost_breakdown_pie,
                    create_fairness_radar,
                    create_student_satisfaction_scatter,
                    create_coach_specialization_heatmap,
                    create_department_diversity_analysis
                )
                
                allocation_df = st.session_state.last_allocation
                repo = st.session_state.last_repos
                
                # Create three main tabs for different analysis types
                adv_tab1, adv_tab2, adv_tab3 = st.tabs([
                    "📊 Preference & Cost Analysis",
                    "⚖️ Fairness & Capacity",
                    "🎯 Deep Dive Analysis"
                ])
                
                # TAB 1: Preference & Cost Analysis
                with adv_tab1:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Preference Satisfaction Funnel**")
                        st.write("Distribution of students across satisfaction levels")
                        st.caption("💡 **What this shows**: How many students got each preference level. The 'percent initial' shows what percentage of total students each level represents. **Preference Rank Values**: 0-2=Tiers (excellent), 10-14=Ranked choices (10=1st, 11=2nd, etc.), 999=Unranked (poor), -1=Forced.")
                        try:
                            fig_funnel = create_preference_funnel(allocation_df)
                            st.plotly_chart(fig_funnel, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error generating funnel: {e}")
                    
                    with col2:
                        st.write("**Cost Breakdown by Category**")
                        st.write("Allocation composition by preference type")
                        st.caption("💡 **What this shows**: The proportion of students assigned to each preference category. Helps identify if most students got good preferences or if many got unranked topics. **Preference Rank Values**: 0-2=Tiers (excellent), 10-14=Ranked choices (10=1st, 11=2nd, etc.), 999=Unranked (poor), -1=Forced.")
                        try:
                            fig_pie = create_cost_breakdown_pie(allocation_df)
                            st.plotly_chart(fig_pie, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error generating pie chart: {e}")
                    
                    st.divider()
                    st.write("**Cost Distribution Violin Plot**")
                    st.write("Statistical distribution showing clustering and outliers")
                    st.caption("💡 **What this shows**: The shape of cost distribution across students. Wide sections = many students with similar costs. Narrow sections = few students with those costs. Outliers = students with very different costs.")
                    try:
                        fig_violin = create_cost_violin_plot(allocation_df)
                        st.plotly_chart(fig_violin, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error generating violin plot: {e}")
                
                # TAB 2: Fairness & Capacity
                with adv_tab2:
                    metrics = calculate_fairness_score(allocation_df)
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Fairness Comparison Radar**")
                        st.write("Multi-dimensional fairness metrics vs benchmark")
                        st.caption("💡 **What this shows**: Multiple fairness dimensions compared to ideal benchmarks. Each axis represents a different type of fairness (cost, load balance, etc.). Closer to edge = more fair.")
                        try:
                            fig_radar = create_fairness_radar(metrics)
                            st.plotly_chart(fig_radar, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error generating radar: {e}")
                    
                    with col2:
                        st.write("**Topic Demand vs Capacity**")
                        st.write("Identifies bottleneck topics and capacity mismatches")
                        st.caption("💡 **What this shows**: Each bar shows topic demand (how many students wanted it) vs capacity (how many it can handle). Bars above capacity line = oversubscribed topics. Bars below = undersubscribed topics.")
                        try:
                            fig_capacity = create_topic_demand_vs_capacity(allocation_df, repo)
                            st.plotly_chart(fig_capacity, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error generating capacity chart: {e}")
                
                # TAB 3: Deep Dive Analysis
                with adv_tab3:
                    st.write("**Student Satisfaction Scatter Plot**")
                    st.write("Cost vs Preference Rank - each bubble is a student, colored by department")
                    st.caption("💡 **What this shows**: Each bubble represents one student. X-axis = preference rank (lower is better), Y-axis = effective cost (lower is better). Bottom-left = happy students, top-right = unhappy students. Colors show departments. **Preference Rank Values**: 0-2=Tiers (excellent), 10-14=Ranked choices (10=1st, 11=2nd, etc.), 999=Unranked (poor), -1=Forced.")
                    
                    # Custom axis limits controls
                    col_x, col_y = st.columns(2)
                    with col_x:
                        x_min = st.number_input("X-axis Min (Preference Rank)", value=0, min_value=0, max_value=1000, step=1)
                        x_max = st.number_input("X-axis Max (Preference Rank)", value=200, min_value=0, max_value=1000, step=1)
                    with col_y:
                        y_min = st.number_input("Y-axis Min (Effective Cost)", value=0, min_value=0, max_value=1000, step=1)
                        y_max = st.number_input("Y-axis Max (Effective Cost)", value=200, min_value=0, max_value=1000, step=1)
                    
                    try:
                        fig_scatter = create_student_satisfaction_scatter(allocation_df, repo, x_range=[x_min, x_max], y_range=[y_min, y_max])
                        st.plotly_chart(fig_scatter, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error generating scatter plot: {e}")
                    
                    st.divider()
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Coach Specialization Heatmap**")
                        st.write("Which coaches handle which topics")
                        st.caption("💡 **What this shows**: Heatmap showing coach-topic assignments. Darker colors = more students assigned to that coach-topic combination. Helps identify coach workload distribution and specialization patterns.")
                        try:
                            fig_coach = create_coach_specialization_heatmap(allocation_df, repo)
                            st.plotly_chart(fig_coach, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error generating coach heatmap: {e}")
                    
                    with col2:
                        st.write("**Department Diversity Analysis**")
                        st.write("Topic diversity within departments (Shannon Entropy)")
                        st.caption("💡 **What this shows**: How diverse topics are within each department. Higher bars = more diverse topics (good for department variety). Lower bars = fewer topic types (more specialized departments).")
                        try:
                            fig_dept = create_department_diversity_analysis(allocation_df)
                            st.plotly_chart(fig_dept, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error generating department analysis: {e}")
                
                st.divider()
                st.success("✅ Really Advanced Charts loaded successfully!")
                
                with st.expander("📚 Guide to Really Advanced Charts"):
                    st.write("""
                    **Tab 1: Preference & Cost Analysis**
                    - **Funnel**: Shows how many students at each satisfaction level
                    - **Pie**: Budget allocation by preference type
                    - **Violin**: Distribution shape - shows if costs are clustered or spread
                    
                    **Tab 2: Fairness & Capacity**
                    - **Radar**: Multi-metric comparison against 75-point benchmark
                    - **Capacity**: Highlights over/under-subscribed topics
                    
                    **Tab 3: Deep Dive**
                    - **Scatter**: Each dot is a student; position shows satisfaction
                    - **Coach Heatmap**: Darker = more students; reveals specialization
                    - **Diversity**: Higher entropy = fairer distribution
                    """)
            
            except ImportError as e:
                st.error(f"❌ Could not import Really Advanced Charts module: {e}")
                st.info("Make sure viz_really_advanced_charts.py is in the same directory")
            except Exception as e:
                st.error(f"❌ Error in Really Advanced Charts: {str(e)}")
                import traceback
                st.error(traceback.format_exc())
        else:
            st.warning("👆 Run allocation first to view Really Advanced Charts")

    # Compare Allocations Page
    elif page == "⚖️ Compare Allocations":
        st.header("⚖️ Compare Allocations")
        st.write("""
        **Compare multiple allocation results with different cost configurations to analyze trade-offs and sensitivity.**
        """)
        
        # Explanation
        with st.expander("📚 What is Allocation Comparison?", expanded=True):
            st.markdown("""
            ### ⚖️ **Allocation Comparison Explained**
            
            This feature allows you to run multiple allocations with different cost configurations and compare the results:
            
            **🎯 Why Compare Allocations?**
            - **Sensitivity Analysis**: See how different cost settings affect outcomes
            - **Trade-off Analysis**: Understand trade-offs between fairness, satisfaction, and efficiency
            - **Optimization**: Find the best cost configuration for your specific goals
            - **Validation**: Verify that your cost settings produce expected behavior
            
            **📊 What Gets Compared?**
            - **Preference Satisfaction**: How many students get their 1st, 2nd, 3rd choices
            - **Fairness Metrics**: Gini coefficient, distribution equality
            - **Cost Distribution**: Average costs, cost variance
            - **Capacity Utilization**: Topic/coach utilization rates
            - **Department Balance**: How well departments are balanced
            
            **🔧 How It Works:**
            1. Define multiple cost configurations
            2. Run allocations with each configuration
            3. Compare results side-by-side
            4. Analyze trade-offs and patterns
            """)
        
        # Check if we have cached results
        if st.session_state.last_allocation is None:
            st.warning("👆 Run at least one allocation first to enable comparison")
            st.info("💡 **Tip**: Run an allocation in the 'Run Allocation' page, then come back here to compare it with other configurations.")
            return
        
        st.divider()
        
        # Comparison Configuration
        st.subheader("🔧 Comparison Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📋 Current Configuration (Baseline)**")
            st.write("This is your current cached allocation:")
            
            # Show current configuration
            current_config = {
                "Rank 1 Cost": st.session_state.config_rank1_cost,
                "Rank 2 Cost": st.session_state.config_rank2_cost,
                "Rank 3 Cost": st.session_state.config_rank3_cost,
                "Rank 4 Cost": st.session_state.config_rank4_cost,
                "Rank 5 Cost": st.session_state.config_rank5_cost,
                "Top-2 Bias": st.session_state.config_top2_bias,
                "Unranked Cost": st.session_state.config_unranked_cost
            }
            
            for key, value in current_config.items():
                st.write(f"• **{key}**: {value}")
        
        with col2:
            st.markdown("**🆚 Alternative Configurations**")
            st.write("Define up to 3 alternative configurations to compare:")
            
            # Alternative configurations
            alt_configs = []
            
            for i in range(3):
                with st.expander(f"Configuration {i+1}", expanded=(i==0)):
                    st.write(f"**Alternative {i+1} Settings:**")
                    
                    alt_rank1 = st.slider(f"Rank 1 Cost", 0, 200, 0, key=f"alt{i}_rank1")
                    alt_rank2 = st.slider(f"Rank 2 Cost", 0, 200, 1, key=f"alt{i}_rank2")
                    alt_rank3 = st.slider(f"Rank 3 Cost", 0, 200, 2, key=f"alt{i}_rank3")
                    alt_rank4 = st.slider(f"Rank 4 Cost", 0, 200, 3, key=f"alt{i}_rank4")
                    alt_rank5 = st.slider(f"Rank 5 Cost", 0, 200, 4, key=f"alt{i}_rank5")
                    alt_top2_bias = st.checkbox(f"Top-2 Bias", False, key=f"alt{i}_top2")
                    alt_unranked = st.slider(f"Unranked Cost", 0, 500, 200, key=f"alt{i}_unranked")
                    
                    alt_configs.append({
                        "name": f"Config {i+1}",
                        "rank1_cost": alt_rank1,
                        "rank2_cost": alt_rank2,
                        "rank3_cost": alt_rank3,
                        "rank4_cost": alt_rank4,
                        "rank5_cost": alt_rank5,
                        "top2_bias": alt_top2_bias,
                        "unranked_cost": alt_unranked
                    })
        
        st.divider()
        
        # Run Comparisons
        if st.button("🚀 Run Comparison Allocations", type="primary"):
            if not alt_configs:
                st.warning("Please define at least one alternative configuration")
                return
            
            st.warning("⚠️ **Note**: This feature is under development. For now, please manually run allocations with different configurations in the 'Run Allocation' page and compare the results manually.")
            st.info("💡 **Tip**: You can save different configurations in the 'Configuration' page, run allocations, and compare the results in 'Results Analysis' and 'Advanced Charts'.")
            
            # Add current configuration as baseline
            baseline_result = {
                "name": "Baseline (Current)",
                "config": current_config,
                "allocation": st.session_state.last_allocation,
                "summary": st.session_state.last_summary
            }
            comparison_results.append(baseline_result)
            
            # Run alternative configurations
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, alt_config in enumerate(alt_configs):
                status_text.text(f"Running {alt_config['name']}...")
                progress_bar.progress((i + 1) / len(alt_configs))
                
                try:
                    # Temporarily update session state with alternative config
                    original_rank1 = st.session_state.config_rank1_cost
                    original_rank2 = st.session_state.config_rank2_cost
                    original_rank3 = st.session_state.config_rank3_cost
                    original_rank4 = st.session_state.config_rank4_cost
                    original_rank5 = st.session_state.config_rank5_cost
                    original_top2_bias = st.session_state.config_top2_bias
                    original_unranked = st.session_state.config_unranked_cost
                    
                    # Set alternative config
                    st.session_state.config_rank1_cost = alt_config["rank1_cost"]
                    st.session_state.config_rank2_cost = alt_config["rank2_cost"]
                    st.session_state.config_rank3_cost = alt_config["rank3_cost"]
                    st.session_state.config_rank4_cost = alt_config["rank4_cost"]
                    st.session_state.config_rank5_cost = alt_config["rank5_cost"]
                    st.session_state.config_top2_bias = alt_config["top2_bias"]
                    st.session_state.config_unranked_cost = alt_config["unranked_cost"]
                    
                    # Run allocation (simplified version)
                    from allocator.data_repository import DataRepository
                    from allocator.preference_model import PreferenceModel, PreferenceModelConfig
                    import tempfile
                    from pathlib import Path
                    
                    # Use default files
                    students_path = Path("data/input/students.csv")
                    capacities_path = Path("data/input/capacities.csv")
                    
                    if not students_path.exists() or not capacities_path.exists():
                        st.error("Default input files not found")
                        continue
                    
                    # Load data
                    repo = DataRepository(students_path, capacities_path)
                    
                    # Build preference model with alternative config
                    pref_model = PreferenceModel(
                        topics=repo.topics,
                        overrides=repo.overrides,
                        cfg=PreferenceModelConfig(
                            allow_unranked=st.session_state.config_allow_unranked,
                            tier2_cost=st.session_state.config_tier2_cost,
                            tier3_cost=st.session_state.config_tier3_cost,
                            unranked_cost=alt_config["unranked_cost"],
                            top2_bias=alt_config["top2_bias"],
                            rank1_cost=alt_config["rank1_cost"],
                            rank2_cost=alt_config["rank2_cost"],
                            rank3_cost=alt_config["rank3_cost"],
                            rank4_cost=alt_config["rank4_cost"],
                            rank5_cost=alt_config["rank5_cost"]
                        )
                    )
                    
                    # Create config object for allocation
                    from allocator.config import CapacityConfig, SolverConfig
                    
                    capacity_cfg = CapacityConfig(
                        enable_topic_overflow=st.session_state.config_enable_topic_overflow,
                        enable_coach_overflow=st.session_state.config_enable_coach_overflow,
                        dept_min_mode=st.session_state.config_dept_min_mode,
                        dept_max_mode=st.session_state.config_dept_max_mode,
                        P_dept_shortfall=st.session_state.config_P_dept_shortfall,
                        P_dept_overflow=st.session_state.config_P_dept_overflow,
                        P_topic=st.session_state.config_P_topic,
                        P_coach=st.session_state.config_P_coach
                    )
                    
                    solver_cfg = SolverConfig(
                        algorithm=st.session_state.config_algorithm,
                        time_limit=st.session_state.config_time_limit,
                        random_seed=st.session_state.config_random_seed,
                        epsilon=st.session_state.config_epsilon
                    )
                    
                    legacy_cfg = LegacyAllocationConfig(
                        pref_cfg=PreferenceModelConfig(
                            allow_unranked=st.session_state.config_allow_unranked,
                            tier2_cost=st.session_state.config_tier2_cost,
                            tier3_cost=st.session_state.config_tier3_cost,
                            unranked_cost=alt_config["unranked_cost"],
                            top2_bias=alt_config["top2_bias"],
                            rank1_cost=alt_config["rank1_cost"],
                            rank2_cost=alt_config["rank2_cost"],
                            rank3_cost=alt_config["rank3_cost"],
                            rank4_cost=alt_config["rank4_cost"],
                            rank5_cost=alt_config["rank5_cost"]
                        ),
                        capacity_cfg=capacity_cfg,
                        solver_cfg=solver_cfg,
                        min_pref=st.session_state.config_min_pref,
                        max_pref=st.session_state.config_max_pref,
                        excluded_prefs=st.session_state.config_excluded_prefs
                    )
                    
                    # Run allocation
                    if st.session_state.config_algorithm == "ilp":
                        model = AllocationModelILP(
                            students=repo.students,
                            topics=repo.topics,
                            coaches=repo.coaches,
                            departments=repo.departments,
                            pref_model=pref_model,
                            cfg=legacy_cfg
                        )
                    else:  # flow
                        model = AllocationModelFlow(
                            students=repo.students,
                            topics=repo.topics,
                            coaches=repo.coaches,
                            departments=repo.departments,
                            pref_model=pref_model,
                            cfg=legacy_cfg
                        )
                    
                    result = model.solve()
                    
                    # Store result
                    alt_result = {
                        "name": alt_config["name"],
                        "config": alt_config,
                        "allocation": result.allocation,
                        "summary": result.summary
                    }
                    comparison_results.append(alt_result)
                    
                    # Restore original config
                    st.session_state.config_rank1_cost = original_rank1
                    st.session_state.config_rank2_cost = original_rank2
                    st.session_state.config_rank3_cost = original_rank3
                    st.session_state.config_rank4_cost = original_rank4
                    st.session_state.config_rank5_cost = original_rank5
                    st.session_state.config_top2_bias = original_top2_bias
                    st.session_state.config_unranked_cost = original_unranked
                    
                except Exception as e:
                    st.error(f"Error running {alt_config['name']}: {str(e)}")
                    continue
            
            progress_bar.progress(1.0)
            status_text.text("✅ Comparison complete!")
            
            # Store results in session state
            st.session_state.comparison_results = comparison_results
            
            st.success(f"🎉 Successfully compared {len(comparison_results)} configurations!")
        
        # Display Comparison Results
        st.divider()
        st.subheader("📊 Manual Comparison Guide")
        
        st.write("""
        **How to compare allocations with different cost configurations:**
        
        1. **Set Configuration 1**: Go to "⚙️ Configuration" and set your first cost configuration
        2. **Run Allocation 1**: Go to "🚀 Run Allocation" and run the allocation
        3. **Note Results**: Check "📊 Results Analysis" and "📈 Advanced Charts" for metrics
        4. **Set Configuration 2**: Go back to "⚙️ Configuration" and change the cost settings
        5. **Run Allocation 2**: Go to "🚀 Run Allocation" and run again
        6. **Compare Results**: Compare the metrics between the two runs
        
        **Key Metrics to Compare:**
        - **Preference Satisfaction**: How many students got 1st, 2nd, 3rd choices
        - **Fairness (Gini Coefficient)**: Lower = more fair distribution
        - **Total Cost**: Lower = more efficient allocation
        - **Utilization**: Higher = better resource usage
        """)
        
        # Show current configuration for reference
        st.subheader("📋 Current Configuration")
        st.write("**Your current cost settings:**")
        st.write(f"• **Rank 1 Cost**: {st.session_state.config_rank1_cost}")
        st.write(f"• **Rank 2 Cost**: {st.session_state.config_rank2_cost}")
        st.write(f"• **Rank 3 Cost**: {st.session_state.config_rank3_cost}")
        st.write(f"• **Rank 4 Cost**: {st.session_state.config_rank4_cost}")
        st.write(f"• **Rank 5 Cost**: {st.session_state.config_rank5_cost}")
        st.write(f"• **Top-2 Bias**: {st.session_state.config_top2_bias}")
        st.write(f"• **Unranked Cost**: {st.session_state.config_unranked_cost}")
        
        if st.session_state.last_allocation is not None:
            st.success("✅ You have a cached allocation result. You can now change the configuration and run a new allocation to compare.")
        else:
            st.warning("👆 Run an allocation first to enable comparison.")


if __name__ == "__main__":
    main()
