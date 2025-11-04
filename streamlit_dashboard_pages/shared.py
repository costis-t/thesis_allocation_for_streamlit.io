"""
Shared utilities and initialization for the dashboard pages
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from pathlib import Path
import json
from io import StringIO, BytesIO
from datetime import datetime
import tempfile
import shutil
import statistics
import zipfile
import sys

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


def apply_dark_theme():
    """Apply dark theme styling across all pages"""
    st.markdown("""
    <style>
        /* Dark background theme */
        .stApp {
            background-color: #1e1e1e !important;
        }
        
        /* Main content area */
        .main {
            background-color: #1e1e1e !important;
        }
        
        /* Sidebar */
        .css-1d391kg {
            background-color: #2d2d2d !important;
        }
        
        /* Block containers */
        [data-testid="stBlockContainer"] {
            background-color: #2d2d2d !important;
            padding: 1rem;
            border-radius: 10px;
            margin: 0.5rem 0;
        }
        
        /* Headers and text */
        h1, h2, h3, h4, h5, h6, p, span, div, label {
            color: #e0e0e0 !important;
        }
        
        .main-header {
            font-size: 3em;
            color: #4a9eff !important;
            font-weight: bold;
            margin-bottom: 1em;
        }
        
        .metric-card {
            background-color: #2d2d2d !important;
            padding: 1.5em;
            border-radius: 0.5em;
            border-left: 5px solid #4a9eff !important;
        }
        
        .success-color { color: #4caf50 !important; }
        .warning-color { color: #ff9800 !important; }
        .error-color { color: #f44336 !important; }
        
        /* Markdown text */
        [data-testid="stMarkdownContainer"] {
            color: #e0e0e0 !important;
        }
        
        /* Info boxes */
        .stAlert {
            background-color: #2d2d2d !important;
        }
        
        /* Input widgets */
        .stTextInput > div > div > input {
            background-color: #2d2d2d !important;
            color: #e0e0e0 !important;
        }
        
        .stSelectbox > div > div > select {
            background-color: #2d2d2d !important;
            color: #e0e0e0 !important;
        }
    </style>
    """, unsafe_allow_html=True)


def add_license_to_sidebar():
    """Add GPLv3 license notice to sidebar on all pages"""
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style='text-align: center; font-size: 11px; color: #666; padding: 10px;'>
        <a href='https://www.gnu.org/licenses/gpl-3.0.en.html' target='_blank' style='color: #666; text-decoration: none;'>
            📄 Licensed under GNU GPLv3
        </a>
    </div>
    """, unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state for all pages"""
    # Add license to sidebar on all pages
    try:
        add_license_to_sidebar()
    except:
        pass  # Silently fail if sidebar not available
    
    # Only initialize if not already done (to avoid issues during testing)
    try:
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

        # Initialize default file paths (relative to project root)
        # Get project root (parent of streamlit_dashboard_pages directory)
        if 'students_file' not in st.session_state or 'capacities_file' not in st.session_state:
            # Try to determine project root
            try:
                # If running from streamlit_dashboard_pages/, go up to project root
                project_root = Path(__file__).parent.parent
            except NameError:
                # Fallback for testing
                import os
                project_root = Path(os.getcwd())
            
            if 'students_file' not in st.session_state:
                st.session_state.students_file = project_root / "data/input/students.csv"
            if 'capacities_file' not in st.session_state:
                st.session_state.capacities_file = project_root / "data/input/capacities.csv"

        # Load configuration from file if it exists (in project root)
        try:
            project_root = Path(__file__).parent.parent
        except NameError:
            import os
            project_root = Path(os.getcwd())
        
        config_file = project_root / "config_streamlit.json"
        if config_file.exists():
            try:
                config = AllocationConfig.load_json(str(config_file))
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
                try:
                    st.error(f"Error loading config file: {e}. Using defaults.")
                except:
                    pass

        # Initialize configuration settings with defaults
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

        # Ensure visualizations folder exists (in project root)
        try:
            project_root = Path(__file__).parent.parent
        except NameError:
            import os
            project_root = Path(os.getcwd())
        
        VISUALIZATIONS_DIR = project_root / "visualisations"
        VISUALIZATIONS_DIR.mkdir(exist_ok=True)
    except Exception:
        pass  # Silently handle if session state is not available during testing


def save_visualization(html_content, filename):
    """Save visualization HTML to visualizations folder and return path."""
    # Get project root
    try:
        project_root = Path(__file__).parent.parent
    except NameError:
        import os
        project_root = Path(os.getcwd())
    
    VISUALIZATIONS_DIR = project_root / "visualisations"
    VISUALIZATIONS_DIR.mkdir(exist_ok=True)
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
        allocation_json = json.dumps(allocation_df.to_dict('records'), indent=2)
        st.download_button(
            "📥 Allocation JSON",
            allocation_json,
            "allocation_result.json",
            "application/json"
        )
    
    with col4:
        combined_data = {
            "allocation": allocation_df.to_dict('records'),
            "summary": summary_text
        }
        combined_json = json.dumps(combined_data, indent=2)
        st.download_button(
            "📥 Combined JSON",
            combined_json,
            "allocation_combined.json",
            "application/json"
        )


def calculate_satisfaction_metrics(allocation_df):
    """Calculate satisfaction metrics from allocation results."""
    if allocation_df is None or len(allocation_df) == 0:
        return {}
    
    total_students = len(allocation_df)
    metrics = {}
    
    if 'preference_rank' in allocation_df.columns:
        rank_col = allocation_df['preference_rank']
        
        # Count students by preference rank
        metrics['rank_1'] = int((rank_col == 10).sum())
        metrics['rank_2'] = int((rank_col == 11).sum())
        metrics['rank_3'] = int((rank_col == 12).sum())
        metrics['rank_4'] = int((rank_col == 13).sum())
        metrics['rank_5'] = int((rank_col == 14).sum())
        metrics['rank_6_plus'] = int((rank_col > 14).sum())
        metrics['unranked'] = int((rank_col == 0).sum())
        
        # Calculate percentages
        metrics['percent_rank_1'] = (metrics['rank_1'] / total_students * 100) if total_students > 0 else 0
        metrics['percent_rank_2'] = (metrics['rank_2'] / total_students * 100) if total_students > 0 else 0
        metrics['total_students'] = total_students
    
    return metrics


def calculate_fairness_score(allocation_df):
    """Calculate fairness metrics from allocation results."""
    metrics = calculate_satisfaction_metrics(allocation_df)
    
    if 'total_students' not in metrics or metrics['total_students'] == 0:
        return metrics
    
    # Calculate Gini coefficient for costs
    if 'effective_cost' in allocation_df.columns:
        costs = allocation_df['effective_cost'].tolist()
        
        # Calculate basic statistics
        metrics['cost_mean'] = statistics.mean(costs) if costs else 0
        metrics['cost_median'] = statistics.median(costs) if costs else 0
        metrics['cost_std'] = statistics.stdev(costs) if len(costs) > 1 else 0
        
        # Calculate coefficient of variation
        if metrics['cost_mean'] > 0:
            metrics['cost_cv'] = metrics['cost_std'] / metrics['cost_mean']
        else:
            metrics['cost_cv'] = 0
        
        # Calculate Gini
        if len(costs) > 1:
            costs_sorted = sorted(costs)
            n = len(costs_sorted)
            cumsum = [costs_sorted[0]]
            for i in range(1, n):
                cumsum.append(cumsum[-1] + costs_sorted[i])
            total = cumsum[-1]
            if total > 0:
                gini = (2 * sum((i + 1) * cost for i, cost in enumerate(costs_sorted)) / (n * total)) - (n + 1) / n
                metrics['gini_cost'] = max(0, min(1, gini))
            else:
                metrics['gini_cost'] = 0
        else:
            metrics['gini_cost'] = 0
    else:
        metrics['gini_cost'] = 0
        metrics['cost_mean'] = 0
        metrics['cost_median'] = 0
        metrics['cost_std'] = 0
        metrics['cost_cv'] = 0
    
    # Calculate load balance metrics (topic and coach distribution)
    if 'assigned_topic' in allocation_df.columns:
        topic_counts = allocation_df['assigned_topic'].value_counts()
        if len(topic_counts) > 1:
            mean_load = topic_counts.mean()
            std_load = topic_counts.std()
            metrics['topic_balance'] = max(0, 1 - (std_load / mean_load)) if mean_load > 0 else 0
            # Calculate Gini for topic load balance
            counts = topic_counts.values.tolist()
            counts_sorted = sorted(counts)
            n = len(counts_sorted)
            if n > 1:
                cumsum = [counts_sorted[0]]
                for i in range(1, n):
                    cumsum.append(cumsum[-1] + counts_sorted[i])
                total = cumsum[-1]
                if total > 0:
                    metrics['gini_topics'] = (2 * sum((i + 1) * count for i, count in enumerate(counts_sorted)) / (n * total)) - (n + 1) / n
                else:
                    metrics['gini_topics'] = 0
            else:
                metrics['gini_topics'] = 0
        else:
            metrics['topic_balance'] = 1
            metrics['gini_topics'] = 0
    else:
        metrics['topic_balance'] = 0
        metrics['gini_topics'] = 0
    
    if 'assigned_coach' in allocation_df.columns:
        coach_counts = allocation_df['assigned_coach'].value_counts()
        if len(coach_counts) > 1:
            mean_load = coach_counts.mean()
            std_load = coach_counts.std()
            metrics['coach_balance'] = max(0, 1 - (std_load / mean_load)) if mean_load > 0 else 0
            # Calculate Gini for coach load balance
            counts = coach_counts.values.tolist()
            counts_sorted = sorted(counts)
            n = len(counts_sorted)
            if n > 1:
                cumsum = [counts_sorted[0]]
                for i in range(1, n):
                    cumsum.append(cumsum[-1] + counts_sorted[i])
                total = cumsum[-1]
                if total > 0:
                    metrics['gini_coaches'] = (2 * sum((i + 1) * count for i, count in enumerate(counts_sorted)) / (n * total)) - (n + 1) / n
                else:
                    metrics['gini_coaches'] = 0
            else:
                metrics['gini_coaches'] = 0
        else:
            metrics['coach_balance'] = 1
            metrics['gini_coaches'] = 0
    else:
        metrics['coach_balance'] = 0
        metrics['gini_coaches'] = 0
    
    # Calculate preference satisfaction (ranked choices)
    if 'preference_rank' in allocation_df.columns:
        rank_col = allocation_df['preference_rank']
        # Good ranks: tiers (0-2) and top 3 ranked choices (10-12)
        good_ranks = ((rank_col >= 0) & (rank_col <= 2)) | ((rank_col >= 10) & (rank_col <= 12))
        metrics['ranked_satisfaction'] = good_ranks.sum() / len(allocation_df) if len(allocation_df) > 0 else 0
    else:
        metrics['ranked_satisfaction'] = 0
    
    # Calculate department balance
    if 'department_id' in allocation_df.columns:
        dept_counts = allocation_df['department_id'].value_counts()
        if len(dept_counts) > 1:
            mean_load = dept_counts.mean()
            std_load = dept_counts.std()
            metrics['dept_balance'] = max(0, 1 - (std_load / mean_load)) if mean_load > 0 else 0
            # Calculate Gini for department balance
            counts = dept_counts.values.tolist()
            counts_sorted = sorted(counts)
            n = len(counts_sorted)
            if n > 1:
                cumsum = [counts_sorted[0]]
                for i in range(1, n):
                    cumsum.append(cumsum[-1] + counts_sorted[i])
                total = cumsum[-1]
                if total > 0:
                    metrics['gini_depts'] = (2 * sum((i + 1) * count for i, count in enumerate(counts_sorted)) / (n * total)) - (n + 1) / n
                else:
                    metrics['gini_depts'] = 0
            else:
                metrics['gini_depts'] = 0
        else:
            metrics['dept_balance'] = 1
            metrics['gini_depts'] = 0
    else:
        metrics['dept_balance'] = 0
        metrics['gini_depts'] = 0
    
    # Calculate Gini for other metrics
    if 'preference_rank' in allocation_df.columns:
        ranks = allocation_df['preference_rank'].tolist()
        if len(ranks) > 1:
            ranks_sorted = sorted(ranks)
            n = len(ranks_sorted)
            cumsum = [ranks_sorted[0]]
            for i in range(1, n):
                cumsum.append(cumsum[-1] + ranks_sorted[i])
            total = cumsum[-1]
            if total > 0:
                metrics['gini_rank'] = (2 * sum((i + 1) * rank for i, rank in enumerate(ranks_sorted)) / (n * total)) - (n + 1) / n
            else:
                metrics['gini_rank'] = 0
    else:
        metrics['gini_rank'] = 0
    
    return metrics


def generate_all_charts_zip():
    """Generate all charts from Advanced Charts and Really Advanced Charts pages as a zip file."""
    try:
        # Check if we have the necessary data
        if st.session_state.last_allocation is None or st.session_state.last_allocation_rows is None:
            return None
        
        allocation_df = st.session_state.last_allocation
        rows = st.session_state.last_allocation_rows
        repo = getattr(st.session_state, 'last_repos', None)
        
        # Create a temporary directory to store all charts
        with tempfile.TemporaryDirectory() as tmpdir:
            chart_files = []
            
            # Get the charts from different modules
            sys.path.insert(0, str(Path(__file__).parent.parent))
            
            # ===== ADVANCED CHARTS GENERATION =====
            try:
                # 1. Sankey diagram (HTML)
                from streamlit_dashboard_pages.viz_sankey_enhanced import create_sankey_html
                
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
                
                sankey_path = f"{tmpdir}/advanced_charts_01_sankey_diagram.html"
                create_sankey_html(rows_dicts, sankey_path)
                chart_files.append(("advanced_charts_01_sankey_diagram.html", sankey_path))
                
                # 2. Student × Topic Heatmap
                if 'effective_cost' in allocation_df.columns and 'student' in allocation_df.columns and 'assigned_topic' in allocation_df.columns:
                    cost_pivot = allocation_df.pivot_table(
                        values='effective_cost',
                        index='student',
                        columns='assigned_topic',
                        fill_value=0,
                        aggfunc='first'
                    )
                    
                    # Handle forced topics (-10000 cost) specially for color scale
                    all_values = allocation_df['effective_cost'].values
                    non_forced_values = all_values[all_values > -9000]  # Filter out forced topics
                    if len(non_forced_values) > 0:
                        zmin = int(np.min(non_forced_values))
                        zmax = int(np.max(non_forced_values))
                    else:
                        zmin = 0
                        zmax = 200
                    
                    fig = go.Figure(data=go.Heatmap(
                        z=cost_pivot.values,
                        x=cost_pivot.columns,
                        y=cost_pivot.index,
                        colorscale='RdYlGn_r',
                        zmin=zmin,
                        zmax=zmax,
                        hovertemplate='Student: %{y}<br>Topic: %{x}<br>Cost: %{z}<extra></extra>'
                    ))
                    fig.update_layout(
                        title=f"Effective Cost Heatmap - All {len(allocation_df)} Students",
                        height=max(600, len(allocation_df) * 10),
                        xaxis_title="Topic",
                        yaxis_title="Student",
                        xaxis_tickangle=-45
                    )
                    fig.write_html(f"{tmpdir}/advanced_charts_02_student_topic_heatmap.html")
                    chart_files.append(("advanced_charts_02_student_topic_heatmap.html", f"{tmpdir}/advanced_charts_02_student_topic_heatmap.html"))
                
                # 3. Coach × Topic Heatmap
                if 'assigned_coach' in allocation_df.columns and 'assigned_topic' in allocation_df.columns and 'effective_cost' in allocation_df.columns:
                    coach_topic_cost = allocation_df.groupby(['assigned_coach', 'assigned_topic'])['effective_cost'].agg(['sum', 'count']).reset_index()
                    
                    cost_pivot_coach = coach_topic_cost.pivot_table(
                        values='sum',
                        index='assigned_coach',
                        columns='assigned_topic',
                        fill_value=0
                    )
                    
                    # Set reasonable bounds for color scale (filter out extreme values from forced assignments)
                    all_values = cost_pivot_coach.values.flatten()
                    reasonable_values = all_values[all_values > -9000]
                    if len(reasonable_values) > 0:
                        zmin = int(np.min(reasonable_values))
                        zmax = int(np.max(reasonable_values))
                    else:
                        zmin = 0
                        zmax = 1000
                    
                    fig = go.Figure(data=go.Heatmap(
                        z=cost_pivot_coach.values,
                        x=cost_pivot_coach.columns,
                        y=cost_pivot_coach.index,
                        colorscale='RdYlGn_r',
                        zmin=zmin,
                        zmax=zmax,
                        hovertemplate='Coach: %{y}<br>Topic: %{x}<br>Total Cost: %{z}<extra></extra>'
                    ))
                    fig.update_layout(
                        title="Coach × Topic Cost Distribution",
                        height=400,
                        xaxis_title="Topic",
                        yaxis_title="Coach",
                        xaxis_tickangle=-45
                    )
                    fig.write_html(f"{tmpdir}/advanced_charts_03_coach_topic_heatmap.html")
                    chart_files.append(("advanced_charts_03_coach_topic_heatmap.html", f"{tmpdir}/advanced_charts_03_coach_topic_heatmap.html"))
                
                # 4. Department × Topic Heatmap
                if 'department_id' in allocation_df.columns and 'assigned_topic' in allocation_df.columns and 'effective_cost' in allocation_df.columns:
                    dept_topic_cost = allocation_df.groupby(['department_id', 'assigned_topic'])['effective_cost'].agg(['sum', 'count']).reset_index()
                    
                    cost_pivot_dept = dept_topic_cost.pivot_table(
                        values='sum',
                        index='department_id',
                        columns='assigned_topic',
                        fill_value=0
                    )
                    
                    # Set reasonable bounds for color scale (filter out extreme values from forced assignments)
                    all_values = cost_pivot_dept.values.flatten()
                    reasonable_values = all_values[all_values > -9000]
                    if len(reasonable_values) > 0:
                        zmin = int(np.min(reasonable_values))
                        zmax = int(np.max(reasonable_values))
                    else:
                        zmin = 0
                        zmax = 1000
                    
                    fig = go.Figure(data=go.Heatmap(
                        z=cost_pivot_dept.values,
                        x=cost_pivot_dept.columns,
                        y=cost_pivot_dept.index,
                        colorscale='RdYlGn_r',
                        zmin=zmin,
                        zmax=zmax,
                        hovertemplate='Department: %{y}<br>Topic: %{x}<br>Total Cost: %{z}<extra></extra>'
                    ))
                    fig.update_layout(
                        title="Department × Topic Cost Distribution",
                        height=400,
                        xaxis_title="Topic",
                        yaxis_title="Department",
                        xaxis_tickangle=-45
                    )
                    fig.write_html(f"{tmpdir}/advanced_charts_04_department_topic_heatmap.html")
                    chart_files.append(("advanced_charts_04_department_topic_heatmap.html", f"{tmpdir}/advanced_charts_04_department_topic_heatmap.html"))
                
                # 5-7. Utilization charts (Topic, Coach, Department)
                if repo:
                    # Topic Utilization
                    if 'assigned_topic' in allocation_df.columns:
                        topic_counts = allocation_df['assigned_topic'].value_counts()
                        topic_names = []
                        used_counts = []
                        total_counts = []
                        for topic_id, count in sorted(topic_counts.items()):
                            if topic_id in repo.topics:
                                topic = repo.topics[topic_id]
                                topic_names.append(topic_id)
                                used_counts.append(count)
                                total_counts.append(topic.topic_cap)
                        
                        if topic_names:
                            fig = go.Figure(data=[
                                go.Bar(name='Used', x=topic_names, y=used_counts, marker_color='#27ae60', opacity=0.8),
                                go.Bar(name='Capacity', x=topic_names, y=total_counts, marker_color='#e74c3c', opacity=0.6)
                            ])
                            fig.update_layout(
                                title="Topic Utilization (Green=Used, Red=Capacity)",
                                xaxis_title="Topic",
                                yaxis_title="Students",
                                barmode='group',
                                height=500,
                                xaxis_tickangle=-45,
                                hovermode='x unified'
                            )
                            fig.write_html(f"{tmpdir}/advanced_charts_05_topic_utilization.html")
                            chart_files.append(("advanced_charts_05_topic_utilization.html", f"{tmpdir}/advanced_charts_05_topic_utilization.html"))
                    
                    # Coach Utilization
                    if 'assigned_coach' in allocation_df.columns:
                        coach_counts = allocation_df['assigned_coach'].value_counts()
                        coach_names = []
                        used_counts = []
                        total_counts = []
                        for coach_id, count in sorted(coach_counts.items()):
                            if coach_id in repo.coaches:
                                coach = repo.coaches[coach_id]
                                coach_names.append(coach_id)
                                used_counts.append(count)
                                total_counts.append(coach.coach_cap)
                        
                        if coach_names:
                            fig = go.Figure(data=[
                                go.Bar(name='Used', x=coach_names, y=used_counts, marker_color='#3498db', opacity=0.8),
                                go.Bar(name='Capacity', x=coach_names, y=total_counts, marker_color='#e67e22', opacity=0.6)
                            ])
                            fig.update_layout(
                                title="Coach Utilization (Blue=Used, Orange=Capacity)",
                                xaxis_title="Coach",
                                yaxis_title="Students",
                                barmode='group',
                                height=500,
                                xaxis_tickangle=-45,
                                hovermode='x unified'
                            )
                            fig.write_html(f"{tmpdir}/advanced_charts_06_coach_utilization.html")
                            chart_files.append(("advanced_charts_06_coach_utilization.html", f"{tmpdir}/advanced_charts_06_coach_utilization.html"))
                    
                    # Department Distribution (with real capacity)
                    if 'department_id' in allocation_df.columns:
                        dept_counts = allocation_df['department_id'].value_counts()
                        dept_names = []
                        counts = []
                        capacities = []
                        for dept_id, count in sorted(dept_counts.items()):
                            if dept_id in repo.departments:
                                dept = repo.departments[dept_id]
                                dept_names.append(dept_id)
                                counts.append(count)
                                # Use desired_max as capacity, fallback to desired_min, or calculate from count if not set
                                if dept.desired_max > 0:
                                    capacity = dept.desired_max
                                elif dept.desired_min > 0:
                                    capacity = dept.desired_min
                                else:
                                    capacity = (count + count // 2 if count > 0 else 5)
                                capacities.append(capacity)
                        
                        if dept_names:
                            # Bar chart with used vs capacity
                            fig = go.Figure(data=[
                                go.Bar(name='Used', x=dept_names, y=counts, marker_color='#9b59b6', opacity=0.8),
                                go.Bar(name='Capacity', x=dept_names, y=capacities, marker_color='#e91e63', opacity=0.6)
                            ])
                            fig.update_layout(
                                title="Department Utilization (Purple=Used, Pink=Capacity)",
                                xaxis_title="Department",
                                yaxis_title="Students",
                                barmode='group',
                                height=500,
                                xaxis_tickangle=-45
                            )
                            fig.write_html(f"{tmpdir}/advanced_charts_07_department_distribution.html")
                            chart_files.append(("advanced_charts_07_department_distribution.html", f"{tmpdir}/advanced_charts_07_department_distribution.html"))
                            
                            # Also generate a pie chart for department distribution
                            fig_pie = px.pie(
                                values=counts,
                                names=dept_names,
                                title="Student Distribution by Department"
                            )
                            fig_pie.write_html(f"{tmpdir}/advanced_charts_08_department_pie.html")
                            chart_files.append(("advanced_charts_08_department_pie.html", f"{tmpdir}/advanced_charts_08_department_pie.html"))
                            
                            # Load balance charts
                            metrics = calculate_fairness_score(allocation_df)
                            
                            # Topic Load Balance
                            topic_counts = allocation_df['assigned_topic'].value_counts()
                            all_topics = sorted(repo.topics.keys())
                            topic_counts_full = {t: topic_counts.get(t, 0) for t in all_topics}
                            avg_val = sum(topic_counts_full.values()) / len(topic_counts_full) if topic_counts_full else 0
                            
                            fig_topic_balance = go.Figure()
                            fig_topic_balance.add_trace(go.Bar(
                                x=list(topic_counts_full.keys()),
                                y=list(topic_counts_full.values()),
                                marker_color='#3498db',
                                name='Students'
                            ))
                            fig_topic_balance.add_hline(y=avg_val, line_dash="dash", line_color="red", annotation_text=f"Avg: {avg_val:.1f}")
                            fig_topic_balance.update_layout(
                                title=f"Topic Load Balance (Gini: {metrics.get('gini_topics', 0):.3f})",
                                xaxis_title="Topic",
                                yaxis_title="Students Assigned",
                                height=500,
                                xaxis_tickangle=-45
                            )
                            fig_topic_balance.write_html(f"{tmpdir}/advanced_charts_09_topic_load_balance.html")
                            chart_files.append(("advanced_charts_09_topic_load_balance.html", f"{tmpdir}/advanced_charts_09_topic_load_balance.html"))
                            
                            # Coach Load Balance
                            coach_counts = allocation_df['assigned_coach'].value_counts()
                            all_coaches = sorted(repo.coaches.keys())
                            coach_counts_full = {c: coach_counts.get(c, 0) for c in all_coaches}
                            avg_val = sum(coach_counts_full.values()) / len(coach_counts_full) if coach_counts_full else 0
                            
                            fig_coach_balance = go.Figure()
                            fig_coach_balance.add_trace(go.Bar(
                                x=list(coach_counts_full.keys()),
                                y=list(coach_counts_full.values()),
                                marker_color='#e67e22',
                                name='Students'
                            ))
                            fig_coach_balance.add_hline(y=avg_val, line_dash="dash", line_color="red", annotation_text=f"Avg: {avg_val:.1f}")
                            fig_coach_balance.update_layout(
                                title=f"Coach Load Balance (Gini: {metrics.get('gini_coaches', 0):.3f})",
                                xaxis_title="Coach",
                                yaxis_title="Students Assigned",
                                height=500,
                                xaxis_tickangle=-45
                            )
                            fig_coach_balance.write_html(f"{tmpdir}/advanced_charts_10_coach_load_balance.html")
                            chart_files.append(("advanced_charts_10_coach_load_balance.html", f"{tmpdir}/advanced_charts_10_coach_load_balance.html"))
                            
                            # Department Load Balance
                            all_depts = sorted(repo.departments.keys())
                            dept_counts_full = {d: dept_counts.get(d, 0) for d in all_depts}
                            avg_val = sum(dept_counts_full.values()) / len(dept_counts_full) if dept_counts_full else 0
                            
                            fig_dept_balance = go.Figure()
                            fig_dept_balance.add_trace(go.Bar(
                                x=list(dept_counts_full.keys()),
                                y=list(dept_counts_full.values()),
                                marker_color='#9b59b6',
                                name='Students'
                            ))
                            fig_dept_balance.add_hline(y=avg_val, line_dash="dash", line_color="red", annotation_text=f"Avg: {avg_val:.1f}")
                            fig_dept_balance.update_layout(
                                title=f"Department Load Balance (Gini: {metrics.get('gini_depts', 0):.3f})",
                                xaxis_title="Department",
                                yaxis_title="Students Assigned",
                                height=500,
                                xaxis_tickangle=-45
                            )
                            fig_dept_balance.write_html(f"{tmpdir}/advanced_charts_11_department_load_balance.html")
                            chart_files.append(("advanced_charts_11_department_load_balance.html", f"{tmpdir}/advanced_charts_11_department_load_balance.html"))
                            
                            # Topic Normalized Load Balance (% of capacity)
                            topic_data = []
                            for topic_id in sorted(repo.topics.keys()):
                                topic = repo.topics[topic_id]
                                assigned = topic_counts.get(topic_id, 0)
                                capacity = topic.topic_cap
                                normalized = (assigned / capacity * 100) if (capacity is not None and capacity > 0) else 0
                                topic_data.append({'Topic': topic_id, 'Assigned': assigned, 'Capacity': capacity, 'Usage %': normalized})
                            
                            topic_norm_df = pd.DataFrame(topic_data)
                            fig_topic_norm = px.bar(topic_norm_df, x='Topic', y='Usage %', title='Topic Utilization (% of Capacity)', 
                                                    color='Usage %', color_continuous_scale='RdYlGn_r', labels={'Usage %': 'Capacity Used (%)'},
                                                    hover_data={'Assigned': True, 'Capacity': True})
                            fig_topic_norm.add_hline(y=100, line_dash="dash", line_color="red", annotation_text="100% Full")
                            fig_topic_norm.add_hline(y=80, line_dash="dash", line_color="orange", annotation_text="80% Utilization")
                            fig_topic_norm.update_layout(height=500, xaxis_tickangle=-45)
                            fig_topic_norm.write_html(f"{tmpdir}/advanced_charts_12_topic_normalized_load_balance.html")
                            chart_files.append(("advanced_charts_12_topic_normalized_load_balance.html", f"{tmpdir}/advanced_charts_12_topic_normalized_load_balance.html"))
                            
                            # Coach Normalized Load Balance (% of capacity)
                            coach_data = []
                            for coach_id in sorted(repo.coaches.keys()):
                                coach = repo.coaches[coach_id]
                                assigned = coach_counts.get(coach_id, 0)
                                capacity = coach.coach_cap
                                normalized = (assigned / capacity * 100) if (capacity is not None and capacity > 0) else 0
                                coach_data.append({'Coach': coach_id, 'Assigned': assigned, 'Capacity': capacity, 'Usage %': normalized})
                            
                            coach_norm_df = pd.DataFrame(coach_data)
                            fig_coach_norm = px.bar(coach_norm_df, x='Coach', y='Usage %', title='Coach Utilization (% of Capacity)', 
                                                    color='Usage %', color_continuous_scale='RdYlGn_r', labels={'Usage %': 'Capacity Used (%)'},
                                                    hover_data={'Assigned': True, 'Capacity': True})
                            fig_coach_norm.add_hline(y=100, line_dash="dash", line_color="red", annotation_text="100% Full")
                            fig_coach_norm.add_hline(y=80, line_dash="dash", line_color="orange", annotation_text="80% Utilization")
                            fig_coach_norm.update_layout(height=500, xaxis_tickangle=-45)
                            fig_coach_norm.write_html(f"{tmpdir}/advanced_charts_13_coach_normalized_load_balance.html")
                            chart_files.append(("advanced_charts_13_coach_normalized_load_balance.html", f"{tmpdir}/advanced_charts_13_coach_normalized_load_balance.html"))
                            
                            # Department Normalized Load Balance (% of capacity) - FIXED
                            dept_data = []
                            for dept_id in sorted(repo.departments.keys()):
                                dept = repo.departments[dept_id]
                                assigned = dept_counts.get(dept_id, 0)
                                # Use desired_max as capacity, fallback to desired_min, or calculate from average
                                if dept.desired_max > 0:
                                    capacity = dept.desired_max
                                elif dept.desired_min > 0:
                                    capacity = dept.desired_min
                                else:
                                    # Calculate from other departments
                                    avg_assigned = sum(dept_counts.values()) / len(dept_counts) if dept_counts else 0
                                    capacity = max(avg_assigned * 1.2, assigned + 1) if avg_assigned > 0 else assigned + 1
                                normalized = (assigned / capacity * 100) if (capacity is not None and capacity > 0) else 0
                                dept_data.append({'Department': dept_id, 'Assigned': assigned, 'Capacity': capacity, 'Usage %': normalized})
                            
                            dept_norm_df = pd.DataFrame(dept_data)
                            fig_dept_norm = px.bar(dept_norm_df, x='Department', y='Usage %', title='Department Utilization (% of Capacity)', 
                                                   color='Usage %', color_continuous_scale='RdYlGn_r', labels={'Usage %': 'Capacity Used (%)'},
                                                   hover_data={'Assigned': True, 'Capacity': True})
                            fig_dept_norm.add_hline(y=100, line_dash="dash", line_color="red", annotation_text="100% Full")
                            fig_dept_norm.add_hline(y=80, line_dash="dash", line_color="orange", annotation_text="80% Utilization")
                            fig_dept_norm.update_layout(height=500, xaxis_tickangle=-45)
                            fig_dept_norm.write_html(f"{tmpdir}/advanced_charts_14_department_normalized_load_balance.html")
                            chart_files.append(("advanced_charts_14_department_normalized_load_balance.html", f"{tmpdir}/advanced_charts_14_department_normalized_load_balance.html"))
                            
                            # Overall Fairness Gauge Chart
                            cost_fairness = (1 - metrics.get('gini_cost', 0)) * 100
                            pref_fairness = metrics.get('ranked_satisfaction', 0) * 100
                            topic_balance_score = metrics.get('topic_balance', 0) * 100
                            coach_balance_score = metrics.get('coach_balance', 0) * 100
                            dept_balance_score = metrics.get('dept_balance', 0) * 100
                            
                            overall_score = (cost_fairness * 0.15 + pref_fairness * 0.3 + topic_balance_score * 0.15 + 
                                            coach_balance_score * 0.15 + dept_balance_score * 0.15 + 100 * 0.1)
                            
                            fig_fairness_gauge = go.Figure(go.Indicator(
                                mode="gauge+number+delta",
                                value=overall_score,
                                domain={'x': [0, 1], 'y': [0, 1]},
                                title={'text': "Overall Fairness Score"},
                                delta={'reference': 75},
                                gauge={'axis': {'range': [None, 100]},
                                       'bar': {'color': "darkblue"},
                                       'steps': [{'range': [0, 60], 'color': "lightgray"},
                                                 {'range': [60, 80], 'color': "gray"}],
                                       'threshold': {'line': {'color': "red", 'width': 4},
                                                     'thickness': 0.75, 'value': 90}}))
                            fig_fairness_gauge.update_layout(height=400)
                            fig_fairness_gauge.write_html(f"{tmpdir}/advanced_charts_15_overall_fairness_gauge.html")
                            chart_files.append(("advanced_charts_15_overall_fairness_gauge.html", f"{tmpdir}/advanced_charts_15_overall_fairness_gauge.html"))
                            
                            # Student Satisfaction Distribution
                            if 'preference_rank' in allocation_df.columns:
                                satisfaction_counts = allocation_df['preference_rank'].value_counts()
                                labels_map = {0: 'Tier 1', 1: 'Tier 2', 2: 'Tier 3', 10: '1st Choice', 11: '2nd Choice', 
                                             12: '3rd Choice', 13: '4th Choice', 14: '5th Choice', 999: 'Unranked', -1: 'Forced'}
                                labels = [labels_map.get(rank, f'Rank {rank}') for rank in satisfaction_counts.index]
                                values = satisfaction_counts.values
                                
                                fig_satisfaction = px.bar(x=labels, y=values, title='Student Satisfaction Distribution',
                                                          labels={'x': 'Satisfaction Level', 'y': 'Number of Students'},
                                                          color=values, color_continuous_scale='RdYlGn')
                                fig_satisfaction.update_layout(height=500, xaxis_tickangle=-45)
                                fig_satisfaction.write_html(f"{tmpdir}/advanced_charts_16_student_satisfaction_distribution.html")
                                chart_files.append(("advanced_charts_16_student_satisfaction_distribution.html", f"{tmpdir}/advanced_charts_16_student_satisfaction_distribution.html"))
            
            except Exception as e:
                st.warning(f"Could not generate some Advanced Charts: {e}")
            
            # ===== REALLY ADVANCED CHARTS GENERATION =====
            try:
                from streamlit_dashboard_pages.viz_really_advanced_charts import (
                    create_preference_funnel, create_cost_breakdown_pie, create_cost_violin_plot,
                    create_fairness_radar, create_topic_demand_vs_capacity,
                    create_student_satisfaction_scatter, create_coach_specialization_heatmap,
                    create_department_diversity_analysis
                )
                
                # Preference & Cost Analysis
                fig_funnel = create_preference_funnel(allocation_df)
                if fig_funnel:
                    fig_funnel.write_html(f"{tmpdir}/really_advanced_01_preference_funnel.html")
                    chart_files.append(("really_advanced_01_preference_funnel.html", f"{tmpdir}/really_advanced_01_preference_funnel.html"))
                
                fig_pie = create_cost_breakdown_pie(allocation_df)
                if fig_pie:
                    fig_pie.write_html(f"{tmpdir}/really_advanced_02_cost_breakdown_pie.html")
                    chart_files.append(("really_advanced_02_cost_breakdown_pie.html", f"{tmpdir}/really_advanced_02_cost_breakdown_pie.html"))
                
                fig_violin = create_cost_violin_plot(allocation_df)
                if fig_violin:
                    fig_violin.write_html(f"{tmpdir}/really_advanced_03_cost_violin_plot.html")
                    chart_files.append(("really_advanced_03_cost_violin_plot.html", f"{tmpdir}/really_advanced_03_cost_violin_plot.html"))
                
                # Fairness & Capacity - Fairness Radar needs metrics
                if hasattr(st.session_state, 'last_repos') and st.session_state.last_repos:
                    metrics = calculate_fairness_score(allocation_df)
                    fig_radar = create_fairness_radar(metrics)
                    if fig_radar:
                        fig_radar.write_html(f"{tmpdir}/really_advanced_04_fairness_radar.html")
                        chart_files.append(("really_advanced_04_fairness_radar.html", f"{tmpdir}/really_advanced_04_fairness_radar.html"))
                
                if repo:
                    fig_demand = create_topic_demand_vs_capacity(allocation_df, repo)
                    if fig_demand:
                        fig_demand.write_html(f"{tmpdir}/really_advanced_05_topic_demand_vs_capacity.html")
                        chart_files.append(("really_advanced_05_topic_demand_vs_capacity.html", f"{tmpdir}/really_advanced_05_topic_demand_vs_capacity.html"))
                    
                    # Deep Dive
                    fig_scatter = create_student_satisfaction_scatter(allocation_df, repo)
                    if fig_scatter:
                        fig_scatter.write_html(f"{tmpdir}/really_advanced_06_student_satisfaction_scatter.html")
                        chart_files.append(("really_advanced_06_student_satisfaction_scatter.html", f"{tmpdir}/really_advanced_06_student_satisfaction_scatter.html"))
                    
                    fig_coach = create_coach_specialization_heatmap(allocation_df, repo)
                    if fig_coach:
                        fig_coach.write_html(f"{tmpdir}/really_advanced_07_coach_specialization_heatmap.html")
                        chart_files.append(("really_advanced_07_coach_specialization_heatmap.html", f"{tmpdir}/really_advanced_07_coach_specialization_heatmap.html"))
                    
                    fig_dept = create_department_diversity_analysis(allocation_df)
                    if fig_dept:
                        fig_dept.write_html(f"{tmpdir}/really_advanced_08_department_diversity.html")
                        chart_files.append(("really_advanced_08_department_diversity.html", f"{tmpdir}/really_advanced_08_department_diversity.html"))
            
            except Exception as e:
                st.warning(f"Could not generate Really Advanced Charts: {e}")
            
            # ===== RESULTS ANALYSIS CHARTS GENERATION =====
            try:
                # Generate charts from Results Analysis page using summary text
                summary_text = getattr(st.session_state, 'last_summary', '')
                
                # Hot Topics Chart (needs repo data)
                if repo and hasattr(repo, 'students') and hasattr(repo, 'topics'):
                    from collections import defaultdict
                    
                    # Build preference counts per topic (pref1, pref2, and pref3)
                    topic_pref_counts = defaultdict(lambda: {'pref1': 0, 'pref2': 0, 'pref3': 0})
                    
                    for student_id, student in repo.students.items():
                        if not student.plan:
                            continue
                        
                        for idx, topic_id in enumerate(student.ranks[:3], start=1):
                            topic_pref_counts[topic_id][f'pref{idx}'] += 1
                    
                    # Prepare data for ALL topics (even those with no preferences)
                    topic_data = []
                    for topic_id in sorted(repo.topics.keys()):  # Get all topics from repo
                        if topic_id in topic_pref_counts:
                            counts = topic_pref_counts[topic_id]
                        else:
                            counts = {'pref1': 0, 'pref2': 0, 'pref3': 0}  # No preferences for this topic
                        
                        topic_data.append({
                            'Topic': topic_id,
                            'Pref 1': counts['pref1'],
                            'Pref 2': counts['pref2'],
                            'Pref 3': counts['pref3'],
                            'Total': counts['pref1'] + counts['pref2'] + counts['pref3']
                        })
                    
                    # Already sorted alphabetically
                    topic_names = [d['Topic'] for d in topic_data]
                    
                    fig_hot = go.Figure()
                    fig_hot.add_trace(go.Bar(
                        name='1st Choice',
                        x=topic_names,
                        y=[d['Pref 1'] for d in topic_data],
                        marker_color='#2ecc71',
                        hovertemplate='<b>%{x}</b><br>1st Choice: %{y}<extra></extra>'
                    ))
                    fig_hot.add_trace(go.Bar(
                        name='2nd Choice',
                        x=topic_names,
                        y=[d['Pref 2'] for d in topic_data],
                        marker_color='#3498db',
                        hovertemplate='<b>%{x}</b><br>2nd Choice: %{y}<extra></extra>'
                    ))
                    fig_hot.add_trace(go.Bar(
                        name='3rd Choice',
                        x=topic_names,
                        y=[d['Pref 3'] for d in topic_data],
                        marker_color='#f39c12',
                        hovertemplate='<b>%{x}</b><br>3rd Choice: %{y}<extra></extra>'
                    ))
                    
                    fig_hot.update_layout(
                        title='Hot Topics - Student Preferences (All Topics)',
                        xaxis_title='Topic',
                        yaxis_title='Number of Students',
                        barmode='stack',
                        height=600,
                        xaxis_tickangle=-45,
                        legend=dict(orientation="v", yanchor="top", y=1, xanchor="right", x=1.02)
                    )
                    fig_hot.write_html(f"{tmpdir}/results_analysis_04_hot_topics.html")
                    chart_files.append(("results_analysis_04_hot_topics.html", f"{tmpdir}/results_analysis_04_hot_topics.html"))
                
                if summary_text:
                    # 1. Preference Satisfaction Chart
                    if "Ranked choice satisfaction:" in summary_text:
                        lines = summary_text.split('\n')
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
                            fig_pref = px.bar(
                                x=list(satisfaction.keys()),
                                y=list(satisfaction.values()),
                                title="Preference Satisfaction (Ranked Choices)",
                                labels={"x": "Choice Rank", "y": "Number of Students"},
                                color=list(satisfaction.values()),
                                color_continuous_scale="Viridis"
                            )
                            fig_pref.update_layout(height=400, showlegend=False)
                            fig_pref.write_html(f"{tmpdir}/results_analysis_01_preference_satisfaction.html")
                            chart_files.append(("results_analysis_01_preference_satisfaction.html", f"{tmpdir}/results_analysis_01_preference_satisfaction.html"))
                    
                    # 2. Department Distribution Pie Chart
                    if "Department totals:" in summary_text:
                        lines = summary_text.split('\n')
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
                            fig_dept_pie = px.pie(
                                values=list(departments.values()),
                                names=list(departments.keys()),
                                title="Student Distribution by Department",
                                color_discrete_sequence=px.colors.qualitative.Set3
                            )
                            fig_dept_pie.update_layout(height=500)
                            fig_dept_pie.write_html(f"{tmpdir}/results_analysis_02_department_distribution.html")
                            chart_files.append(("results_analysis_02_department_distribution.html", f"{tmpdir}/results_analysis_02_department_distribution.html"))
                    
                    # 3. Topic Capacity Utilization Chart
                    if "Topic utilization:" in summary_text and repo:
                        lines = summary_text.split('\n')
                        topics = {}
                        in_util = False
                        for line in lines:
                            if "Topic utilization:" in line:
                                in_util = True
                            elif in_util and ':' in line and line.startswith('  topic'):
                                parts = line.strip().split(':')
                                if len(parts) == 2:
                                    topic_name = parts[0].strip()
                                    util_parts = parts[1].split('/')
                                    if len(util_parts) == 2:
                                        used = int(util_parts[0].strip())
                                        total = int(util_parts[1].strip())
                                        topics[topic_name] = {'used': used, 'total': total}
                            elif in_util and line.strip() == "" and topics:
                                break
                        
                        # Fallback: generate from allocation data
                        if not topics and 'assigned_topic' in allocation_df.columns:
                            topic_counts = allocation_df['assigned_topic'].value_counts()
                            for topic_id, count in topic_counts.items():
                                if topic_id in repo.topics:
                                    topic = repo.topics[topic_id]
                                    topics[topic_id] = {'used': count, 'total': topic.topic_cap}
                        
                        if topics:
                            topic_names = []
                            used_counts = []
                            total_counts = []
                            for topic_id, data in sorted(topics.items()):
                                topic_names.append(topic_id)
                                used_counts.append(data['used'])
                                total_counts.append(data['total'])
                            
                            df_util = pd.DataFrame({
                                'Topic': topic_names,
                                'Used': used_counts,
                                'Total': total_counts
                            })
                            
                            fig_util = px.bar(
                                df_util,
                                x='Topic',
                                y=['Used', 'Total'],
                                title="Topic Capacity Utilization",
                                labels={"value": "Number of Students", "Topic": "Topic"},
                                barmode="overlay",
                                color_discrete_map={"Used": "#0066cc", "Total": "#cccccc"}
                            )
                            fig_util.update_layout(height=500, xaxis_tickangle=-45)
                            fig_util.write_html(f"{tmpdir}/results_analysis_03_topic_capacity_utilization.html")
                            chart_files.append(("results_analysis_03_topic_capacity_utilization.html", f"{tmpdir}/results_analysis_03_topic_capacity_utilization.html"))
            
            except Exception as e:
                st.warning(f"Could not generate some Results Analysis charts: {e}")
            
            # ===== NETWORK GRAPHS GENERATION =====
            try:
                # Import from project root (these files are not in streamlit_dashboard_pages)
                viz_network_flow = __import__('viz_network_flow', fromlist=['create_network_visualization', 'create_multipartite_visualization', 'create_edge_colormap_visualization'])
                viz_atlas = __import__('viz_atlas', fromlist=['create_atlas_visualization', 'create_analytical_atlas', 'create_pattern_analysis'])
                
                # Convert rows to dict format for network visualizations
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
                
                # 1. Network Flow Main
                try:
                    network_main_path = f"{tmpdir}/network_flow_main.html"
                    viz_network_flow.create_network_visualization(rows_dicts, network_main_path)
                    chart_files.append(("network_graphs_01_main_network.html", network_main_path))
                except Exception as e:
                    st.warning(f"Could not generate main network graph: {e}")
                
                # 2. Network Flow Multipartite
                try:
                    network_multipartite_path = f"{tmpdir}/network_flow_multipartite.html"
                    viz_network_flow.create_multipartite_visualization(rows_dicts, network_multipartite_path)
                    chart_files.append(("network_graphs_02_multipartite.html", network_multipartite_path))
                except Exception as e:
                    st.warning(f"Could not generate multipartite network: {e}")
                
                # 3. Network Flow Colormap
                try:
                    network_colormap_path = f"{tmpdir}/network_flow_colormap.html"
                    viz_network_flow.create_edge_colormap_visualization(rows_dicts, network_colormap_path)
                    chart_files.append(("network_graphs_03_colormap.html", network_colormap_path))
                except Exception as e:
                    st.warning(f"Could not generate colormap network: {e}")
                
                # 4. Atlas (Theoretical Patterns)
                try:
                    atlas_path = f"{tmpdir}/atlas_theoretical.html"
                    viz_atlas.create_atlas_visualization(rows_dicts, atlas_path)
                    chart_files.append(("network_graphs_04_atlas_theoretical.html", atlas_path))
                except Exception as e:
                    st.warning(f"Could not generate atlas: {e}")
                
                # 5. Analytical Atlas
                try:
                    analytical_atlas_path = f"{tmpdir}/atlas_analytical.html"
                    viz_atlas.create_analytical_atlas(rows_dicts, analytical_atlas_path)
                    chart_files.append(("network_graphs_05_atlas_analytical.html", analytical_atlas_path))
                except Exception as e:
                    st.warning(f"Could not generate analytical atlas: {e}")
                
                # 6. Pattern Analysis
                try:
                    patterns_path = f"{tmpdir}/patterns_analysis.html"
                    viz_atlas.create_pattern_analysis(rows_dicts, patterns_path)
                    chart_files.append(("network_graphs_06_patterns.html", patterns_path))
                except Exception as e:
                    st.warning(f"Could not generate pattern analysis: {e}")
            
            except Exception as e:
                st.warning(f"Could not generate some Network Graphs: {e}")
            
            # Save allocation data as CSV for reference
            allocation_df.to_csv(f"{tmpdir}/allocation_data.csv", index=False)
            chart_files.append(("allocation_data.csv", f"{tmpdir}/allocation_data.csv"))
            
            # Create zip file
            zip_buffer = BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                for filename, filepath in chart_files:
                    zip_file.write(filepath, filename)
            
            zip_buffer.seek(0)
            return zip_buffer.read()
    
    except Exception as e:
        st.error(f"Error generating charts zip: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None


def safe_set_page_config(*args, **kwargs):
    """Safely call st.set_page_config once; ignore if already set or called late."""
    try:
        import streamlit as st
        st.set_page_config(*args, **kwargs)
    except Exception:
        pass
