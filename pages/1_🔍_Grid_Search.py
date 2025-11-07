"""
Grid Search page for Thesis Allocation System
"""
import streamlit as st
import pandas as pd
from pathlib import Path
import sys
import subprocess
import tempfile
import json
from datetime import datetime
from collections import Counter
import io
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

# Add project root to path for imports
try:
    current_dir = Path(__file__).parent
    project_root = current_dir.parent  # project root
    sys.path.insert(0, str(project_root))
except NameError:
    # Fallback for testing
    project_root = Path.cwd()
    sys.path.insert(0, str(project_root))

from streamlit_dashboard_pages.shared import initialize_session_state
from allocator.data_repository import DataRepository
from allocator.preference_model import PreferenceModel
from allocator.allocation_model_ilp import AllocationModelILP

# Initialize session state
initialize_session_state()


st.header("🔍 Grid Search for Optimal Costs")
st.write("Explore different cost combinations to find Pareto-optimal solutions balancing satisfaction and fairness.")

# Check if configuration is being used
config_file_path = project_root / "config_streamlit.json"
if config_file_path.exists() and hasattr(st.session_state, 'config_algorithm'):
    st.success("⚙️ Using saved configuration from ⚙️ Configuration page!")

# Explanatory section
with st.expander("📖 What is Grid Search?", expanded=True):
    st.markdown("""
    **Grid Search** systematically tests different combinations of preference costs to find optimal solutions.
    
    **How it works:**
    1. 🎯 **Generate combinations**: Creates thousands of different cost configurations
    2. 🚀 **Run allocations**: Tests each configuration
    3. 📊 **Calculate metrics**: Measures satisfaction and fairness for each
    4. ⭐ **Find Pareto frontier**: Identifies solutions where neither satisfaction nor fairness can be improved without degrading the other
    5. 🎉 **Recommend solutions**: Suggests the best options based on different priorities
    
    **Outputs:**
    - Pareto frontier plot showing the optimal trade-offs
    - CSV with all results
    - Summary with recommended solutions
    - 4-panel visualization of the search space
    """)

# File uploads and grid search type
col1, col2 = st.columns(2)

with col1:
    st.subheader("📥 Input Files")
    
    st.info("📄 **Default files preloaded:**")
    st.write(f"• Students: `{st.session_state.students_file}`")
    st.write(f"• Capacities: `{st.session_state.capacities_file}`")
    st.write("💡 Upload custom files below to override defaults")
    
    st.divider()
    
    students_file = st.file_uploader(
        "Students CSV (Override Default)",
        type=['csv'],
        key="grid_students",
        help="CSV with student preferences. Leave empty to use default file."
    )
    capacities_file = st.file_uploader(
        "Capacities CSV (Override Default)",
        type=['csv'],
        key="grid_capacities",
        help="CSV with topic/coach capacities. Leave empty to use default file."
    )

with col2:
    st.subheader("🎯 Grid Search Options")
    
    grid_type = st.radio(
        "Select grid search type:",
        ["Ranks only (rank1-5)", "With tiers (rank1-5 + tier2-3)", "Both (all combinations)"],
        help="Choose which cost parameters to explore"
    )
    
    st.divider()
    
    # Granularity settings
    st.subheader("⚙️ Search Parameters")
    
    granularity = st.selectbox(
        "Search Granularity:",
        options=[1, 2],
        index=0,
        help="Higher = more combinations (slower but more thorough)"
    )
    
    granularity_info = {
        1: "Fast: ~12K combinations (5-6 min)",
        2: "Medium: ~30K combinations (12-15 min)",
        3: "Detailed: ~50K combinations (20-25 min)",
        4: "More Detailed: ~100K combinations (40-50 min)",
        5: "Very Detailed: ~150K combinations (60-75 min)",
        6: "Very Detailed: ~200K combinations (80-100 min)",
        7: "Extremely Detailed: ~250K combinations (100-125 min)",
        8: "Exhaustive: ~300K combinations (120-150 min)",
        9: "Most Exhaustive: ~350K combinations (140-175 min)",
        10: "Ultra Exhaustive: ~400K combinations (160-200 min)"
    }
    
    st.info(f"ℹ️ {granularity_info[granularity]}")
    
    cores = st.number_input(
        "CPU Cores:",
        min_value=1,
        max_value=32,
        value=14,
        help="Number of parallel workers"
    )

# Check for forced topics
st.subheader("🔍 Data Analysis")

try:
    # Load the actual data
    students_path = students_file if students_file else Path(st.session_state.students_file)
    capacities_path = capacities_file if capacities_file else Path(st.session_state.capacities_file)
    
    if students_file:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
            tmp.write(students_file.getvalue().decode('utf-8'))
            students_path = tmp.name
    
    if capacities_file:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
            tmp.write(capacities_file.getvalue().decode('utf-8'))
            capacities_path = tmp.name
    
    # Load data repository
    repo = DataRepository(str(students_path), str(capacities_path))
    repo.load()
    
    # Check for forced topics
    students_with_forced = [s for s in repo.students.values() if s.forced_topic]
    
    col3, col4, col5 = st.columns(3)
    
    with col3:
        st.metric("Total Students", len([s for s in repo.students.values() if s.plan]))
    
    with col4:
        st.metric("Total Topics", len(repo.topics))
    
    with col5:
        if students_with_forced:
            st.metric("🔒 Students with Forced Topics", len(students_with_forced), delta="Forced assignments will be used")
        else:
            st.metric("Students with Forced Topics", 0)
    
    if students_with_forced:
        st.warning(f"⚠️ Found {len(students_with_forced)} students with forced topic assignments. These will be respected in the grid search.")
        
        with st.expander(f"View students with forced topics ({len(students_with_forced)})"):
            forced_data = []
            for student in students_with_forced:
                forced_data.append({
                    'Student ID': student.student,
                    'Forced Topic': student.forced_topic
                })
            st.dataframe(pd.DataFrame(forced_data), use_container_width=True)
    
    # Check preference format
    students_with_tiers = [s for s in repo.students.values() if s.tiers and s.plan]
    students_with_ranks = [s for s in repo.students.values() if s.ranks and s.plan]
    
    st.info(f"📊 **Preference formats detected:** Tiers={len(students_with_tiers)} students, Ranks={len(students_with_ranks)} students")

except Exception as e:
    st.error(f"❌ Error loading data: {e}")
    st.stop()

# Run grid search button
st.divider()
run_button = st.button("🚀 Run Grid Search", type="primary", use_container_width=True)

if run_button:
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(project_root) / "data" / "output" / "simulations" / "grid_search" / f"{timestamp}_{grid_type.replace(' ', '_')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare command
    if grid_type == "Ranks only (rank1-5)":
        script = "comprehensive_grid_search_ranks_only.py"
        cmd = [
            sys.executable, script,
            "--students", str(students_path),
            "--capacities", str(capacities_path),
            "--output", str(output_dir),
            "--cores", str(cores),
            "--granularity", str(granularity)
        ]
    elif grid_type == "With tiers (rank1-5 + tier2-3)":
        script = "comprehensive_grid_search_with_tiers.py"
        cmd = [
            sys.executable, script,
            "--students", str(students_path),
            "--capacities", str(capacities_path),
            "--output", str(output_dir)
        ]
    else:  # Both
        st.warning("⚠️ 'Both' option runs the tier search (most comprehensive). For exhaustive search, run ranks-only separately.")
        script = "comprehensive_grid_search_with_tiers.py"
        cmd = [
            sys.executable, script,
            "--students", str(students_path),
            "--capacities", str(capacities_path),
            "--output", str(output_dir)
        ]
    
    # Run with progress
    with st.spinner(f"🔍 Running grid search... This may take {granularity_info[granularity].split('(')[1] if 'Ranks only' in grid_type else '5-10 minutes'}"):
        try:
            # Run the script
            result = subprocess.run(
                cmd,
                cwd=project_root,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            if result.returncode == 0:
                st.success("✅ Grid search completed successfully!")
                
                # Display results
                st.subheader("📊 Results")
                
                # Check for output files
                csv_files = list(output_dir.glob("*results*.csv"))
                txt_files = list(output_dir.glob("*summary*.txt"))
                png_files = list(output_dir.glob("*.png"))
                
                if csv_files:
                    st.success(f"📄 Found {len(csv_files)} CSV results file(s)")
                    
                    for csv_file in csv_files:
                        with st.expander(f"📊 View {csv_file.name}"):
                            df = pd.read_csv(csv_file)
                            st.dataframe(df, use_container_width=True)
                            
                            # Download button
                            st.download_button(
                                "⬇️ Download CSV",
                                df.to_csv(index=False),
                                file_name=csv_file.name,
                                mime="text/csv"
                            )
                
                if txt_files:
                    st.success(f"📝 Found {len(txt_files)} summary file(s)")
                    
                    for txt_file in txt_files:
                        with st.expander(f"📄 View {txt_file.name}"):
                            with open(txt_file, 'r') as f:
                                content = f.read()
                            st.text(content)
                            
                            st.download_button(
                                "⬇️ Download Summary",
                                content,
                                file_name=txt_file.name,
                                mime="text/plain"
                            )
                
                if png_files:
                    st.success(f"🖼️ Found {len(png_files)} visualization file(s)")
                    
                    for png_file in png_files:
                        st.image(str(png_file))
                        
                        # Download button
                        with open(png_file, 'rb') as f:
                            img_data = f.read()
                        st.download_button(
                            "⬇️ Download Image",
                            img_data,
                            file_name=png_file.name,
                            mime="image/png"
                        )
                
                # Interactive Pareto frontier plot if CSV available
                if csv_files:
                    try:
                        st.divider()
                        st.subheader("📈 Interactive Pareto Frontier")
                        
                        for csv_file in csv_files:
                            df = pd.read_csv(csv_file)
                            
                            if 'satisfaction_score' in df.columns and 'fairness_score' in df.columns and 'is_pareto' in df.columns:
                                # Create interactive plot
                                fig = go.Figure()
                                
                                # Non-Pareto solutions
                                non_pareto = df[~df['is_pareto']]
                                fig.add_trace(go.Scatter(
                                    x=non_pareto['satisfaction_score'],
                                    y=non_pareto['fairness_score'],
                                    mode='markers',
                                    name='All solutions',
                                    marker=dict(
                                        color='lightgray',
                                        size=5,
                                        opacity=0.5
                                    ),
                                    hovertemplate='Satisfaction: %{x:.3f}<br>Fairness: %{y:.3f}<extra></extra>'
                                ))
                                
                                # Pareto solutions
                                pareto = df[df['is_pareto']]
                                if not pareto.empty:
                                    fig.add_trace(go.Scatter(
                                        x=pareto['satisfaction_score'],
                                        y=pareto['fairness_score'],
                                        mode='markers',
                                        name=f'Pareto frontier ({len(pareto)} solutions)',
                                        marker=dict(
                                            color='red',
                                            size=8,
                                            symbol='star'
                                        ),
                                        hovertemplate='Satisfaction: %{x:.3f}<br>Fairness: %{y:.3f}<extra></extra>'
                                    ))
                                
                                fig.update_layout(
                                    title='Pareto Frontier: Satisfaction vs Fairness',
                                    xaxis_title='Satisfaction Score',
                                    yaxis_title='Fairness Score',
                                    hovermode='closest',
                                    height=600,
                                    showlegend=True
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                                
                                st.info(f"⭐ **Pareto Frontier**: {len(pareto)} solutions where neither satisfaction nor fairness can be improved without degrading the other.")
                                break  # Only show first valid CSV
                    except Exception as e:
                        st.warning(f"Could not generate interactive Pareto plot: {e}")
                
                # Show output directory
                st.info(f"📁 All files saved to: `{output_dir}`")
                
            else:
                st.error("❌ Grid search failed!")
                st.text("Error output:")
                st.code(result.stderr)
                
        except subprocess.TimeoutExpired:
            st.error("⏱️ Grid search timed out (exceeded 1 hour)")
        except Exception as e:
            st.error(f"❌ Error running grid search: {e}")
            import traceback
            st.code(traceback.format_exc())

