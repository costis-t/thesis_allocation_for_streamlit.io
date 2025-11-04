"""
Network Graphs page for Thesis Allocation System
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import sys
import tempfile
import os
import subprocess

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

from shared import initialize_session_state

# Initialize session state
initialize_session_state()

# Set page config
st.set_page_config(
    page_title="Network Graphs - Thesis Allocation Dashboard",
    page_icon="🔗",
    layout="wide"
)

st.header("🔗 Network Flow Graphs")

st.info("""
📊 **Network Flow Visualizations:**

View the min-cost max-flow network structure from your allocation:

- **Detailed View**: Individual student paths (hover to highlight)
- **Bundled View**: Aggregated flow patterns with curved edges
- **Interactive**: Hover over any edge to see the complete student flow path
- **Color-Coded**: Edge colors represent student preference satisfaction

**Flow Structure**: SOURCE → Students → Topics → Coaches → SINK
""")

# Explanation of Preference Rank
with st.expander("📚 What is Preference Rank?", expanded=False):
    st.write("""
    **Preference Ranks** indicate how well each student's preferences were satisfied:
    
    - **Tier 1 (Green)**: Student got their top preferred topic
    - **Tier 2 (Blue)**: Student got their second tier preference
    - **Tier 3 (Yellow)**: Student got their third tier preference
    - **Ranked 1st-5th (Green→Red)**: Student got their 1st through 5th ranked choice
    - **Unranked (Gray)**: Student got a topic they didn't rank
    
    **Lower rank number = Better preference satisfaction** ✅
    """)

# Check if we have a cached allocation
if st.session_state.get('last_allocation_rows') is None:
    st.warning("⚠️ No allocation results found. Please run an allocation first from the 'Run Allocation' page.")
    st.info("""
    **How to generate a network graph:**
    1. Go to **Configuration** page and set your preferences
    2. Go to **Run Allocation** page and run the allocation
    3. Return here to view the network visualization
    """)
else:
    rows = st.session_state.last_allocation_rows
    
    # Create temporary CSV file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as f:
        # Write header
        f.write('student,assigned_topic,assigned_coach,department_id,preference_rank,effective_cost\n')
        
        # Write data
        for row in rows:
            f.write(f'{row.student},{row.assigned_topic},{row.assigned_coach},{row.department_id},{row.preference_rank},{row.effective_cost}\n')
        
        temp_csv = f.name
    
    # Debug: Show CSV info
    st.info(f"📊 Processing {len(rows)} student allocations...")
    
    # Generate network visualizations
    try:
        # Check if the viz script exists
        viz_script = project_root / "viz_network_flow.py"
        
        if viz_script.exists():
            # Run the visualization script
            with st.spinner("🔄 Generating network visualization..."):
                result = subprocess.run(
                    [
                        "python3", str(viz_script),
                        "--allocation", temp_csv,
                        "--output", str(project_root / "visualisations/network_flow.html")
                    ],
                    capture_output=True,
                    text=True,
                    cwd=str(project_root)
                )
            
            
            if result.returncode == 0:
                # Define the three HTML files
                html_files = {
                    "Main Network (Detailed + Bundled View)": "visualisations/network_flow_main.html",
                    "Multipartite Layout": "visualisations/network_flow_multipartite.html",
                    "Edge Colormap": "visualisations/network_flow_colormap.html"
                }
                
                # Display each graph separately
                for title, html_file in html_files.items():
                    html_path = project_root / html_file
                    
                    if html_path.exists():
                        st.markdown(f"### 📊 {title}")
                        
                        # Create columns for display and download
                        col1, col2 = st.columns([4, 1])
                        
                        with col2:
                            # Create a download button for the HTML file
                            import os
                            with open(html_path, 'r', encoding='utf-8') as f:
                                html_data = f.read()
                            st.download_button(
                                label="📥 Download",
                                data=html_data,
                                file_name=os.path.basename(html_file),
                                mime="text/html",
                                key=f"download_{html_file}"
                            )
                        
                        # Display the HTML
                        with open(html_path, 'r', encoding='utf-8') as f:
                            html_content = f.read()
                        
                        st.components.v1.html(html_content, height=700)
                        st.markdown("---")  # Separator between graphs
                    
                # Show message if any files are missing
                missing_files = [f for f, path in html_files.items() if not (project_root / path).exists()]
                if missing_files:
                    st.warning(f"⚠️ Could not generate: {', '.join(missing_files)}")
                
                # Generate Atlas visualization
                st.markdown("## 🗺️ Allocation Pattern Reference")
                st.info("""
                **Pattern Reference Guide**
                
                This section shows example graph structures that can appear in your allocation network. 
                Each pattern represents how students, topics, and coaches can be connected:
                
                - **Star Pattern**: A popular topic assigned to many students
                - **Hub Pattern**: One coach supervising multiple projects
                - **Path Pattern**: Sequential allocation chains
                - **Parallel Pattern**: Independent allocation paths
                
                Hover over any node to see what it represents in the allocation context.
                """)
                
                atlas_script = project_root / "viz_atlas.py"
                if atlas_script.exists():
                    with st.spinner("🔄 Generating graph atlas..."):
                        atlas_result = subprocess.run(
                            [
                                "python3", str(atlas_script),
                                "--allocation", temp_csv,
                                "--output", str(project_root / "visualisations/atlas.html")
                            ],
                            capture_output=True,
                            text=True,
                            cwd=str(project_root)
                        )
                    
                    if atlas_result.returncode == 0:
                        atlas_path = project_root / "visualisations/atlas.html"
                        analytical_atlas_path = project_root / "visualisations/analytical_atlas.html"
                        patterns_path = project_root / "visualisations/patterns.html"
                        
                        if atlas_path.exists():
                            st.markdown("### 📚 Theoretical Pattern Reference")
                            st.caption("Example patterns that could appear in any allocation")
                            with open(atlas_path, 'r', encoding='utf-8') as f:
                                atlas_html = f.read()
                            st.components.v1.html(atlas_html, height=1000)
                            st.markdown("---")
                        
                        if analytical_atlas_path.exists():
                            st.markdown("### 🔍 Analytical Atlas from Your Allocation")
                            st.caption("Actual patterns detected in your student-topic-coach network")
                            with open(analytical_atlas_path, 'r', encoding='utf-8') as f:
                                analytical_html = f.read()
                            st.components.v1.html(analytical_html, height=800)
                            st.markdown("---")
                        
                        if patterns_path.exists():
                            st.markdown("### 📈 Allocation Pattern Analysis")
                            with open(patterns_path, 'r', encoding='utf-8') as f:
                                patterns_html = f.read()
                            st.components.v1.html(patterns_html, height=500)
                    
                # Graph information section
                st.markdown("### 📊 Graph Information")
                st.write(f"- **Total Students**: {len(rows)}")
                st.write(f"- **Total Topics**: {len(set(row.assigned_topic for row in rows))}")
                st.write(f"- **Total Coaches**: {len(set(row.assigned_coach for row in rows))}")
                
                # Preference satisfaction breakdown
                with st.expander("📈 Preference Satisfaction Breakdown"):
                    pref_counts = {}
                    for row in rows:
                        rank = row.preference_rank
                        pref_counts[rank] = pref_counts.get(rank, 0) + 1
                    
                    # Create a DataFrame for display
                    pref_df = pd.DataFrame([
                        {'Rank': rank, 'Count': count, 'Percentage': f'{(count/len(rows)*100):.1f}%'}
                        for rank, count in sorted(pref_counts.items())
                    ])
                    st.dataframe(pref_df, use_container_width=True)
                
                # Statistics
                with st.expander("📊 Allocation Statistics"):
                    avg_cost = sum(row.effective_cost for row in rows) / len(rows)
                    st.metric("Average Cost", f"{avg_cost:.2f}")
                    
                    # Students who got their first choice
                    first_choice = sum(1 for row in rows if row.preference_rank == 10)
                    st.metric("Got 1st Choice", f"{first_choice} ({first_choice/len(rows)*100:.1f}%)")
                    
                    # Students who got top 3 choices
                    top3 = sum(1 for row in rows if row.preference_rank <= 12)
                    st.metric("Got Top 3", f"{top3} ({top3/len(rows)*100:.1f}%)")
            else:
                st.error("HTML files not found after generation")
        else:
            st.error(f"Failed to generate visualization: {result.stderr}")
            st.code(result.stdout)
        if not viz_script.exists():
            st.error(f"Visualization script not found at: {viz_script}")
            
    except Exception as e:
        st.error(f"Error generating visualization: {str(e)}")
        import traceback
        with st.expander("Error Details"):
            st.code(traceback.format_exc())
    finally:
        # Clean up temp file
        try:
            os.unlink(temp_csv)
        except:
            pass

# Footer
st.markdown("---")
st.markdown("💡 **Tip**: Hover over any edge in the network graph to highlight that student's complete flow path from SOURCE to SINK.")

