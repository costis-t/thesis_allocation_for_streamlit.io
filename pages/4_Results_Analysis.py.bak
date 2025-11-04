"""
Results Analysis page for Thesis Allocation System
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

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

# Initialize session state
initialize_session_state()


# Helper functions for charts
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


def create_hot_topics_chart(repo):
    """Create Hot Topics chart showing how many students ranked each topic as pref1, pref2, and pref3."""
    from collections import defaultdict
    
    # Build preference counts per topic (pref1, pref2, and pref3)
    topic_pref_counts = defaultdict(lambda: {'pref1': 0, 'pref2': 0, 'pref3': 0})
    
    for student_id, student in repo.students.items():
        if not student.plan:
            continue
        
        for idx, topic_id in enumerate(student.ranks[:3], start=1):
            topic_pref_counts[topic_id][f'pref{idx}'] += 1
    
    if not repo.topics:
        return None
    
    # Prepare data for ALL topics (even those with no preferences)
    topic_data = []
    for topic_id in sorted(repo.topics.keys()):  # Get all topics from repo, sorted
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
    
    # Create stacked bar chart
    topic_names = [d['Topic'] for d in topic_data]
    
    fig = go.Figure()
    
    # Add bars for pref1, pref2, and pref3
    fig.add_trace(go.Bar(
        name='1st Choice',
        x=topic_names,
        y=[d['Pref 1'] for d in topic_data],
        marker_color='#2ecc71',  # Green for 1st choice
        hovertemplate='<b>%{x}</b><br>1st Choice: %{y}<extra></extra>'
    ))
    fig.add_trace(go.Bar(
        name='2nd Choice',
        x=topic_names,
        y=[d['Pref 2'] for d in topic_data],
        marker_color='#3498db',  # Blue for 2nd choice
        hovertemplate='<b>%{x}</b><br>2nd Choice: %{y}<extra></extra>'
    ))
    fig.add_trace(go.Bar(
        name='3rd Choice',
        x=topic_names,
        y=[d['Pref 3'] for d in topic_data],
        marker_color='#f39c12',  # Orange for 3rd choice
        hovertemplate='<b>%{x}</b><br>3rd Choice: %{y}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Hot Topics - Student Preferences (All Topics)',
        xaxis_title='Topic',
        yaxis_title='Number of Students',
        barmode='stack',
        height=600,
        xaxis_tickangle=-45,
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="right", x=1.02)
    )
    
    return fig


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
    
    # Hot Topics Chart
    if st.session_state.last_repos is not None:
        st.markdown("---")
        st.subheader("🔥 Hot Topics")
        st.caption("Most popular topics ranked by student preference (top 20). Shows how many students had each topic as 1st, 2nd, 3rd, 4th, or 5th choice.")
        fig_hot = create_hot_topics_chart(st.session_state.last_repos)
        if fig_hot:
            st.plotly_chart(fig_hot, use_container_width=True)
    
    # Allocation details table
    st.divider()
    st.subheader("📋 Allocation Details")
    st.dataframe(allocation_df, use_container_width=True)
    
    # Download results
    st.divider()
    st.subheader("📥 Download Results")
    download_combined_results(allocation_df, summary_text)
    
    # Bulk download all charts
    st.markdown("---")
    st.subheader("📦 Download All Charts")
    st.info("📸 Download all visualizations from Advanced Charts and Really Advanced Charts pages as a single zip file.")
    
    if st.button("🎁 Download All Charts as ZIP", type="primary", help="Generate and download all charts from Advanced Charts and Really Advanced Charts"):
        with st.spinner("🔄 Generating all charts... This may take a moment."):
            try:
                zip_data = generate_all_charts_zip()
                if zip_data:
                    from datetime import datetime
                    filename = f"all_charts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
                    st.download_button(
                        label="✅ Download ZIP File",
                        data=zip_data,
                        file_name=filename,
                        mime="application/zip",
                        key="download_all_charts"
                    )
                    st.success(f"✅ Generated {filename} with all charts!")
                else:
                    st.error("❌ Could not generate charts. Please ensure you have run an allocation first.")
            except Exception as e:
                st.error(f"❌ Error generating charts: {str(e)}")
                import traceback
                st.error(traceback.format_exc())
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
