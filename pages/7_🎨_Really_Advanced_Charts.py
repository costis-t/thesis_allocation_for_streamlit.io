"""
Really Advanced Charts page for Thesis Allocation System
"""
import streamlit as st
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

from streamlit_dashboard_pages.shared import (
    initialize_session_state,
    calculate_fairness_score,
    calculate_satisfaction_metrics
)

# Initialize session state
initialize_session_state()


st.header("🚀 Really Advanced Charts")

st.write("**Advanced analytical visualizations for comprehensive allocation insights:**")

# Verbose explanations
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
                st.caption("💡 **What this shows**: How many students got each preference level.")
                try:
                    fig_funnel = create_preference_funnel(allocation_df)
                    st.plotly_chart(fig_funnel, use_container_width=True)
                except Exception as e:
                    st.error(f"Error generating funnel: {e}")
            
            with col2:
                st.write("**Cost Breakdown by Category**")
                st.write("Allocation composition by preference type")
                st.caption("💡 **What this shows**: The proportion of students assigned to each preference category.")
                try:
                    fig_pie = create_cost_breakdown_pie(allocation_df)
                    st.plotly_chart(fig_pie, use_container_width=True)
                except Exception as e:
                    st.error(f"Error generating pie chart: {e}")
            
            st.divider()
            st.write("**Cost Distribution Violin Plot**")
            st.write("Statistical distribution showing clustering and outliers")
            st.caption("💡 **What this shows**: The shape of cost distribution across students.")
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
                st.caption("💡 **What this shows**: Multiple fairness dimensions compared to ideal benchmarks.")
                try:
                    fig_radar = create_fairness_radar(metrics)
                    st.plotly_chart(fig_radar, use_container_width=True)
                except Exception as e:
                    st.error(f"Error generating radar: {e}")
            
            with col2:
                st.write("**Topic Demand vs Capacity**")
                st.write("Identifies bottleneck topics and capacity mismatches")
                st.caption("💡 **What this shows**: Each bar shows topic demand vs capacity.")
                try:
                    fig_capacity = create_topic_demand_vs_capacity(allocation_df, repo)
                    st.plotly_chart(fig_capacity, use_container_width=True)
                except Exception as e:
                    st.error(f"Error generating capacity chart: {e}")
        
        # TAB 3: Deep Dive Analysis
        with adv_tab3:
            st.write("**Student Satisfaction Scatter Plot**")
            st.write("Cost vs Preference Rank - each bubble is a student, colored by department")
            st.caption("💡 **What this shows**: Each bubble represents one student. X-axis = preference rank (lower is better), Y-axis = effective cost (lower is better).")
            
            # Auto-detect data ranges
            x_min_data = int(allocation_df['preference_rank'].min()) if len(allocation_df) > 0 else 0
            x_max_data = int(allocation_df['preference_rank'].max()) if len(allocation_df) > 0 else 200
            y_min_data = int(allocation_df['effective_cost'].min()) if len(allocation_df) > 0 else 0
            y_max_data = int(allocation_df['effective_cost'].max()) if len(allocation_df) > 0 else 200
            
            # Custom axis limits controls (pre-filled with data min/max)
            col_x, col_y = st.columns(2)
            with col_x:
                x_min = st.number_input("X-axis Min (Preference Rank)", value=x_min_data, min_value=-1, max_value=1000, step=1)
                x_max = st.number_input("X-axis Max (Preference Rank)", value=x_max_data, min_value=-1, max_value=1000, step=1)
            with col_y:
                y_min = st.number_input("Y-axis Min (Effective Cost)", value=y_min_data, min_value=-10000, max_value=10000, step=1)
                y_max = st.number_input("Y-axis Max (Effective Cost)", value=y_max_data, min_value=-10000, max_value=10000, step=1)
            
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
                st.caption("💡 **What this shows**: Heatmap showing coach-topic assignments.")
                try:
                    fig_coach = create_coach_specialization_heatmap(allocation_df, repo)
                    st.plotly_chart(fig_coach, use_container_width=True)
                except Exception as e:
                    st.error(f"Error generating coach heatmap: {e}")
            
            with col2:
                st.write("**Department Diversity Analysis**")
                st.write("Topic diversity within departments (Shannon Entropy)")
                st.caption("💡 **What this shows**: How diverse topics are within each department.")
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
