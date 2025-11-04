"""
Compare Allocations page for Thesis Allocation System
"""
import streamlit as st
import pandas as pd
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
    calculate_satisfaction_metrics,
    calculate_fairness_score
)

# Initialize session state
initialize_session_state()


st.header("⚖️ Compare Allocations")
st.write("**Compare multiple allocation results with different cost configurations to analyze trade-offs and sensitivity.**")

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
else:
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
        st.markdown("**📊 Current Results**")
        
        # Calculate metrics for current allocation
        if st.session_state.last_allocation is not None:
            df = st.session_state.last_allocation
            satisfaction_metrics = calculate_satisfaction_metrics(df)
            fairness_metrics = calculate_fairness_score(df)
            
            st.metric("Total Assigned", len(df))
            st.metric("1st Choice %", f"{satisfaction_metrics.get('1st_choice_pct', 0):.1f}%")
            st.metric("Fairness Score", f"{fairness_metrics.get('fairness_score', 0):.2f}")
            st.metric("Avg Cost", f"{df['effective_cost'].mean():.2f}")
    
    st.divider()
    st.subheader("📊 Side-by-Side Comparison")
    st.info("💡 **Note**: To compare multiple allocations, run allocations with different configurations in the 'Run Allocation' page and compare the results here.")
    
    # Show current allocation details
    if st.session_state.last_allocation is not None:
        st.write("**Current Allocation Summary:**")
        df = st.session_state.last_allocation
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Students Assigned", len(df))
            st.metric("1st Choices", len(df[df['preference_rank'] == 10]))
            st.metric("2nd Choices", len(df[df['preference_rank'] == 11]))
        
        with col2:
            st.metric("Total Cost", f"{df['effective_cost'].sum():.0f}")
            st.metric("Avg Cost", f"{df['effective_cost'].mean():.2f}")
            st.metric("Std Dev Cost", f"{df['effective_cost'].std():.2f}")
        
        with col3:
            first_choice_pct = (len(df[df['preference_rank'] == 10]) / len(df) * 100) if len(df) > 0 else 0
            st.metric("1st Choice %", f"{first_choice_pct:.1f}%")
            fairness = calculate_fairness_score(df)
            st.metric("Fairness", f"{fairness.get('fairness_score', 0):.2f}")
            st.metric("Unranked", len(df[df['preference_rank'] == 999]))
        
        st.divider()
        st.write("**Recommendation**: ")
        st.info("""
        Run additional allocations with different cost configurations and compare:
        - Lower ranked choice costs = prioritize preferences
        - Higher unranked cost = avoid unranked assignments
        - Enable Top-2 Bias = strongly prefer 1st & 2nd choices
        
        Use the **Configuration** page to adjust settings, then **Run Allocation** to create new results.
        """)
