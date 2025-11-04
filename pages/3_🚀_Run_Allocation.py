"""
Run Allocation page for Thesis Allocation System
"""
import streamlit as st
import pandas as pd
from pathlib import Path
import sys
import tempfile
import shutil
from datetime import datetime
from collections import Counter
import io

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


st.header("🚀 Run Allocation")
st.write("Run thesis allocation directly from the dashboard with live progress tracking.")

# Show if configuration is being used
config_file_path = project_root / "config_streamlit.json"
if config_file_path.exists() and hasattr(st.session_state, 'config_algorithm'):
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
with st.expander("⚙️ Active Configuration Settings"):
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
    
    st.info("💡 These settings are automatically updated when you change them in ⚙️ Configuration and click 💾 Save Configuration.")

st.divider()

# Validation section
if (students_file and capacities_file) or (st.session_state.students_file.exists() and st.session_state.capacities_file.exists()):
    st.subheader("✓ Validation")
    col1, col2 = st.columns(2)
    
    with col1:
        if students_file:
            students_df = pd.read_csv(students_file)
            st.write(f"**Students:** {len(students_df)} records (uploaded file)")
        else:
            # Use default file
            students_df = pd.read_csv(st.session_state.students_file)
            st.write(f"**Students:** {len(students_df)} records (default file)")
        st.write(f"**Columns:** {', '.join(students_df.columns.tolist()[:5])}")
    
    with col2:
        if capacities_file:
            capacities_df = pd.read_csv(capacities_file)
            st.write(f"**Capacities:** {len(capacities_df)} records (uploaded file)")
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
                 # Create persistent output directory  
                output_dir = project_root / "data" / "output"
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # Use persistent file paths
                tmpdir = output_dir
                students_path = output_dir / f"students_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                capacities_path = output_dir / f"capacities_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                output_path = output_dir / f"allocation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                summary_path = output_dir / f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                
                # Use uploaded files or default files
                if students_file is not None:
                    students_path.write_text(students_file.getvalue().decode())
                    st.info(f"📄 Using uploaded students file")
                else:
                    # Copy default students file
                    shutil.copy2(st.session_state.students_file, students_path)
                    st.info(f"📄 Using default students file: {st.session_state.students_file}")
                
                if capacities_file is not None:
                    capacities_path.write_text(capacities_file.getvalue().decode())
                    st.info(f"📄 Using uploaded capacities file")
                else:
                    # Copy default capacities file
                    shutil.copy2(st.session_state.capacities_file, capacities_path)
                    st.info(f"📄 Using default capacities file: {st.session_state.capacities_file}")
                
                # Load data
                st.info("📂 Loading data...")
                from allocator.data_repository import DataRepository
                repo = DataRepository(
                    str(students_path),
                    str(capacities_path),
                    str(output_dir / f"overrides_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv") if overrides_file else None
                )
                if overrides_file:
                    overrides_path = output_dir / f"overrides_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                    overrides_path.write_text(overrides_file.getvalue().decode())
                repo.load()
                
                st.info(f"✓ Loaded {len(repo.students)} students, {len(repo.topics)} topics")
                
                # Validate
                st.info("🔍 Validating data...")
                from allocator.validation import InputValidator
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
                from allocator.allocation_model_ilp import AllocationConfig as LegacyAllocationConfig
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
                    from allocator.allocation_model_ilp import AllocationModelILP
                    model = AllocationModelILP(
                        students=repo.students,
                        topics=repo.topics,
                        coaches=repo.coaches,
                        departments=repo.departments,
                        pref_model=pref_model,
                        cfg=legacy_cfg
                    )
                elif run_algorithm == "flow":
                    from allocator.allocation_model_flow import AllocationModelFlow
                    model = AllocationModelFlow(
                        students=repo.students,
                        topics=repo.topics,
                        coaches=repo.coaches,
                        departments=repo.departments,
                        pref_model=pref_model,
                        cfg=legacy_cfg
                    )
                else:  # hybrid
                    from allocator.allocation_model_ilp import AllocationModelILP
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
                
                # Mark allocation as complete
                st.session_state.allocation_complete = True
                
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
                
                # Save files to disk for persistence
                allocation_df.to_csv(output_path, index=False)
                summary_path.write_text(summary_text)
                st.success(f"✅ Files saved to disk:")
                st.write(f"  • Allocation: `{output_path}`")
                st.write(f"  • Summary: `{summary_path}`")
                
                # Download buttons
                st.divider()
                download_combined_results(allocation_df, summary_text)
                
                # Store in session for Results Analysis
                st.session_state.last_allocation = allocation_df
                st.session_state.last_summary = summary_text
                st.session_state.last_allocation_timestamp = datetime.now().isoformat()
                
                # Show navigation button to Results Analysis
                st.markdown("---")
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.markdown("""
                    <div style='text-align: center; padding: 20px; background: #2d2d2d; border: 2px solid #4caf50; border-radius: 10px;'>
                        <h3 style='color: #4caf50; margin: 0;'>✓ Allocation Complete!</h3>
                        <p style='color: #e0e0e0; margin: 10px 0 0 0;'>Ready to view detailed analysis?</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if st.button("📊 Go to Results Analysis", type="primary", use_container_width=True, key="go_to_results_nav"):
                        # Use JavaScript to navigate
                        st.markdown('<meta http-equiv="refresh" content="0; url=/4_Results_Analysis">', unsafe_allow_html=True)
            
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                import traceback
                st.error(traceback.format_exc())
else:
    if not st.session_state.students_file.exists() or not st.session_state.capacities_file.exists():
        st.warning("👆 Default files not found. Please upload students.csv and capacities.csv files to begin")
    else:
        st.warning("👆 Default files are available but validation failed. Please check the file formats or upload custom files.")
