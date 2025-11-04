"""
Configuration page for  Thesis Allocation System
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

from shared import initialize_session_state

# Initialize session state
initialize_session_state()

# Set page config
st.set_page_config(
    page_title="Configuration - Thesis Allocation Dashboard",
    page_icon="⚙️",
    layout="wide"
)

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
        0, 100, st.session_state.config_tier2_cost,
        help="Penalty for Tier 2 topics (moderately preferred). Lower = algorithm prefers Tier 2. Higher = algorithm avoids Tier 2. Default: 1"
    )
    tier3_cost = st.slider(
        "Tier 3 Cost", 
        0, 100, st.session_state.config_tier3_cost,
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
    # Ensure rank2_cost >= rank1_cost
    max_rank2 = max(rank1_cost + 1, 201)  # Ensure max > min
    rank2_cost = st.slider(
        "2nd Choice Cost", 
        rank1_cost, max_rank2, max(rank1_cost, st.session_state.config_rank2_cost),
        help=f"Penalty for 2nd choice (student's #2 ranked topic). Must be ≥ {rank1_cost} (1st choice cost)."
    )

with col2:
    # Ensure rank3_cost >= rank2_cost
    max_rank3 = max(rank2_cost + 1, 201)  # Ensure max > min
    rank3_cost = st.slider(
        "3rd Choice Cost", 
        rank2_cost, max_rank3, max(rank2_cost, st.session_state.config_rank3_cost),
        help=f"Penalty for 3rd choice (student's #3 ranked topic). Must be ≥ {rank2_cost} (2nd choice cost)."
    )
    # Ensure rank4_cost >= rank3_cost
    max_rank4 = max(rank3_cost + 1, 201)  # Ensure max > min
    rank4_cost = st.slider(
        "4th Choice Cost", 
        rank3_cost, max_rank4, max(rank3_cost, st.session_state.config_rank4_cost),
        help=f"Penalty for 4th choice (student's #4 ranked topic). Must be ≥ {rank3_cost} (3rd choice cost)."
    )

with col3:
    # Ensure rank5_cost >= rank4_cost
    max_rank5 = max(rank4_cost + 1, 201)  # Ensure max > min
    rank5_cost = st.slider(
        "5th Choice Cost", 
        rank4_cost, max_rank5, max(rank4_cost, st.session_state.config_rank5_cost),
        help=f"Penalty for 5th choice (student's #5 ranked topic). Must be ≥ {rank4_cost} (4th choice cost)."
    )
    
    # Display current constraint satisfaction
    if rank1_cost <= rank2_cost <= rank3_cost <= rank4_cost <= rank5_cost:
        st.success(f"✓ Monotonic costs: {rank1_cost} ≤ {rank2_cost} ≤ {rank3_cost} ≤ {rank4_cost} ≤ {rank5_cost}")
    else:
        st.error(f"❌ Non-monotonic costs detected! This violates preference logic.")

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
    from datetime import datetime
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
    
    # Save main config file (overwrites existing)
    config_file_path = project_root / "config_streamlit.json"
    config.save_json(str(config_file_path))
    
    # Auto-save timestamped version to data/output directory
    output_dir = project_root / "data" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    timestamped_config_path = output_dir / f"config_{timestamp}.json"
    config.save_json(str(timestamped_config_path))
    
    # Display file paths
    st.success(f"✅ Configuration saved!")
    st.write(f"  • Active config: `{config_file_path}`")
    st.write(f"  • Timestamped backup: `{timestamped_config_path}`")
    
    # Reload configuration from file to update session state
    try:
        if config_file_path.exists():
            config = AllocationConfig.load_json(str(config_file_path))
            # Update session state
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
            st.session_state.config_enable_topic_overflow = config.capacity.enable_topic_overflow
            st.session_state.config_enable_coach_overflow = config.capacity.enable_coach_overflow
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
        st.warning(f"⚠️ Could not reload config: {e}")
    
    # Mark that config was saved
    st.session_state.config_saved = True
    
    # Show success message at the bottom
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style='text-align: center; padding: 20px; background: #2d2d2d; border: 2px solid #4caf50; border-radius: 10px;'>
            <h3 style='color: #4caf50; margin: 0;'>✓ Configuration Saved!</h3>
            <p style='color: #e0e0e0; margin: 10px 0 0 0;'>Ready to run allocation with these settings?</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Direct button with JavaScript navigation
        st.markdown("""
        <script>
        function navigateToRunAllocation() {
            window.location.href = "/3_Run_Allocation";
        }
        </script>
        """, unsafe_allow_html=True)
        
        if st.button("🚀 Go to Run Allocation", type="primary", use_container_width=True, key="go_to_run_nav"):
            # Reset flag
            st.session_state.config_saved = False
            # Use JavaScript to navigate
            st.markdown('<meta http-equiv="refresh" content="0; url=/3_Run_Allocation">', unsafe_allow_html=True)
