"""
Data Explorer page for Thesis Allocation System
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

from streamlit_dashboard_pages.shared import initialize_session_state

# Initialize session state
initialize_session_state()


st.header("🔍 Data Explorer")
st.info("""
📊 **Explore your input data:**
- **Students**: View student preferences and profiles
- **Topics**: Check topic descriptions and capacities
- **Coaches**: Review coach assignments and capacity
""")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📊 Students")
    students_file = st.file_uploader("Upload Students CSV", type=['csv'], key="explorer_students")
    if students_file:
        df = pd.read_csv(students_file)
        st.write(f"**Total Students:** {len(df)}")
        st.dataframe(df.head(10), use_container_width=True)
    else:
        # Try to load default file
        default_students = st.session_state.students_file
        if default_students and default_students.exists():
            df = pd.read_csv(default_students)
            st.info(f"📄 Using default file: {default_students}")
            st.write(f"**Total Students:** {len(df)}")
            st.dataframe(df.head(10), use_container_width=True)

with col2:
    st.subheader("🎯 Topics")
    capacities_file = st.file_uploader("Upload Capacities CSV", type=['csv'], key="explorer_capacities")
    if capacities_file:
        df = pd.read_csv(capacities_file)
        st.write(f"**Total Topics:** {len(df)}")
        st.dataframe(df.head(10), use_container_width=True)
    else:
        # Try to load default file
        default_capacities = st.session_state.capacities_file
        if default_capacities and default_capacities.exists():
            df = pd.read_csv(default_capacities)
            st.info(f"📄 Using default file: {default_capacities}")
            st.write(f"**Total Topics:** {len(df)}")
            st.dataframe(df.head(10), use_container_width=True)

with col3:
    st.subheader("👥 Coaches")
    if capacities_file:
        df = pd.read_csv(capacities_file)
        coaches_df = df[['coach_id', 'maximum_students_per_coach']].drop_duplicates('coach_id')
        st.write(f"**Total Coaches:** {len(coaches_df)}")
        st.dataframe(coaches_df.head(10), use_container_width=True)
    else:
        default_capacities = st.session_state.capacities_file
        if default_capacities and default_capacities.exists():
            df = pd.read_csv(default_capacities)
            # Try to find coach columns with various possible names
            coach_cols = [col for col in df.columns if 'coach' in col.lower()]
            if coach_cols:
                try:
                    coaches_df = df[[coach_cols[0], coach_cols[1]]].drop_duplicates(coach_cols[0])
                    st.info(f"📄 Using default file: {default_capacities}")
                    st.write(f"**Total Coaches:** {len(coaches_df)}")
                    st.dataframe(coaches_df.head(10), use_container_width=True)
                except:
                    st.info(f"📄 Using default file: {default_capacities}")
                    st.write("**Coaches:** Check capacities file above")
