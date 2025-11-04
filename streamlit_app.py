"""
Thesis Allocation System - Main Dashboard Entry Point
"""
import streamlit as st

# Must be the first Streamlit command
st.set_page_config(
    page_title="Thesis Allocation Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🎓 Thesis Allocation Dashboard")

st.write("""
Welcome to the Thesis Allocation System!

Use the sidebar to navigate to different pages.
""")

st.info("👈 Check the sidebar for navigation links to other pages!")
