import streamlit as st

# Import homework pages
import HWs.HW1 as HW1
import HWs.HW2 as HW2
import HWs.HW3 as HW3

st.set_page_config(page_title="HW Manager", page_icon="📚")

st.title("📚 HW Manager")
st.sidebar.title("Navigation")

# Sidebar menu
page = st.sidebar.radio(
    "Choose Homework",
    ("HW1 – Document Q&A", "HW2 – URL Summarizer", "HW3 – Chatbot with Memory"),
)

# Route to the right page
if page.startswith("HW1"):
    HW1.render()
elif page.startswith("HW2"):
    HW2.render()
elif page.startswith("HW3"):
    HW3.render()
