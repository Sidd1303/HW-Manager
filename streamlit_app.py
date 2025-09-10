import streamlit as st
from importlib import import_module

st.set_page_config(page_title="HW manager", page_icon="🧰", layout="centered")
st.title("HW manager")
st.caption("Select a homework page from the sidebar.")

page = st.sidebar.radio("Pages", ["HW2 (URL Summarizer)", "HW1 (Document Q&A)"], index=0)

if page.startswith("HW2"):
    import_module("HWs.HW2").render()
else:
    import_module("HWs.HW1").render()
