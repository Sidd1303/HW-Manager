# --- sqlite3 patch for Chroma in Codespaces/Cloud ---
__import__("pysqlite3")
import sys as _sys
_sys.modules["sqlite3"] = _sys.modules.pop("pysqlite3")

import streamlit as st

# Import homework pages (each exposes render())
from HWs import HW1, HW2, HW3, HW4, HW5

st.sidebar.title("📚 Homework Selector")
choice = st.sidebar.radio(
    "Select Homework",
    [
        "HW1 – Document Q&A",
        "HW2 – Multi-Provider Q&A",
        "HW3 – Chatbot with Memory",
        "HW4 – Orgs RAG Chatbot",
        "HW5 – Orgs RAG Chatbot (Chat + Evaluation)",
        "HW7 – News Info Bot",
    ],
)

if choice.startswith("HW1"):
    HW1.render()
elif choice.startswith("HW2"):
    HW2.render()
elif choice.startswith("HW3"):
    HW3.render()
elif choice.startswith("HW4"):
    HW4.render()
elif choice.startswith("HW5"):
    HW5.render()
else:
    HW7.render()