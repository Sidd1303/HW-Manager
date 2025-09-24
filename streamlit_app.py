# --- sqlite3 patch for ChromaDB (must run before anything else) ---
__import__("pysqlite3")
import sys as _sys
_sys.modules["sqlite3"] = _sys.modules.pop("pysqlite3")

import streamlit as st

# Import homework apps (each must expose render())
import HWs.HW1 as HW1
import HWs.HW2 as HW2
import HWs.HW3 as HW3
import HWs.HW4 as HW4

st.sidebar.title("📚 Homework Selector")
choice = st.sidebar.radio(
    "Select Homework",
    [
        "HW1 – Document Q&A",
        "HW2 – Multi-Provider Q&A",
        "HW3 – Chatbot with Memory",
        "HW4 – Orgs RAG Chatbot",
    ],
)

if choice.startswith("HW1"):
    HW1.render()
elif choice.startswith("HW2"):
    HW2.render()
elif choice.startswith("HW3"):
    HW3.render()
else:
    HW4.render()
