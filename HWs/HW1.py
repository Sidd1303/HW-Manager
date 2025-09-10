import io
import streamlit as st
from openai import OpenAI
from pypdf import PdfReader

def _read_pdf(uploaded_file) -> str:
    file_bytes = uploaded_file.getvalue()
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")
    return "\n\n".join(pages)

def _read_txt(uploaded_file) -> str:
    return uploaded_file.getvalue().decode("utf-8", errors="ignore")

def _get_client():
    key_from_secrets = st.secrets.get("OPENAI_API_KEY")
    if key_from_secrets:
        return OpenAI(api_key=key_from_secrets), True

    api_key = st.text_input("🔑 OpenAI API Key", type="password")
    if not api_key:
        st.info("Please add your OpenAI API key to continue.", icon="🗝️")
        return None, False
    return OpenAI(api_key=api_key), True

def render():
    st.header("📄 HW1 – Document Q&A")
    st.write("Upload a **.pdf** or **.txt** file, ask a question, and compare answers across models.")

    client, ready = _get_client()
    if not ready:
        return

    uploaded_file = st.file_uploader("Upload a .pdf or .txt file", type=("pdf", "txt"))

    question = st.text_area(
        "Now ask a question about the document!",
        placeholder="Is this course hard?",
        disabled=not uploaded_file,
    )

    if not (uploaded_file and question):
        return

    ext = uploaded_file.name.split(".")[-1].lower()
    if ext == "txt":
        document = _read_txt(uploaded_file)
    elif ext == "pdf":
        document = _read_pdf(uploaded_file)
    else:
        st.error("Unsupported file type.")
        return

    if not document.strip():
        st.error("No extractable text found in this file.")
        return

    messages = [
        {
            "role": "system",
            "content": "Answer ONLY using the provided document. If info is missing, say so briefly."
        },
        {
            "role": "user",
            "content": f"Document:\n{document}\n\n---\n\nQuestion: {question}",
        }
    ]

    models = [
        "gpt-5",        # flagship
        "gpt-5mini",   # cheaper / faster
    ]

    for m in models:
        st.subheader(f"Answer with `{m}`")
        try:
            stream = client.chat.completions.create(
                model=m,
                messages=messages,
                stream=True,
            )
            st.write_stream(stream)
        except Exception as e:
            st.error(f"{m} failed: {e}")
