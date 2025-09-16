import streamlit as st
import requests
from bs4 import BeautifulSoup
import time

# LangChain memory
from langchain.memory import (
    ConversationSummaryMemory,
    ConversationTokenBufferMemory,
    ConversationBufferWindowMemory,   # ✅ use windowed buffer
)
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI


# ------------------ Utility: Fetch & Clean Web Page ------------------
def read_url(url: str) -> str:
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        return " ".join(soup.get_text().split())
    except Exception as e:
        return f"[Error reading {url}: {e}]"


# ------------------ Streamlit Page ------------------
def render():
    st.header("HW3 – Chatbot with Memory")
    st.write("Ask questions about one or two URLs. The bot remembers context using different memory strategies.")

    # Sidebar controls
    url1 = st.sidebar.text_input("First URL")
    url2 = st.sidebar.text_input("Second URL (optional)")

    provider = st.sidebar.selectbox("Choose provider", ["OpenAI", "Anthropic", "Google Gemini"])
    model_type = st.sidebar.radio("Model type", ["Flagship", "Lite"])

    memory_type = st.sidebar.selectbox(
        "Conversation memory type",
        ["Buffer (6 exchanges)", "Summary memory", "Token buffer (2000 tokens)"],
    )

    # Pick model name
    models = {
        "OpenAI": {
            "Flagship": "gpt-4.1",           # ✅ non-reasoning flagship
            "Lite": "gpt-5-chat-latest",     # ✅ non-reasoning lite
        },
        "Anthropic": {
            "Flagship": "claude-opus-4-1",
            "Lite": "claude-3-5-haiku-latest",
        },
        "Google Gemini": {
            "Flagship": "gemini-2.5-pro",
            "Lite": "gemini-2.5-flash-lite",
        },
    }
    model_name = models[provider][model_type]

    # ------------------ LLM Setup ------------------
    if provider == "OpenAI":
        llm = ChatOpenAI(
            model=model_name,
            api_key=st.secrets["OPENAI_API_KEY"],
            streaming=True,
        )
    elif provider == "Anthropic":
        llm = ChatAnthropic(
            model=model_name,
            api_key=st.secrets["ANTHROPIC_API_KEY"],
            streaming=True,
        )
    else:
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            api_key=st.secrets["GEMINI_API_KEY"],
            streaming=True,
        )

    # ------------------ Persistent Memory ------------------
    if "memory" not in st.session_state or st.session_state.get("memory_type") != memory_type:
        if memory_type == "Buffer (6 exchanges)":
            st.session_state.memory = ConversationBufferWindowMemory(k=6, return_messages=True)
        elif memory_type == "Summary memory":
            st.session_state.memory = ConversationSummaryMemory(llm=llm)
        else:
            st.session_state.memory = ConversationTokenBufferMemory(llm=llm, max_token_limit=2000)
        st.session_state.memory_type = memory_type  # track type
        st.session_state.chat_history = []  # reset history

    memory = st.session_state.memory

    # ------------------ Debug Panel ------------------
    with st.sidebar.expander("🔍 Debug: Memory Contents"):
        st.write(memory.load_memory_variables({}))

    # ------------------ Display past chat history ------------------
    if st.session_state.get("chat_history"):
        for role, content in st.session_state.chat_history:
            st.chat_message(role).write(content)

    # ------------------ Chat UI ------------------
    question = st.chat_input("Ask me something about the URLs")
    if question:
        # Collect documents
        docs = []
        if url1:
            docs.append(read_url(url1))
        if url2:
            docs.append(read_url(url2))
        context = "\n\n".join(docs)

        # Load past memory (✅ FIX: extract just "history")
        history = memory.load_memory_variables({}).get("history", "")
        prompt = f"Documents:\n{context}\n\nConversation so far:\n{history}\n\nUser: {question}\nAssistant:"

        # Display user message
        st.chat_message("user").write(question)
        st.session_state.chat_history.append(("user", question))

        # Stream assistant response with typing effect
        with st.chat_message("assistant"):
            try:
                placeholder = st.empty()
                response_text = ""

                for chunk in llm.stream(prompt):
                    if hasattr(chunk, "content") and chunk.content:
                        response_text += chunk.content
                        placeholder.markdown(response_text)   # update gradually
                        time.sleep(0.03)  # ⏳ typing effect

                # Save Q&A into memory
                memory.save_context({"input": question}, {"output": response_text})
                st.session_state.chat_history.append(("assistant", response_text))
            except Exception as e:
                st.error(f"{provider} failed: {e}")
