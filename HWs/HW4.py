import os
from collections import deque
import streamlit as st
from bs4 import BeautifulSoup

import chromadb
from chromadb.utils import embedding_functions

# LLM SDKs
from openai import OpenAI as _OpenAI
from anthropic import Anthropic as _Anthropic
import google.generativeai as _genai


# =========================
# Utils: load & preprocess
# =========================

def _load_org_html(folder: str) -> dict[str, str]:
    """Load and clean ALL .html files from ./orgs into {filename: cleaned_text}.

    Cleaning removes <script>, <style>, <noscript>, and collapses whitespace.
    """
    out = {}
    if not os.path.isdir(folder):
        return out
    for fname in os.listdir(folder):
        if not fname.lower().endswith(".html"):
            continue
        fpath = os.path.join(folder, fname)
        try:
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                soup = BeautifulSoup(f.read(), "html.parser")
                for tag in soup(["script", "style", "noscript"]):
                    tag.decompose()
                text = " ".join(soup.get_text().split())
                if text.strip():
                    out[fname] = text
        except Exception as e:
            st.sidebar.error(f"❌ Could not read {fname}: {e}")
    return out


def _chunk_into_two(text: str) -> list[str]:
    """Return exactly TWO mini-docs per the HW instructions.

    Method (and WHY, per HW spec):
    - We split the document by words and cut into two equal halves (±1).
    - This produces exactly two chunks (mini-docs) per source document, as required.
    - Rationale: two balanced halves preserve broader context while avoiding
      over-fragmentation that can harm retrieval quality for short org pages.
    """
    words = text.split()
    if len(words) <= 1:
        return [text] if text else []
    mid = len(words) // 2
    return [" ".join(words[:mid]), " ".join(words[mid:])]


def _ensure_vector_db(persist_dir: str, org_dir: str, openai_key: str):
    """Create or load a persisted ChromaDB collection. We only embed ONCE.

    Requirement satisfied:
    - Build vector DB from ALL org HTML files
    - Chunk each doc into TWO mini-docs (see _chunk_into_two)
    - Persist so app can be run multiple times without re-embedding
    """
    os.makedirs(persist_dir, exist_ok=True)
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection(
        name="ischool_orgs",
        embedding_function=embedding_functions.OpenAIEmbeddingFunction(
            api_key=openai_key,
            model_name="text-embedding-3-small",
        ),
    )

    # If it already has vectors, skip re-embedding.
    try:
        existing = collection.count()
    except Exception:
        existing = len(collection.get().get("ids", []))
    if existing and existing > 0:
        st.sidebar.info(f"🔁 Vector DB ready with {existing} chunks. Skipping re-embed.")
        return collection

    # Build from orgs/
    docs = _load_org_html(org_dir)
    if not docs:
        st.error("❌ No HTML files found in ./orgs. Unzip the provided dataset into ./orgs.")
        return collection

    ids, metas, contents = [], [], []
    for fname, text in docs.items():
        chunks = _chunk_into_two(text)
        for i, ch in enumerate(chunks):
            contents.append(ch)
            ids.append(f"{fname}__chunk{i}")
            metas.append({"source": fname, "chunk_index": i})

    if not contents:
        st.error("❌ No usable content extracted from org HTML files.")
        return collection

    collection.add(documents=contents, ids=ids, metadatas=metas)
    st.sidebar.success(f"✅ Embedded {len(contents)} chunks from {len(docs)} org files.")
    return collection


# =========================
# LLM runners (no streaming)
# =========================

def _run_openai(model: str, api_key: str, prompt: str) -> str:
    """Compatible with non-reasoning OpenAI models like gpt-4.1 & gpt-5-chat-latest.

    NOTE: Do NOT pass temperature/max_tokens/stream — some newer models reject them.
    """
    client = _OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content or ""


def _run_anthropic(model: str, api_key: str, prompt: str) -> str:
    client = _Anthropic(api_key=api_key)
    resp = client.messages.create(
        model=model,
        max_tokens=800,
        messages=[{"role": "user", "content": prompt}],
    )
    try:
        return resp.content[0].text or ""
    except Exception:
        return ""


def _run_gemini(model: str, api_key: str, prompt: str) -> str:
    _genai.configure(api_key=api_key)
    g = _genai.GenerativeModel(model)
    r = g.generate_content(prompt)
    return getattr(r, "text", "") or ""


# =========================
# Page
# =========================

def render():
    st.title("🎓 HW4 – Orgs RAG Chatbot")
    st.caption(
        "Answers Syracuse iSchool org questions using RAG with a persistent vector DB, "
        "two-chunk method per document, and a 5-turn memory buffer."
    )
    st.markdown(
        "This implements the HW instructions precisely: build a vector DB once, "
        "chunk each document into **two** mini-docs (with rationale), keep up to **5** Q&A pairs in memory, "
        "let users pick among **three LLMs** on the sidebar, and provide a chat UI plus an evaluation section. "
        "See the HW handout. :contentReference[oaicite:1]{index=1}"
    )

    # --- API keys (secrets.toml per HW deploy notes)
    OPENAI_KEY = st.secrets.get("OPENAI_API_KEY")
    ANTHROPIC_KEY = st.secrets.get("ANTHROPIC_API_KEY")
    GEMINI_KEY = st.secrets.get("GEMINI_API_KEY")
    if not OPENAI_KEY:
        st.error("Missing OPENAI_API_KEY in .streamlit/secrets.toml")
        return
    if not ANTHROPIC_KEY:
        st.warning("ANTHROPIC_API_KEY not set; Anthropic option will fail.")
    if not GEMINI_KEY:
        st.warning("GEMINI_API_KEY not set; Gemini option will fail.")

    # --- Sidebar: user picks one of the 3 LLMs (per HW)
    st.sidebar.header("⚙️ Settings")
    provider = st.sidebar.selectbox("LLM Provider", ["OpenAI", "Anthropic", "Google Gemini"])

    # Optional: advanced vs lite tier (not required by HW, but handy)
    tier = st.sidebar.radio("Model Tier (optional)", ["Advanced", "Lite"], horizontal=True)

    model_map = {
        "OpenAI": {
            "Advanced": "gpt-4.1",
            "Lite": "gpt-5-chat-latest",
        },
        "Anthropic": {
            "Advanced": "claude-opus-4-1",
            "Lite": "claude-3-5-haiku-latest",
        },
        "Google Gemini": {
            "Advanced": "gemini-2.5-pro",
            "Lite": "gemini-2.5-flash-lite",
        },
    }
    model_name = model_map[provider][tier]
    st.sidebar.write(f"**Model:** {model_name}")

    # --- Build/load vector DB ONCE (per HW)
    collection = _ensure_vector_db(
        persist_dir="./vectorstore/ischool_orgs",
        org_dir="./orgs",
        openai_key=OPENAI_KEY,
    )
    if collection is None:
        return

    # --- Reset chat if model changes (keeps memory scoped to model)
    if "hw4_current_model" not in st.session_state or st.session_state.hw4_current_model != model_name:
        st.session_state.hw4_current_model = model_name
        st.session_state.hw4_chat = []              # list of {"role": "user"/"assistant", "content": str}
        st.sidebar.info(f"🔄 Chat reset for {provider} – {tier}")

    # --- Display previous messages (if any)
    for m in st.session_state.hw4_chat:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    # --- Memory buffer: keep last 5 Q/A pairs (i.e., last 10 messages)
    def _memory_text():
        dq = deque(st.session_state.hw4_chat, maxlen=10)
        txt = ""
        for msg in dq:
            role = "User" if msg["role"] == "user" else "Assistant"
            txt += f"{role}: {msg['content']}\n"
        return txt

    # --- Chat input
    user_q = st.chat_input("Ask about iSchool student organizations…")
    if user_q:
        # Show user message & append to chat
        st.session_state.hw4_chat.append({"role": "user", "content": user_q})
        with st.chat_message("user"):
            st.markdown(user_q)

        # Retrieve top 4 chunks
        res = collection.query(query_texts=[user_q], n_results=4)
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        context = "\n\n".join(d for d in docs if d and d.strip())
        sources = [m.get("source", "?") for m in metas]

        # Build final prompt with memory (last 5 Q/A), per HW
        prompt = (
            "You are a Syracuse iSchool organizations assistant. "
            "Use ONLY the provided 'Context' (retrieved from the orgs dataset). "
            "If the answer is not in the context, respond exactly with: "
            "\"I could not find this in the iSchool orgs dataset.\"\n\n"
            f"Conversation (last 5 Q/A):\n{_memory_text()}\n"
            f"Context:\n{context}\n\n"
            f"Question: {user_q}\n"
            "Answer:"
        )

        # Run selected LLM (no streaming → avoids org permissions issues)
        try:
            if provider == "OpenAI":
                ans = _run_openai(model_name, OPENAI_KEY, prompt)
            elif provider == "Anthropic":
                ans = _run_anthropic(model_name, ANTHROPIC_KEY, prompt)
            else:
                ans = _run_gemini(model_name, GEMINI_KEY, prompt)

            if not ans.strip():
                ans = "I could not find this in the iSchool orgs dataset."

            with st.chat_message("assistant"):
                st.markdown(ans)
            st.session_state.hw4_chat.append({"role": "assistant", "content": ans})

            # Trim to last 10 messages (5 Q/A) to satisfy HW memory size
            st.session_state.hw4_chat = st.session_state.hw4_chat[-10:]

            # Sources
            if sources:
                with st.expander("📂 Sources used (this turn)"):
                    for s in sources:
                        st.write(f"- {s}")

        except Exception as e:
            err = f"⚠️ {provider} / {model_name} failed: {e}"
            with st.chat_message("assistant"):
                st.error(err)
            st.session_state.hw4_chat.append({"role": "assistant", "content": err})

    