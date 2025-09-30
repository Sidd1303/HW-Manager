import os
from collections import deque
import streamlit as st
from bs4 import BeautifulSoup

import chromadb
from chromadb.utils import embedding_functions

# LLM SDKs (non-streaming; no temp/max_tokens tweaks)
from openai import OpenAI as _OpenAI
from anthropic import Anthropic as _Anthropic
import google.generativeai as _genai


# =========================
# Utils: load & preprocess
# =========================

def _load_org_html(folder: str) -> dict[str, str]:
    """Load and clean ALL .html files from ./orgs into {filename: cleaned_text}."""
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
    """Return exactly TWO mini-docs (balanced halves) per document."""
    words = text.split()
    if len(words) <= 1:
        return [text] if text else []
    mid = len(words) // 2
    return [" ".join(words[:mid]), " ".join(words[mid:])]


def _ensure_vector_db(persist_dir: str, org_dir: str, openai_key: str):
    """
    Create or load a persisted ChromaDB collection.
    - Build from ./orgs if empty
    - Use exactly TWO chunks per document (see _chunk_into_two)
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

    # If collection already has vectors, skip re-embedding
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


def _retrieve(collection, query: str, k: int = 4):
    """Vector search -> (context_text, sources_list)."""
    res = collection.query(query_texts=[query], n_results=k)
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    context = "\n\n".join(d for d in docs if d and d.strip())
    sources = [m.get("source", "?") for m in metas]
    return context, sources


# =========================
# LLM runners (no streaming)
# =========================

def _run_openai(model: str, api_key: str, prompt: str) -> str:
    client = _OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return (resp.choices[0].message.content or "").strip()


def _run_anthropic(model: str, api_key: str, prompt: str) -> str:
    client = _Anthropic(api_key=api_key)
    resp = client.messages.create(
        model=model,
        max_tokens=800,
        messages=[{"role": "user", "content": prompt}],
    )
    try:
        return (resp.content[0].text or "").strip()
    except Exception:
        return ""


def _run_gemini(model: str, api_key: str, prompt: str) -> str:
    _genai.configure(api_key=api_key)
    g = _genai.GenerativeModel(model)
    r = g.generate_content(prompt)
    return (getattr(r, "text", "") or "").strip()


# =========================
# Prompt builder
# =========================

def _memory_text(history: list[dict], max_pairs: int = 5) -> str:
    """Serialize last N Q/A pairs (2N messages)."""
    dq = deque(history, maxlen=max_pairs * 2)
    lines = []
    for msg in dq:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)


def _build_prompt(memory_txt: str, context: str, question: str) -> str:
    return (
        "You are a Syracuse iSchool organizations assistant. "
        "Use ONLY the provided 'Context' (retrieved from the orgs dataset). "
        "If the answer is not in the context, respond exactly with: "
        "\"I could not find this in the iSchool orgs dataset.\"\n\n"
        f"Conversation (last 5 Q/A):\n{memory_txt}\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


# =========================
# Page
# =========================

def render():
    st.header("HW5 – Orgs RAG Chatbot (Chat + Evaluation)")
    st.caption(
        "Extends HW4: same RAG pipeline with 2-chunk docs & short-term memory; "
        "adds an Evaluation mode that compares Advanced vs Lite models per provider."
    )

    # --- API keys (secrets.toml)
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

    # --- Sidebar: provider + tier + mode
    st.sidebar.header("⚙️ Settings")
    provider = st.sidebar.selectbox("LLM Provider", ["OpenAI", "Anthropic", "Google Gemini"])
    mode = st.sidebar.radio("Mode", ["Chatbot", "Evaluation"], horizontal=True)

    # Advanced vs Lite (same mapping as HW4)
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

    adv_model = model_map[provider]["Advanced"]
    lite_model = model_map[provider]["Lite"]

    st.sidebar.write(f"**Advanced:** {adv_model}")
    st.sidebar.write(f"**Lite:** {lite_model}")

    # --- Build/load vector DB once
    collection = _ensure_vector_db(
        persist_dir="./vectorstore/ischool_orgs",
        org_dir="./orgs",
        openai_key=OPENAI_KEY,
    )
    if collection is None:
        return

    # =============== CHATBOT MODE ===============
    if mode == "Chatbot":
        # picker: which model to use during chat (either Advanced or Lite)
        tier_choice = st.sidebar.radio("Chat model", ["Advanced", "Lite"], horizontal=True)
        chat_model = adv_model if tier_choice == "Advanced" else lite_model

        # Reset chat if model OR provider changed
        chat_key = f"{provider}:{tier_choice}"
        if "hw5_chat_key" not in st.session_state or st.session_state.hw5_chat_key != chat_key:
            st.session_state.hw5_chat_key = chat_key
            st.session_state.hw5_chat = []
            st.sidebar.info(f"🔄 Chat reset for {chat_key}")

        # Show history
        for m in st.session_state.hw5_chat:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

        # Input
        user_q = st.chat_input("Ask about iSchool student organizations…")
        if not user_q:
            return

        # Append user
        st.session_state.hw5_chat.append({"role": "user", "content": user_q})
        with st.chat_message("user"):
            st.markdown(user_q)

        # Retrieve
        context, sources = _retrieve(collection, user_q, k=4)
        mem_txt = _memory_text(st.session_state.hw5_chat, max_pairs=5)
        prompt = _build_prompt(mem_txt, context, user_q)

        # Call LLM
        try:
            if provider == "OpenAI":
                ans = _run_openai(chat_model, OPENAI_KEY, prompt)
            elif provider == "Anthropic":
                ans = _run_anthropic(chat_model, ANTHROPIC_KEY, prompt)
            else:
                ans = _run_gemini(chat_model, GEMINI_KEY, prompt)

            if not ans:
                ans = "I could not find this in the iSchool orgs dataset."
        except Exception as e:
            ans = f"⚠️ {provider} / {chat_model} failed: {e}"

        with st.chat_message("assistant"):
            st.markdown(ans)
        st.session_state.hw5_chat.append({"role": "assistant", "content": ans})

        # Keep last 5 Q/A pairs (10 messages)
        st.session_state.hw5_chat = st.session_state.hw5_chat[-10:]

        # Sources
        if sources:
            with st.expander("📂 Sources used (this turn)"):
                for s in sources:
                    st.write(f"- {s}")

    # =============== EVALUATION MODE ===============
    else:
        st.subheader("🔎 Evaluation: Advanced vs Lite (selected provider)")
        st.write(
            "Runs a fixed set of questions against **both** Advanced and Lite models "
            "for the chosen provider, using the same RAG context."
        )

        # 5 test questions (edit if you like)
        QUESTIONS = [
            "What is the mission or purpose of any one iSchool organization?",
            "How can a student join the Data Science Club? Are there requirements?",
            "Summarize one organization’s typical activities or events.",
            "Who can I contact for leadership or officer info for any org?",
            "Are there dues or membership fees mentioned for any org?",
        ]

        run_eval = st.button("Run Evaluation")
        if not run_eval:
            st.write("Click **Run Evaluation** to compare outputs.")
            with st.expander("View the 5 evaluation questions"):
                for i, q in enumerate(QUESTIONS, 1):
                    st.write(f"{i}. {q}")
            return

        # Execute for each question: retrieve once, ask both models
        for idx, q in enumerate(QUESTIONS, 1):
            st.markdown(f"### Q{idx}. {q}")

            context, sources = _retrieve(collection, q, k=4)
            mem_txt = ""  # no chat memory in batch eval to keep runs comparable
            prompt = _build_prompt(mem_txt, context, q)

            # Run Advanced
            try:
                if provider == "OpenAI":
                    adv_ans = _run_openai(adv_model, OPENAI_KEY, prompt)
                elif provider == "Anthropic":
                    adv_ans = _run_anthropic(adv_model, ANTHROPIC_KEY, prompt)
                else:
                    adv_ans = _run_gemini(adv_model, GEMINI_KEY, prompt)
                if not adv_ans:
                    adv_ans = "I could not find this in the iSchool orgs dataset."
            except Exception as e:
                adv_ans = f"⚠️ {provider} / {adv_model} failed: {e}"

            # Run Lite
            try:
                if provider == "OpenAI":
                    lite_ans = _run_openai(lite_model, OPENAI_KEY, prompt)
                elif provider == "Anthropic":
                    lite_ans = _run_anthropic(lite_model, ANTHROPIC_KEY, prompt)
                else:
                    lite_ans = _run_gemini(lite_model, GEMINI_KEY, prompt)
                if not lite_ans:
                    lite_ans = "I could not find this in the iSchool orgs dataset."
            except Exception as e:
                lite_ans = f"⚠️ {provider} / {lite_model} failed: {e}"

            # Show side-by-side
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**Advanced – {adv_model}**")
                st.write(adv_ans)
            with c2:
                st.markdown(f"**Lite – {lite_model}**")
                st.write(lite_ans)

            # Sources
            with st.expander(f"📂 Sources for Q{idx}"):
                for s in sources:
                    st.write(f"- {s}")

            st.divider()
