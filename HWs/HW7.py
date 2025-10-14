# HWs/HW7.py
import os
import csv
import pickle
from typing import List, Dict, Tuple
from collections import deque

import numpy as np
import streamlit as st

# ---------- FAISS for local vector store ----------
try:
    import faiss  # faiss-cpu
except Exception as e:
    faiss = None

# ---------- LLM SDKs (non-streaming; keep params minimal for compat) ----------
from openai import OpenAI as _OpenAI
from anthropic import Anthropic as _Anthropic
import google.generativeai as _genai


# =========================
# Config
# =========================
DATASET_PATH = "./news_data/news_dataset.csv"   # <- your local CSV
VSTORE_DIR   = "./vectorstore/news"             # FAISS + metadata
INDEX_FILE   = os.path.join(VSTORE_DIR, "faiss.index")
META_FILE    = os.path.join(VSTORE_DIR, "meta.pkl")

# Model map (Advanced vs Lite)
MODEL_MAP = {
    "OpenAI": {
        "Advanced": "gpt-4.1",
        "Lite":     "gpt-5-chat-latest",
        # Non-reasoning models as requested earlier
    },
    "Anthropic": {
        "Advanced": "claude-opus-4-1",
        "Lite":     "claude-3-5-haiku-latest",
    },
    "Google Gemini": {
        "Advanced": "gemini-2.5-pro",
        "Lite":     "gemini-2.5-flash-lite",
    },
}


# =========================
# Embeddings (OpenAI)
# =========================
def _embed_texts(texts: List[str], api_key: str, batch: int = 64) -> np.ndarray:
    """
    Use OpenAI text-embedding-3-small to get embeddings for a list of texts.
    Returns a numpy array of shape (N, D).
    """
    client = _OpenAI(api_key=api_key)

    vectors = []
    for i in range(0, len(texts), batch):
        chunk = texts[i:i+batch]
        resp = client.embeddings.create(
            model="text-embedding-3-small",
            input=chunk
        )
        # normalize to unit length for cosine similarity
        for d in resp.data:
            v = np.array(d.embedding, dtype=np.float32)
            v /= (np.linalg.norm(v) + 1e-12)
            vectors.append(v)
    return np.vstack(vectors)


# =========================
# CSV → docs
# =========================
def _read_csv_as_docs(path: str) -> List[Dict]:
    """
    Reads a CSV and turns each row into a doc dict:
    {
      "id": str,
      "text": "<joined cleaned fields>",
      "source": "<useful id>",
      "meta": { original columns }
    }
    """
    if not os.path.exists(path):
        return []

    docs = []
    with open(path, "r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        idx = 0
        for row in reader:
            # Heuristic: join non-empty textual fields
            parts = []
            for k, v in row.items():
                v = (v or "").strip()
                if v and not v.startswith("http"):
                    parts.append(f"{k}: {v}")
            text = " | ".join(parts).strip()

            if text:
                src = row.get("id") or row.get("title") or row.get("url") or f"row_{idx}"
                docs.append({
                    "id": str(src),
                    "text": text,
                    "source": str(src),
                    "meta": row
                })
                idx += 1
    return docs


# =========================
# FAISS store helpers
# =========================
def _faiss_available() -> bool:
    return faiss is not None

def _save_faiss(index: faiss.Index, metas: List[Dict]):
    os.makedirs(VSTORE_DIR, exist_ok=True)
    faiss.write_index(index, INDEX_FILE)
    with open(META_FILE, "wb") as f:
        pickle.dump(metas, f)

def _load_faiss() -> Tuple[faiss.Index, List[Dict]]:
    if not (os.path.exists(INDEX_FILE) and os.path.exists(META_FILE) and _faiss_available()):
        return None, None
    index = faiss.read_index(INDEX_FILE)
    with open(META_FILE, "rb") as f:
        metas = pickle.load(f)
    return index, metas


def _ensure_vector_db(openai_key: str) -> Tuple[faiss.Index, List[Dict]]:
    """
    Ensures FAISS index exists. If not, builds from DATASET_PATH.
    Returns (index, metas).
    """
    if not _faiss_available():
        st.error("FAISS is not available. Install faiss-cpu in requirements.")
        return None, None

    index, metas = _load_faiss()
    if index is not None:
        st.sidebar.info(f"🔁 FAISS index loaded with {index.ntotal} docs.")
        return index, metas

    # Build from CSV
    docs = _read_csv_as_docs(DATASET_PATH)
    if not docs:
        st.error("❌ No usable text found in the CSV rows.")
        return None, None

    texts = [d["text"] for d in docs]
    st.sidebar.write("🔧 Embedding documents…")
    embs = _embed_texts(texts, openai_key)  # shape (N, D)

    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)  # we used cosine -> normalized => IP works as cosine
    index.add(embs)

    _save_faiss(index, docs)
    st.sidebar.success(f"✅ Built FAISS index with {index.ntotal} docs.")
    return index, docs


def _retrieve(index: faiss.Index, metas: List[Dict], query: str, openai_key: str, k: int = 5) -> Tuple[str, List[str]]:
    """
    Retrieve top-k docs using cosine (inner product on normalized vectors).
    Returns (context_text, sources_list)
    """
    if not query.strip():
        return "", []

    qvec = _embed_texts([query], openai_key)[0].reshape(1, -1)  # (1, D)
    scores, idxs = index.search(qvec, k)
    idxs = idxs[0].tolist()

    chosen = []
    sources = []
    for i in idxs:
        if 0 <= i < len(metas):
            chosen.append(metas[i]["text"])
            sources.append(metas[i]["source"])
    context = "\n\n".join(chosen)
    return context, sources


# =========================
# LLM runners (simple, non-streaming)
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
# Prompt + memory
# =========================
def _memory_text(history: List[Dict], max_pairs: int = 5) -> str:
    dq = deque(history, maxlen=max_pairs * 2)
    return "\n".join(
        ("User: " + m["content"]) if m["role"] == "user" else ("Assistant: " + m["content"])
        for m in dq
    )

def _build_prompt(memory_txt: str, context: str, question: str) -> str:
    return (
        "You are a helpful news assistant. Use ONLY the provided 'Context' (retrieved from the vector store). "
        "If the answer is not in the context, respond exactly with: "
        "\"I could not find this in the news dataset.\"\n\n"
        f"Conversation (last 5 Q/A):\n{memory_txt}\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


# =========================
# Page
# =========================
def render():
    st.header("HW7 – News Q&A with Local CSV Vector DB (FAISS)")
    st.caption(
        "Uses local CSV → embeddings → FAISS (no uploads). "
        "Chat mode has short token-style memory. Evaluation compares Advanced vs Lite per provider."
    )

    # Keys
    OPENAI_KEY = st.secrets.get("OPENAI_API_KEY")
    ANTHROPIC_KEY = st.secrets.get("ANTHROPIC_API_KEY")
    GEMINI_KEY = st.secrets.get("GEMINI_API_KEY")
    if not OPENAI_KEY:
        st.error("Missing OPENAI_API_KEY in .streamlit/secrets.toml (required for embeddings + OpenAI chat).")
        return
    if not ANTHROPIC_KEY:
        st.warning("ANTHROPIC_API_KEY not set; Anthropic option will fail.")
    if not GEMINI_KEY:
        st.warning("GEMINI_API_KEY not set; Gemini option will fail.")

    # Sidebar
    st.sidebar.header("⚙️ Settings")
    provider = st.sidebar.selectbox("LLM Provider", ["OpenAI", "Anthropic", "Google Gemini"])
    mode = st.sidebar.radio("Mode", ["Chatbot", "Evaluation"], horizontal=True)

    adv_model = MODEL_MAP[provider]["Advanced"]
    lite_model = MODEL_MAP[provider]["Lite"]
    st.sidebar.write(f"**Advanced:** {adv_model}")
    st.sidebar.write(f"**Lite:** {lite_model}")

    # Rebuild vector store
    import shutil
    if st.sidebar.button("🗑️ Rebuild vector index"):
        try:
            shutil.rmtree(VSTORE_DIR)
            st.sidebar.success("Vector store removed. It will be rebuilt automatically on next run.")
        except Exception as e:
            st.sidebar.error(f"Could not remove vector store: {e}")

    # Ensure FAISS index
    index, metas = _ensure_vector_db(OPENAI_KEY)
    if index is None:
        return

    # --------------- CHAT ---------------
    if mode == "Chatbot":
        tier_choice = st.sidebar.radio("Chat model", ["Advanced", "Lite"], horizontal=True)
        chat_model = adv_model if tier_choice == "Advanced" else lite_model
        chat_key = f"{provider}:{tier_choice}"

        if "hw7_chat_key" not in st.session_state or st.session_state.hw7_chat_key != chat_key:
            st.session_state.hw7_chat_key = chat_key
            st.session_state.hw7_chat = []
            st.sidebar.info(f"🔄 Chat reset for {chat_key}")

        # show history
        for m in st.session_state.hw7_chat:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

        user_q = st.chat_input("Ask about the news dataset…")
        if not user_q:
            return

        st.session_state.hw7_chat.append({"role": "user", "content": user_q})
        with st.chat_message("user"):
            st.markdown(user_q)

        context, sources = _retrieve(index, metas, user_q, OPENAI_KEY, k=5)
        mem_txt = _memory_text(st.session_state.hw7_chat, max_pairs=5)
        prompt = _build_prompt(mem_txt, context, user_q)

        try:
            if provider == "OpenAI":
                ans = _run_openai(chat_model, OPENAI_KEY, prompt)
            elif provider == "Anthropic":
                ans = _run_anthropic(chat_model, ANTHROPIC_KEY, prompt)
            else:
                ans = _run_gemini(chat_model, GEMINI_KEY, prompt)
            if not ans:
                ans = "I could not find this in the news dataset."
        except Exception as e:
            ans = f"⚠️ {provider} / {chat_model} failed: {e}"

        with st.chat_message("assistant"):
            st.markdown(ans)
        st.session_state.hw7_chat.append({"role": "assistant", "content": ans})
        st.session_state.hw7_chat = st.session_state.hw7_chat[-10:]

        if sources:
            with st.expander("📂 Sources used (this turn)"):
                for s in sources:
                    st.write(f"- {s}")

    # --------------- EVALUATION ---------------
    else:
        st.subheader("🔎 Evaluation: Advanced vs Lite (selected provider)")
        st.write("Runs 5 questions against both models using the same retrieved context for fairness.")

        QUESTIONS = [
            "What does any article say about the event's time and location?",
            "Summarize the main claim or takeaway mentioned in one article.",
            "Who is quoted or referenced in any of the items, if anyone?",
            "Is there a mention of impact or consequences? Summarize briefly.",
            "Extract one concrete fact (date, number, venue) cited in the dataset.",
        ]

        run_eval = st.button("Run Evaluation")
        if not run_eval:
            with st.expander("View the 5 evaluation questions"):
                for i, q in enumerate(QUESTIONS, 1):
                    st.write(f"{i}. {q}")
            return

        for i, q in enumerate(QUESTIONS, 1):
            st.markdown(f"### Q{i}. {q}")
            context, sources = _retrieve(index, metas, q, OPENAI_KEY, k=5)
            prompt = _build_prompt("", context, q)

            # Advanced
            try:
                if provider == "OpenAI":
                    adv_ans = _run_openai(adv_model, OPENAI_KEY, prompt)
                elif provider == "Anthropic":
                    adv_ans = _run_anthropic(adv_model, ANTHROPIC_KEY, prompt)
                else:
                    adv_ans = _run_gemini(adv_model, GEMINI_KEY, prompt)
                if not adv_ans:
                    adv_ans = "I could not find this in the news dataset."
            except Exception as e:
                adv_ans = f"⚠️ {provider} / {adv_model} failed: {e}"

            # Lite
            try:
                if provider == "OpenAI":
                    lite_ans = _run_openai(lite_model, OPENAI_KEY, prompt)
                elif provider == "Anthropic":
                    lite_ans = _run_anthropic(lite_model, ANTHROPIC_KEY, prompt)
                else:
                    lite_ans = _run_gemini(lite_model, GEMINI_KEY, prompt)
                if not lite_ans:
                    lite_ans = "I could not find this in the news dataset."
            except Exception as e:
                lite_ans = f"⚠️ {provider} / {lite_model} failed: {e}"

            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**Advanced – {adv_model}**")
                st.write(adv_ans)
            with c2:
                st.markdown(f"**Lite – {lite_model}**")
                st.write(lite_ans)

            with st.expander(f"📂 Sources for Q{i}"):
                for s in sources:
                    st.write(f"- {s}")

            st.divider()
