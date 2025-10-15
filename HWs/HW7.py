# HW7 – News RAG Bot (ChromaDB + Local CSV)
# - Reads only local news_data/news_dataset.csv
# - Builds a persistent Chroma vectorstore
# - "Most interesting news" ranking + "topic search"
# - Chatbot answers only from embedded rows
# - Absolute paths + robust vectorstore creation
# - Works with OpenAI / Anthropic / Gemini (no streaming, minimal params)

from __future__ import annotations

import os
import shutil
import time
from datetime import datetime, timezone
from collections import deque
from typing import List, Dict, Tuple

import streamlit as st
import pandas as pd

# --- Optional fix for sqlite3 issues with Chroma on some hosts ---
try:
    __import__("pysqlite3")  # noqa: F401
    import sys as _sys
    _sys.modules["sqlite3"] = _sys.modules.pop("pysqlite3")
except Exception:
    pass

import chromadb
from chromadb.utils import embedding_functions

# LLM SDKs (non-reasoning, no streaming)
from openai import OpenAI as _OpenAI
from anthropic import Anthropic as _Anthropic
import google.generativeai as _genai


# =========================================================
# Paths & Environment
# =========================================================

def _paths() -> Tuple[str, str, str]:
    """
    Resolve absolute paths, create folders, and return:
      (project_root, csv_path, persist_dir)
    """
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    csv_path = os.path.join(root, "news_data", "news_dataset.csv")
    persist_dir = os.path.join(root, "vectorstore", "news")

    os.makedirs(os.path.join(root, "vectorstore"), exist_ok=True)
    os.makedirs(persist_dir, exist_ok=True)

    return root, csv_path, persist_dir


# =========================================================
# CSV Loading & Preprocessing
# =========================================================

TEXT_COL_CANDIDATES = ["headline", "title", "summary", "description", "content", "body", "text"]
DATE_COL_CANDIDATES = ["date", "published", "published_at", "pub_date"]
TOPIC_COL_CANDIDATES = ["topic", "category", "section", "tags", "keywords"]
SOURCE_COL_CANDIDATES = ["source", "publisher", "outlet", "domain"]


def _pick_col(cols: List[str], candidates: List[str]) -> str | None:
    cols_lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand in cols_lower:
            return cols_lower[cand]
    return None


def _load_news_csv(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        st.error(f"❌ news_dataset.csv not found at:\n{csv_path}")
        return pd.DataFrame()

    df = pd.read_csv(csv_path)

    # Try to locate useful columns (best-effort)
    text_cols = [c for c in df.columns if str(df[c].dtype) == "object"]
    date_col = _pick_col(df.columns.tolist(), DATE_COL_CANDIDATES)
    topic_col = _pick_col(df.columns.tolist(), TOPIC_COL_CANDIDATES)
    title_col = _pick_col(df.columns.tolist(), ["headline", "title"]) or (text_cols[0] if text_cols else None)
    source_col = _pick_col(df.columns.tolist(), SOURCE_COL_CANDIDATES)

    df._meta = {
        "text_cols": text_cols,
        "date_col": date_col,
        "topic_col": topic_col,
        "title_col": title_col,
        "source_col": source_col,
    }
    return df


def _row_as_text(row: pd.Series, meta: Dict[str, str]) -> str:
    """
    Build an embedding text string from a row.
    We concatenate known textual columns; include date/topic/source if present.
    """
    parts = []

    # Preferred fields first
    title = str(row.get(meta.get("title_col", ""), "") or "").strip()
    if title:
        parts.append(f"Title: {title}")

    topic_col = meta.get("topic_col")
    if topic_col:
        t = str(row.get(topic_col, "") or "").strip()
        if t:
            parts.append(f"Topic: {t}")

    # Add more text columns
    for c in meta.get("text_cols", []):
        val = str(row.get(c, "") or "").strip()
        if val and (c.lower() not in ["title", "headline"]):  # avoid repeating
            parts.append(f"{c}: {val}")

    # Add date/source if available
    date_col = meta.get("date_col")
    if date_col:
        d = str(row.get(date_col, "") or "").strip()
        if d:
            parts.append(f"Date: {d}")

    source_col = meta.get("source_col")
    if source_col:
        s = str(row.get(source_col, "") or "").strip()
        if s:
            parts.append(f"Source: {s}")

    txt = " | ".join(parts)
    return txt


# =========================================================
# Interestingness Scoring (heuristic for a global law firm)
# =========================================================

LAW_KEYWORDS = {
    "regulatory": 1.0,
    "regulation": 1.0,
    "lawsuit": 1.2,
    "litigation": 1.2,
    "acquisition": 1.0,
    "merger": 1.0,
    "antitrust": 1.2,
    "compliance": 0.9,
    "settlement": 1.1,
    "investigation": 1.0,
    "ipo": 1.0,
    "sanction": 1.1,
    "privacy": 0.9,
    "cybersecurity": 0.9,
    "governance": 0.8,
    "bankruptcy": 1.1,
}

BIG_NAMES = [
    "microsoft", "google", "alphabet", "amazon", "apple", "meta",
    "openai", "anthropic", "morgan stanley", "jpmorgan", "jp morgan",
    "goldman sachs", "bank of america", "citi", "citigroup",
    "tesla", "nvidia", "intel", "amd", "oracle", "salesforce",
]


def _parse_date_safe(date_str: str) -> datetime | None:
    if not date_str or not isinstance(date_str, str):
        return None
    # Try pandas robust parser
    try:
        dt = pd.to_datetime(date_str, errors="coerce", utc=True)
        if pd.isna(dt):
            return None
        # return naive UTC datetime (to compare with datetime.utcnow())
        return dt.to_pydatetime().astimezone(timezone.utc).replace(tzinfo=None)
    except Exception:
        return None


def recency_score(date_str: str, now: datetime | None = None) -> float:
    """Newer = higher score. Safe with/without tz."""
    now = now or datetime.utcnow()
    dt = _parse_date_safe(date_str)
    if not dt:
        return 0.3  # unknown date gets a small baseline
    # days difference
    days = max(1, (now - dt).days)
    # Exponential decay: 1.0 -> recent; ~0.1 after ~45 days
    return max(0.05, min(1.0, 1.2 ** (-days / 14)))


def legal_relevance(text: str) -> float:
    t = (text or "").lower()
    score = 0.0
    for k, w in LAW_KEYWORDS.items():
        if k in t:
            score += w
    for name in BIG_NAMES:
        if name in t:
            score += 0.3
    # normalize (roughly)
    return min(2.0, 0.3 + score)


def interestingness(row: pd.Series, meta: Dict[str, str]) -> float:
    """
    Overall interestingness for a global law firm audience:
      50% recency + 40% legal/business relevance + 10% length
    """
    # collect text for relevance
    text = _row_as_text(row, meta)

    # recency
    dcol = meta.get("date_col")
    dval = str(row.get(dcol, "") or "") if dcol else ""
    s_rec = recency_score(dval)

    # legal/business relevance
    s_rel = legal_relevance(text)

    # crude length score (longer summaries get a small boost)
    length = max(30, len(text))
    s_len = min(1.0, length / 800.0)

    final = 0.5 * s_rec + 0.4 * (s_rel / 2.0) + 0.1 * s_len
    return float(final)


# =========================================================
# Chroma – Build/Load persistent vector DB
# =========================================================

def _fresh_persist_dir(base_dir: str) -> str:
    """Create a fresh subfolder if the DB becomes read-only/corrupt."""
    new_dir = os.path.join(os.path.dirname(base_dir), f"news_v{int(time.time())}")
    os.makedirs(new_dir, exist_ok=True)
    return new_dir


def _ensure_vector_db_from_csv(openai_key: str) -> Tuple[chromadb.Collection | None, str, pd.DataFrame]:
    root, csv_path, persist_dir = _paths()

    st.sidebar.caption("**HW7 Paths**")
    st.sidebar.code(f"CSV:        {csv_path}")
    st.sidebar.code(f"Vectorstore:{persist_dir}")

    df = _load_news_csv(csv_path)
    if df.empty:
        return None, "", df

    meta = df._meta
    docs, metas, ids = [], [], []

    for i, row in df.iterrows():
        txt = _row_as_text(row, meta)
        if txt:
            docs.append(txt)
            metas.append({
                "row": int(i),
                "date": str(row.get(meta.get("date_col", ""), "") or ""),
                "title": str(row.get(meta.get("title_col", ""), "") or ""),
                "topic": str(row.get(meta.get("topic_col", ""), "") or ""),
                "source": str(row.get(meta.get("source_col", ""), "") or ""),
            })
            ids.append(f"row_{i}")

    if not docs:
        st.error("❌ No usable text found in the CSV rows.")
        return None, "", df

    def build_client(path_dir: str) -> chromadb.PersistentClient:
        return chromadb.PersistentClient(path=path_dir)

    # Try to create/load collection. If read-only error -> rotate folder once.
    for attempt in range(2):
        path_try = persist_dir if attempt == 0 else _fresh_persist_dir(persist_dir)

        try:
            client = build_client(path_try)
            collection = client.get_or_create_collection(
                name="news_collection",
                embedding_function=embedding_functions.OpenAIEmbeddingFunction(
                    api_key=openai_key,
                    model_name="text-embedding-3-small",
                ),
            )

            try:
                existing = collection.count()
            except Exception:
                existing = len(collection.get().get("ids", []))

            if existing and existing > 0:
                st.sidebar.success(f"🔁 Vector DB ready at {path_try} with {existing} items.")
                return collection, path_try, df

            # Add fresh
            collection.add(documents=docs, metadatas=metas, ids=ids)
            st.sidebar.success(f"✅ Embedded {len(docs)} rows into {path_try}.")
            return collection, path_try, df

        except Exception as e:
            if attempt == 0:
                st.sidebar.warning(f"Vector DB path '{persist_dir}' failed ({e}). Rotating once…")
                continue
            st.error(f"❌ Could not create vector DB: {e}")
            return None, "", df

    return None, "", df


def _retrieve(collection: chromadb.Collection, query: str, k: int = 6) -> Tuple[str, List[Dict]]:
    res = collection.query(query_texts=[query], n_results=k)
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    context = "\n\n".join([d for d in docs if d and d.strip()])
    return context, metas


# =========================================================
# LLM Runners (no streaming; minimal params)
# =========================================================

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
        max_tokens=900,
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


# =========================================================
# Prompts & Memory
# =========================================================

def _memory_text(history: List[Dict], max_pairs: int = 5) -> str:
    dq = deque(history, maxlen=max_pairs * 2)
    lines = []
    for msg in dq:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)


def _build_chat_prompt(memory_txt: str, context: str, question: str) -> str:
    return (
        "You are a news assistant for a global law firm. Use ONLY the provided 'Context' "
        "(retrieved from the internal news dataset). If the answer is not in the context, "
        "respond exactly with: \"I could not find this in the news dataset.\"\n\n"
        f"Conversation (last 5 Q/A):\n{memory_txt}\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer in 3-6 concise sentences with a lawyerly tone."
    )


def _build_list_prompt(context: str, task: str) -> str:
    """
    Format a prompt that asks the model to produce a structured list from retrieved items.
    """
    return (
        "You are a news analyst for a global law firm. Use ONLY the 'Context' items below. "
        "Return a structured list with bullets. For each item, extract Title, Date, Source, and a 1-2 sentence rationale. "
        "If insufficient context, reply: \"I could not find this in the news dataset.\"\n\n"
        f"Task: {task}\n\n"
        f"Context:\n{context}\n\n"
        "Output format:\n"
        "- Title — Date — Source: Rationale…\n"
        "- …"
    )


# =========================================================
# App (Streamlit)
# =========================================================

def render():
    st.header("HW7 – News Reporting Bot (RAG on Local CSV + Chroma)")
    st.caption(
        "Builds a persistent vector index from local news_data/news_dataset.csv only. "
        "Answers: (1) most interesting news, (2) topic-specific news, (3) general Q&A chatbot."
    )

    # API keys from secrets (only required for embeddings + optional generation)
    KEYS = {
        "OPENAI_API_KEY": st.secrets.get("OPENAI_API_KEY"),
        "ANTHROPIC_API_KEY": st.secrets.get("ANTHROPIC_API_KEY"),
        "GEMINI_API_KEY": st.secrets.get("GEMINI_API_KEY"),
    }
    if not KEYS["OPENAI_API_KEY"]:
        st.error("Missing OPENAI_API_KEY in .streamlit/secrets.toml (needed for embeddings).")
        return

    # Provider selection (for generation; embeddings always use OpenAI text-embedding-3-small)
    st.sidebar.header("⚙️ Provider & Models")
    provider = st.sidebar.selectbox("Generator Vendor", ["OpenAI", "Anthropic", "Google Gemini"])
    tier = st.sidebar.radio("Model Tier", ["Advanced", "Lite"], horizontal=True)

    model_map = {
        "OpenAI": {"Advanced": "gpt-4.1", "Lite": "gpt-5-chat-latest"},
        "Anthropic": {"Advanced": "claude-opus-4-1", "Lite": "claude-3-5-haiku-latest"},
        "Google Gemini": {"Advanced": "gemini-2.5-pro", "Lite": "gemini-2.5-flash-lite"},
    }
    model_name = model_map[provider][tier]
    st.sidebar.write(f"**Using model:** {model_name}")

    # Build/load vector DB from local CSV
    collection, persist_dir, df = _ensure_vector_db_from_csv(KEYS["OPENAI_API_KEY"])
    if collection is None or df.empty:
        return

    # Tabs: Most Interesting, Topic Search, Chatbot, Evaluation
    tab1, tab2, tab3, tab4 = st.tabs(["🔥 Most Interesting", "🔎 Topic Search", "💬 Chatbot", "📊 Evaluation"])

    # -------------------- Most Interesting --------------------
    with tab1:
        st.subheader("Find the Most Interesting News (Law-Firm Heuristic Ranking)")
        st.write("Ranks all rows using recency + legal/business relevance heuristics.")

        # Compute scores
        meta = df._meta
        scores = []
        for i, row in df.iterrows():
            s = interestingness(row, meta)
            scores.append((float(s), int(i)))
        scores.sort(reverse=True)

        top_k = st.slider("How many items?", 5, 30, 10, step=1)
        top_rows = scores[:top_k]

        # Build a "context" to pass to the model for a neat bullet list
        ctx_parts = []
        for s, idx in top_rows:
            r = df.iloc[idx]
            title = str(r.get(meta.get("title_col", ""), "") or "").strip()
            date = str(r.get(meta.get("date_col", ""), "") or "").strip()
            src = str(r.get(meta.get("source_col", ""), "") or "").strip()
            topic = str(r.get(meta.get("topic_col", ""), "") or "").strip()
            txt = _row_as_text(r, meta)
            ctx_parts.append(f"[Score={s:.2f}] Title: {title} | Date: {date} | Source: {src} | Topic: {topic} | {txt}")

        context = "\n".join(ctx_parts)
        prompt = _build_list_prompt(context, "Produce a ranked list from the supplied items, highest score first.")

        if st.button("Generate Ranked Summary"):
            if provider == "OpenAI":
                out = _run_openai(model_name, KEYS["OPENAI_API_KEY"], prompt)
            elif provider == "Anthropic":
                out = _run_anthropic(model_name, KEYS["ANTHROPIC_API_KEY"], prompt)
            else:
                out = _run_gemini(model_name, KEYS["GEMINI_API_KEY"], prompt)
            st.markdown(out if out else "I could not find this in the news dataset.")

        with st.expander("Show Top Items (raw)"):
            for s, idx in top_rows:
                r = df.iloc[idx]
                title = str(r.get(meta.get("title_col", ""), "") or "").strip()
                date = str(r.get(meta.get("date_col", ""), "") or "").strip()
                src = str(r.get(meta.get("source_col", ""), "") or "").strip()
                st.write(f"- **{title}** — {date} — {src} (score {s:.2f})")

    # -------------------- Topic Search --------------------
    with tab2:
        st.subheader("Find News About a Topic")
        topic_q = st.text_input("Topic (e.g., 'antitrust', 'JP Morgan branch', 'IPO')", "")

        if topic_q:
            context, metas = _retrieve(collection, topic_q, k=12)

            # Ask LLM to pick and present best matches from the retrieved context
            task = f"From the context items, find news about: '{topic_q}'. Return a curated list."
            prompt = _build_list_prompt(context, task)

            if provider == "OpenAI":
                out = _run_openai(model_name, KEYS["OPENAI_API_KEY"], prompt)
            elif provider == "Anthropic":
                out = _run_anthropic(model_name, KEYS["ANTHROPIC_API_KEY"], prompt)
            else:
                out = _run_gemini(model_name, KEYS["GEMINI_API_KEY"], prompt)

            st.markdown(out if out else "I could not find this in the news dataset.")

            with st.expander("📂 Sources used (retrieved)"):
                for m in metas:
                    title = m.get("title", "")
                    date = m.get("date", "")
                    src = m.get("source", "")
                    st.write(f"- {title} — {date} — {src}")

    # -------------------- Chatbot --------------------
    with tab3:
        st.subheader("Chatbot (Answers Only from Embedded CSV)")
        # Keep history separate per model
        chat_key = f"{provider}:{tier}"
        if "hw7_chat_key" not in st.session_state or st.session_state.hw7_chat_key != chat_key:
            st.session_state.hw7_chat_key = chat_key
            st.session_state.hw7_chat = []

        # show history
        for m in st.session_state.hw7_chat:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

        user_q = st.chat_input("Ask a question about the news…")
        if user_q:
            st.session_state.hw7_chat.append({"role": "user", "content": user_q})
            with st.chat_message("user"):
                st.markdown(user_q)

            # Retrieve & answer
            context, metas = _retrieve(collection, user_q, k=10)
            mem = _memory_text(st.session_state.hw7_chat, max_pairs=5)
            prompt = _build_chat_prompt(mem, context, user_q)

            if provider == "OpenAI":
                ans = _run_openai(model_name, KEYS["OPENAI_API_KEY"], prompt)
            elif provider == "Anthropic":
                ans = _run_anthropic(model_name, KEYS["ANTHROPIC_API_KEY"], prompt)
            else:
                ans = _run_gemini(model_name, KEYS["GEMINI_API_KEY"], prompt)

            if not ans:
                ans = "I could not find this in the news dataset."

            with st.chat_message("assistant"):
                st.markdown(ans)
            st.session_state.hw7_chat.append({"role": "assistant", "content": ans})

            # trim to last 10 messages (5 Q&A)
            st.session_state.hw7_chat = st.session_state.hw7_chat[-10:]

            with st.expander("📂 Sources used (retrieved)"):
                for m in metas:
                    title = m.get("title", "")
                    date = m.get("date", "")
                    src = m.get("source", "")
                    st.write(f"- {title} — {date} — {src}")

    # -------------------- Evaluation (2 vendors, Adv vs Lite) --------------------
    with tab4:
        st.subheader("Evaluation (Compare Models)")
        st.write("Pick a vendor below; we’ll run the same questions on Advanced vs Lite for that vendor.")
        eval_vendor = st.selectbox("Vendor to compare", ["OpenAI", "Anthropic", "Google Gemini"], index=0)

        eval_models = model_map[eval_vendor]
        qset = [
            "Find the most interesting news overall.",
            "Find news about antitrust actions in tech.",
            "Summarize IPO-related stories.",
            "Find news involving major banks (e.g., JPMorgan).",
            "What are the latest regulatory updates mentioned?",
        ]

        if st.button("Run Evaluation"):
            for i, q in enumerate(qset, 1):
                st.markdown(f"### Q{i}. {q}")
                # Retrieve once
                context, metas = _retrieve(collection, q, k=12)
                prompt = _build_list_prompt(context, f"Answer: {q}")

                # Advanced
                try:
                    if eval_vendor == "OpenAI":
                        adv_out = _run_openai(eval_models["Advanced"], KEYS["OPENAI_API_KEY"], prompt)
                    elif eval_vendor == "Anthropic":
                        adv_out = _run_anthropic(eval_models["Advanced"], KEYS["ANTHROPIC_API_KEY"], prompt)
                    else:
                        adv_out = _run_gemini(eval_models["Advanced"], KEYS["GEMINI_API_KEY"], prompt)
                except Exception as e:
                    adv_out = f"⚠️ Advanced failed: {e}"

                # Lite
                try:
                    if eval_vendor == "OpenAI":
                        lite_out = _run_openai(eval_models["Lite"], KEYS["OPENAI_API_KEY"], prompt)
                    elif eval_vendor == "Anthropic":
                        lite_out = _run_anthropic(eval_models["Lite"], KEYS["ANTHROPIC_API_KEY"], prompt)
                    else:
                        lite_out = _run_gemini(eval_models["Lite"], KEYS["GEMINI_API_KEY"], prompt)
                except Exception as e:
                    lite_out = f"⚠️ Lite failed: {e}"

                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"**Advanced – {eval_models['Advanced']}**")
                    st.write(adv_out or "I could not find this in the news dataset.")
                with c2:
                    st.markdown(f"**Lite – {eval_models['Lite']}**")
                    st.write(lite_out or "I could not find this in the news dataset.")

                with st.expander("📂 Sources (retrieved)"):
                    for m in metas:
                        st.write(f"- {m.get('title','')} — {m.get('date','')} — {m.get('source','')}")
                st.divider()
