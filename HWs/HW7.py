# --- sqlite shim so Chroma works reliably in Codespaces/Streamlit Cloud
__import__("pysqlite3")
import sys as _sys
_sys.modules["sqlite3"] = _sys.modules.pop("pysqlite3")

import os
import io
import hashlib
from typing import List, Dict, Any, Tuple
from collections import deque
from datetime import datetime, timezone

import streamlit as st
import pandas as pd
from dateutil import parser as dateparser

import chromadb
from chromadb.utils import embedding_functions

# LLM SDKs (non-streaming; keep params minimal for model compatibility)
from openai import OpenAI as _OpenAI
from anthropic import Anthropic as _Anthropic
import google.generativeai as _genai

default_path = "./news_data/news_dataset.csv"
# =========================
# Utility: safe date parsing
# =========================

def _parse_date(x) -> datetime | None:
    if pd.isna(x):
        return None
    try:
        d = dateparser.parse(str(x))
        if not d.tzinfo:
            d = d.replace(tzinfo=timezone.utc)
        return d
    except Exception:
        return None


# =========================
# Heuristic Interestingness
# =========================

_KEYWORDS = [
    "acquire", "acquisition", "merger", "merge", "lawsuit", "litigation",
    "antitrust", "regulation", "settlement", "sanction", "fine",
    "ipo", "investigation", "indictment", "class action", "compliance"
]
_SOURCE_WEIGHTS = {
    "reuters": 1.15,
    "financial times": 1.12,
    "bloomberg": 1.12,
    "wsj": 1.12,
    "ap": 1.08,
    "bbc": 1.08,
}

def _keyword_score(text: str) -> float:
    t = text.lower()
    hits = sum(1 for k in _KEYWORDS if k in t)
    return min(hits, 5) / 5.0  # 0..1

def _recency_score(published: datetime | None, now: datetime) -> float:
    if not published:
        return 0.4  # modest baseline if unknown
    days = (now - published).days
    if days < 0:
        return 1.0
    if days <= 1:
        return 1.0
    if days <= 7:
        return 0.6
    if days <= 30:
        return 0.2
    return 0.1

def _source_score(source: str | None) -> float:
    if not source:
        return 1.0
    s = str(source).lower().strip()
    for k, w in _SOURCE_WEIGHTS.items():
        if k in s:
            return w
    return 1.0

def compute_interest_scores(df: pd.DataFrame) -> pd.Series:
    """
    Score ∈ [~0..1.3] combining:
      50% recency, 35% legal-impact keywords, 15% source authority
    """
    now = datetime.now(timezone.utc)

    def _row_score(row) -> float:
        text = f"{row.get('title','')} {row.get('summary','')}"
        rs = _recency_score(row.get("_parsed_date"), now) * 0.50
        ks = _keyword_score(text) * 0.35
        ss = (_source_score(row.get("source")) - 1.0) * 0.15 + 0.15
        return rs + ks + ss

    return df.apply(_row_score, axis=1)


# =========================
# Vector DB helpers (Chroma)
# =========================

def _hash_df(df: pd.DataFrame) -> str:
    sig = io.BytesIO()
    cols = [c for c in df.columns if c.lower() in ["id", "title", "summary", "description", "content", "url", "source", "date", "published"]]
    df[cols].to_csv(sig, index=False)
    sig = hashlib.sha256(sig.getvalue()).hexdigest()[:16]
    return sig

def _ensure_news_collection(df: pd.DataFrame, persist_dir: str, openai_key: str):
    """
    Create/load a Chroma collection for the uploaded CSV.
    We persist per-hash so different CSVs don't collide.
    """
    first_time = not os.path.exists(persist_dir)
    os.makedirs(persist_dir, exist_ok=True)
    if first_time:
        st.sidebar.info("🧠 Creating new vectorstore/news folder (first-time setup)…")

    client = chromadb.PersistentClient(path=persist_dir)

    def pick_text(row) -> str:
        for col in ["summary", "description", "content"]:
            if col in df.columns and isinstance(row.get(col), str) and row.get(col).strip():
                return row.get(col)
        return str(row.get("title", ""))

    if "_parsed_date" not in df.columns:
        date_col = None
        for c in ["date", "published", "pub_date", "time"]:
            if c in df.columns:
                date_col = c
                break
        df["_parsed_date"] = df[date_col].apply(_parse_date) if date_col else None

    if "id" not in df.columns:
        df = df.reset_index().rename(columns={"index": "id"})

    h = _hash_df(df)
    col_name = f"news_{h}"

    collection = client.get_or_create_collection(
        name=col_name,
        embedding_function=embedding_functions.OpenAIEmbeddingFunction(
            api_key=openai_key,
            model_name="text-embedding-3-small",
        ),
    )

    try:
        if collection.count() > 0:
            st.sidebar.success(f"🔁 Vector DB ready (collection: {col_name}).")
            return collection, col_name
    except Exception:
        pass

    docs, ids, metas = [], [], []
    for _, row in df.iterrows():
        text = f"{row.get('title','')}\n\n{pick_text(row)}"
        docs.append(text)
        ids.append(str(row.get("id")))
        metas.append({
            "title": row.get("title", ""),
            "source": row.get("source", ""),
            "url": row.get("url", ""),
            "date": str(row.get("date", row.get("published",""))),
        })

    if docs:
        collection.add(documents=docs, ids=ids, metadatas=metas)
        st.sidebar.success(f"✅ Embedded {len(docs)} stories into {col_name}.")

    return collection, col_name

def _retrieve(collection, query: str, k: int = 8) -> Tuple[str, List[Dict[str, Any]]]:
    res = collection.query(query_texts=[query], n_results=k)
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    context = "\n\n".join(d for d in docs if d and d.strip())
    return context, metas


# =========================
# LLM helpers (non-streaming)
# =========================

def _run_openai(model: str, api_key: str, prompt: str) -> str:
    client = _OpenAI(api_key=api_key)
    resp = client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}])
    return (resp.choices[0].message.content or "").strip()

def _run_anthropic(model: str, api_key: str, prompt: str) -> str:
    client = _Anthropic(api_key=api_key)
    resp = client.messages.create(model=model, max_tokens=800, messages=[{"role": "user", "content": prompt}])
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
# Prompts
# =========================

def _prompt_topic(memory_txt: str, context: str, question: str) -> str:
    return (
        "You are a News Info Bot for a global law firm. "
        "Answer ONLY using the provided Context. "
        "If the answer is not in the context, reply exactly: "
        "\"I could not find this in the news dataset.\".\n\n"
        f"Conversation (last 5 Q/A):\n{memory_txt}\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer clearly with brief bullet points and include any dates/sources if present."
    )

def _prompt_rerank_for_interest(context_items: List[Dict[str, Any]]) -> str:
    lines = []
    for i, m in enumerate(context_items, 1):
        title = m.get("title", "")
        src = m.get("source", "")
        date = m.get("date", "")
        url = m.get("url", "")
        lines.append(f"{i}. {title} (source: {src}, date: {date}) {url}")
    block = "\n".join(lines)
    return (
        "You are advising partners at a large global law firm. "
        "Rank the following news headlines by **interestingness for legal impact** "
        "(e.g., mergers, lawsuits, regulations, sanctions, settlements, IPOs). "
        "Return a concise ordered list with 1-2 bullets each explaining why it matters.\n\n"
        f"{block}\n\n"
        "Return the ranking as:\n"
        "1) <title> — <why it matters>\n"
        "2) <title> — <why it matters>\n"
        "...\n"
    )


# =========================
# Page: HW7
# =========================

def render():
    st.header("HW7 – News Info Bot (RAG + Ranking)")
    st.caption("Upload a CSV of news stories, then ask topic questions or get the most interesting items.")

    # API keys
    OPENAI_KEY = st.secrets.get("OPENAI_API_KEY")
    ANTHROPIC_KEY = st.secrets.get("ANTHROPIC_API_KEY")
    GEMINI_KEY = st.secrets.get("GEMINI_API_KEY")
    if not OPENAI_KEY:
        st.error("Missing OPENAI_API_KEY in .streamlit/secrets.toml")
        return
    if not ANTHROPIC_KEY:
        st.warning("ANTHROPIC_API_KEY not set; Anthropic provider will fail.")
    if not GEMINI_KEY:
        st.warning("GEMINI_API_KEY not set; Google provider will fail.")

    # Sidebar controls
    st.sidebar.header("⚙️ Settings")
    provider = st.sidebar.selectbox("LLM Provider", ["OpenAI", "Anthropic", "Google Gemini"])
    tier = st.sidebar.radio("Model tier", ["Advanced", "Lite"], horizontal=True)
    mode = st.sidebar.radio("Mode", ["Chatbot", "Find Most Interesting", "Topic Search", "Evaluation"], index=0)

    model_map = {
        "OpenAI": {"Advanced": "gpt-4.1", "Lite": "gpt-5-chat-latest"},
        "Anthropic": {"Advanced": "claude-opus-4-1", "Lite": "claude-3-5-haiku-latest"},
        "Google Gemini": {"Advanced": "gemini-2.5-pro", "Lite": "gemini-2.5-flash-lite"},
    }
    model_name = model_map[provider][tier]
    st.sidebar.write(f"**Using model:** {model_name}")

  

    # normalize columns
    cols_lower = {c.lower(): c for c in df.columns}
    if "title" not in cols_lower:
        st.error("CSV must contain a 'title' column.")
        return
    if "source" not in cols_lower:
        df["source"] = ""
    if not any(c in cols_lower for c in ["summary", "description", "content"]):
        df["summary"] = df[cols_lower["title"]]  # fallback: embed title
    if not any(c in cols_lower for c in ["date", "published", "pub_date", "time"]):
        df["date"] = ""
    if "url" not in cols_lower:
        df["url"] = ""

    with st.expander("👁️ Preview (first 8 rows)"):
        st.dataframe(df.head(8))

    # Build/load vector DB per CSV
    collection, col_name = _ensure_news_collection(df, persist_dir="./vectorstore/news", openai_key=OPENAI_KEY)
    st.success(f"Vector collection ready: {col_name}")

    # Simple short-term memory for Chatbot/Topic modes
    mem_key = f"hw7_{provider}_{tier}"
    if "hw7_chat_key" not in st.session_state or st.session_state.hw7_chat_key != mem_key:
        st.session_state.hw7_chat_key = mem_key
        st.session_state.hw7_chat = []
        st.sidebar.info("🔄 Chat reset for this provider/tier.")

    def _memory_text():
        dq = deque(st.session_state.hw7_chat, maxlen=10)  # last 5 Q/A pairs
        return "\n".join([("User: " if m["role"] == "user" else "Assistant: ") + m["content"] for m in dq])

    # ========= MODE: Chatbot (free-form Q with RAG) =========
    if mode == "Chatbot":
        for m in st.session_state.hw7_chat:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

        user_q = st.chat_input("Ask about the news… (e.g., “What’s happening with antitrust cases this week?”)")
        if not user_q:
            return

        st.session_state.hw7_chat.append({"role": "user", "content": user_q})
        with st.chat_message("user"):
            st.markdown(user_q)

        context, metas = _retrieve(collection, user_q, k=8)
        prompt = _prompt_topic(_memory_text(), context, user_q)

        try:
            if provider == "OpenAI":
                ans = _run_openai(model_name, OPENAI_KEY, prompt)
            elif provider == "Anthropic":
                ans = _run_anthropic(model_name, ANTHROPIC_KEY, prompt)
            else:
                ans = _run_gemini(model_name, GEMINI_KEY, prompt)
            if not ans:
                ans = "I could not find this in the news dataset."
        except Exception as e:
            ans = f"⚠️ {provider}/{model_name} failed: {e}"

        with st.chat_message("assistant"):
            st.markdown(ans)
        st.session_state.hw7_chat.append({"role": "assistant", "content": ans})
        st.session_state.hw7_chat = st.session_state.hw7_chat[-10:]

        if metas:
            with st.expander("📂 Sources (this turn)"):
                for m in metas:
                    title = m.get("title", "")
                    src = m.get("source", "")
                    date = m.get("date", "")
                    url = m.get("url", "")
                    st.write(f"- **{title}** — {src} — {date} {('— ' + url) if url else ''}")

    # ========= MODE: Find Most Interesting =========
    elif mode == "Find Most Interesting":
        st.subheader("🏆 Most Interesting News (Heuristic Ranking)")

        if "_parsed_date" not in df.columns:
            date_col = None
            for c in ["date", "published", "pub_date", "time"]:
                if c in df.columns:
                    date_col = c
                    break
            df["_parsed_date"] = df[date_col].apply(_parse_date) if date_col else None

        scores = compute_interest_scores(df)
        out = df.copy()
        out["_interest"] = scores
        out = out.sort_values("_interest", ascending=False).head(10)

        st.write("Top 10 (heuristic):")
        for i, row in out.reset_index(drop=True).iterrows():
            st.markdown(
                f"**{i+1}. {row.get('title','(no title)')}**  \n"
                f"Source: {row.get('source','')} — Date: {row.get('date', row.get('published',''))}  \n"
                f"Score: `{row.get('_interest'):.3f}`  \n"
                f"{row.get('url','')}"
            )

        # Optional: ask LLM to re-rank the top-N by legal impact
        if st.checkbox("Use LLM to re-rank by legal impact (Advanced vs Lite)"):
            top_items = out.head(8)[["title", "source", "date", "url"]].to_dict("records")
            prompt = _prompt_rerank_for_interest(top_items)

            model_map = {
                "OpenAI": {"Advanced": "gpt-4.1", "Lite": "gpt-5-chat-latest"},
                "Anthropic": {"Advanced": "claude-opus-4-1", "Lite": "claude-3-5-haiku-latest"},
                "Google Gemini": {"Advanced": "gemini-2.5-pro", "Lite": "gemini-2.5-flash-lite"},
            }
            adv = model_map[provider]["Advanced"]
            lite = model_map[provider]["Lite"]

            st.markdown("#### LLM Re-Rank – Advanced")
            try:
                if provider == "OpenAI":
                    a_txt = _run_openai(adv, OPENAI_KEY, prompt)
                elif provider == "Anthropic":
                    a_txt = _run_anthropic(adv, ANTHROPIC_KEY, prompt)
                else:
                    a_txt = _run_gemini(adv, GEMINI_KEY, prompt)
                st.write(a_txt or "_(no response)_")
            except Exception as e:
                st.error(f"{provider}/{adv} failed: {e}")

            st.markdown("#### LLM Re-Rank – Lite")
            try:
                if provider == "OpenAI":
                    l_txt = _run_openai(lite, OPENAI_KEY, prompt)
                elif provider == "Anthropic":
                    l_txt = _run_anthropic(lite, ANTHROPIC_KEY, prompt)
                else:
                    l_txt = _run_gemini(lite, GEMINI_KEY, prompt)
                st.write(l_txt or "_(no response)_")
            except Exception as e:
                st.error(f"{provider}/{lite} failed: {e}")

    # ========= MODE: Topic Search (RAG) =========
    elif mode == "Topic Search":
        topic_q = st.text_input("Find news about (topic)…", value="merger enforcement in tech")
        if st.button("Search"):
            context, metas = _retrieve(collection, topic_q, k=8)
            prompt = _prompt_topic("", context, topic_q)

            try:
                if provider == "OpenAI":
                    ans = _run_openai(model_name, OPENAI_KEY, prompt)
                elif provider == "Anthropic":
                    ans = _run_anthropic(model_name, ANTHROPIC_KEY, prompt)
                else:
                    ans = _run_gemini(model_name, GEMINI_KEY, prompt)
                if not ans:
                    ans = "I could not find this in the news dataset."
            except Exception as e:
                ans = f"⚠️ {provider}/{model_name} failed: {e}"

            st.markdown("### Answer")
            st.write(ans)

            if metas:
                with st.expander("📂 Sources"):
                    for m in metas:
                        title = m.get("title", "")
                        src = m.get("source", "")
                        date = m.get("date", "")
                        url = m.get("url", "")
                        st.write(f"- **{title}** — {src} — {date} {('— ' + url) if url else ''}")

    # ========= MODE: Evaluation =========
    else:
        st.subheader("🔎 Evaluation – Provider Advanced vs Lite")
        QUESTIONS = [
            "Find the most significant mergers mentioned.",
            "List any lawsuits or regulatory actions and summarize them briefly.",
            "Identify IPOs or financing news and their jurisdictions.",
            "What sanctions, fines, or settlements are reported?",
            "Summarize notable antitrust developments this week.",
        ]
        st.write("This runs the same query for **Advanced** and **Lite** models and shows both responses.")

        q = st.selectbox("Pick an evaluation question:", QUESTIONS)
        if st.button("Run Evaluation"):
            context, metas = _retrieve(collection, q, k=10)
            prompt = _prompt_topic("", context, q)

            model_map = {
                "OpenAI": {"Advanced": "gpt-4.1", "Lite": "gpt-5-chat-latest"},
                "Anthropic": {"Advanced": "claude-opus-4-1", "Lite": "claude-3-5-haiku-latest"},
                "Google Gemini": {"Advanced": "gemini-2.5-pro", "Lite": "gemini-2.5-flash-lite"},
            }
            adv = model_map[provider]["Advanced"]
            lite = model_map[provider]["Lite"]

            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**Advanced – {adv}**")
                try:
                    if provider == "OpenAI":
                        a_txt = _run_openai(adv, OPENAI_KEY, prompt)
                    elif provider == "Anthropic":
                        a_txt = _run_anthropic(adv, ANTHROPIC_KEY, prompt)
                    else:
                        a_txt = _run_gemini(adv, GEMINI_KEY, prompt)
                    st.write(a_txt or "_(no response)_")
                except Exception as e:
                    st.error(f"{provider}/{adv} failed: {e}")

            with c2:
                st.markdown(f"**Lite – {lite}**")
                try:
                    if provider == "OpenAI":
                        l_txt = _run_openai(lite, OPENAI_KEY, prompt)
                    elif provider == "Anthropic":
                        l_txt = _run_anthropic(lite, ANTHROPIC_KEY, prompt)
                    else:
                        l_txt = _run_gemini(lite, GEMINI_KEY, prompt)
                    st.write(l_txt or "_(no response)_")
                except Exception as e:
                    st.error(f"{provider}/{lite} failed: {e}")

            if metas:
                with st.expander("📂 Sources used"):
                    for m in metas:
                        title = m.get("title", "")
                        src = m.get("source", "")
                        date = m.get("date", "")
                        url = m.get("url", "")
                        st.write(f"- **{title}** — {src} — {date} {('— ' + url) if url else ''}")
