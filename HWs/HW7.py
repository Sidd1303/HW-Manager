# HWs/HW7.py
# HW7 – News RAG & Ranking Bot (ChromaDB persistent)
from __future__ import annotations

import os
import re
import csv
import math
from datetime import datetime
from collections import deque
from typing import List, Dict, Tuple, Optional

import streamlit as st

# --- Make Chroma use a modern sqlite even on older images (codespaces/streamlit) ---
try:
    __import__("pysqlite3")  # pip install pysqlite3-binary
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except Exception:
    # If not available, Chroma may still work if system sqlite >= 3.35
    pass

import chromadb
from chromadb.utils import embedding_functions

# LLM SDKs (non-streaming; no temp/max_tokens to satisfy new non-reasoning models)
from openai import OpenAI as _OpenAI
from anthropic import Anthropic as _Anthropic
import google.generativeai as _genai


# =========================
# Basic CSV loader (no pandas to avoid version pin issues)
# =========================
def load_news_csv(path: str) -> List[Dict[str, str]]:
    """
    Load CSV into a list of dicts.
    Expected helpful columns if present (not required):
      id, title, text/content, date, published, url, source, region, topic, category
    We'll synthesize a 'full_text' = title + " - " + text (if available).
    """
    if not os.path.isfile(path):
        return []
    rows: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for r in reader:
            # Normalize keys (lowercase)
            r_norm = { (k or "").strip().lower(): (v or "").strip() for k, v in r.items() }
            # Build 'full_text'
            title = r_norm.get("title", "") or r_norm.get("headline", "")
            text = r_norm.get("text", "") or r_norm.get("content", "") or r_norm.get("body", "")
            full_text = (title + " - " + text).strip(" -")
            r_norm["full_text"] = full_text
            rows.append(r_norm)
    return rows


def parse_date(d: str) -> Optional[datetime]:
    if not d:
        return None
    # Try a few common patterns
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%d-%m-%Y", "%Y/%m/%d", "%b %d, %Y", "%d %b %Y"):
        try:
            return datetime.strptime(d, fmt)
        except Exception:
            continue
    # ISO-ish
    try:
        return datetime.fromisoformat(d.replace("Z", "+00:00"))
    except Exception:
        return None


# =========================
# Heuristic "Interestingness" scoring for a global law firm
# =========================
LEGAL_KEYWORDS = [
    "lawsuit", "litigation", "sue", "sued", "settlement", "class action",
    "indictment", "subpoena", "injunction", "arbitration", "mediation",
    "regulator", "regulatory", "sanction", "fine", "penalty",
    "compliance", "breach", "fraud", "insider trading",
    "sec", "doj", "ftc", "ofac", "cftc", "eu commission",
    "gdpr", "hipaa", "ccpa", "fca", "bribery", "further investigation",
    "intellectual property", "patent", "trademark", "copyright",
]

CURRENCY_RE = re.compile(r"(\$|usd|eur|£|€)\s?([\d,.]+)\s*(million|billion|m|bn|b)?", re.I)

JURISDICTIONS = [
    # Short list; expand as desired
    "united states", "u.s.", "u.s.a", "usa", "european union", "eu",
    "united kingdom", "uk", "england", "wales", "scotland",
    "germany", "france", "italy", "spain", "canada", "mexico",
    "brazil", "china", "india", "japan", "australia", "singapore",
    "hong kong", "uae", "saudi arabia", "south africa",
]

def money_score(text: str) -> float:
    """
    Score based on monetary amounts mentioned.
    Rough heuristic: any money mention adds log-scaled points.
    """
    score = 0.0
    for m in CURRENCY_RE.finditer(text):
        _, amt, scale = m.groups()
        try:
            val = float(amt.replace(",", ""))
        except Exception:
            val = 0.0
        scale = (scale or "").lower()
        if scale in ("million", "m"):
            val *= 1_000_000
        elif scale in ("billion", "bn", "b"):
            val *= 1_000_000_000
        # log scale to avoid domination
        if val > 0:
            score += math.log10(val + 1)
    return score

def legal_keyword_score(text: str) -> float:
    t = text.lower()
    return sum(1.0 for kw in LEGAL_KEYWORDS if kw in t)

def jurisdiction_score(text: str) -> float:
    t = text.lower()
    return sum(0.5 for j in JURISDICTIONS if j in t)

def recency_score(dt: Optional[datetime]) -> float:
    """Favor newer items; fallback to small base if no date."""
    if not dt:
        return 0.1
    days = max(1, (datetime.utcnow() - dt).days)
    # Newer -> larger; use inverse decay
    return 5.0 / math.log(days + 5.0)

def interestingness(row: Dict[str, str]) -> float:
    """
    Composite score reflecting a global law firm's likely priorities:
    legal/regulatory signals, monetary impact, jurisdiction, recency.
    """
    text = row.get("full_text", "")
    date = parse_date(row.get("date", "") or row.get("published", ""))

    s_money = money_score(text)
    s_legal = legal_keyword_score(text)
    s_juris = jurisdiction_score(text)
    s_rec = recency_score(date)

    # Weighted sum (tune as needed)
    return 1.2 * s_legal + 1.0 * s_money + 0.6 * s_juris + 1.0 * s_rec


# =========================
# ChromaDB persistent store
# =========================
def build_or_load_news_collection(
    items: List[Dict[str, str]],
    persist_dir: str,
    openai_key: str,
    collection_name: str = "news_items",
) -> chromadb.api.models.Collection.Collection:
    os.makedirs(persist_dir, exist_ok=True)
    client = chromadb.PersistentClient(path=persist_dir)

    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_functions.OpenAIEmbeddingFunction(
            api_key=openai_key,
            model_name="text-embedding-3-small",
        ),
    )

    # If it already has content, return
    try:
        existing = collection.count()
    except Exception:
        existing = len(collection.get().get("ids", []))
    if existing and existing > 0:
        return collection

    docs, ids, metas = [], [], []
    for i, row in enumerate(items):
        txt = row.get("full_text", "")
        if not txt.strip():
            continue
        doc_id = str(row.get("id") or f"row_{i}")
        docs.append(txt)
        ids.append(doc_id)
        metas.append({
            "title": row.get("title", ""),
            "url": row.get("url", ""),
            "source": row.get("source", ""),
            "date": row.get("date", "") or row.get("published", ""),
        })

    if docs:
        collection.add(documents=docs, ids=ids, metadatas=metas)

    return collection


def retrieve_similar(collection, query: str, k: int = 5) -> Tuple[List[Dict[str, str]], List[str]]:
    res = collection.query(query_texts=[query], n_results=k)
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    ids = res.get("ids", [[]])[0]
    rows = []
    for i in range(len(docs)):
        rows.append({
            "id": ids[i],
            "text": docs[i],
            "meta": metas[i],
        })
    return rows, ids


# =========================
# LLM runners (non-streaming)
# =========================
def run_openai(model: str, api_key: str, prompt: str) -> str:
    client = _OpenAI(api_key=api_key)
    resp = client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}])
    return (resp.choices[0].message.content or "").strip()

def run_anthropic(model: str, api_key: str, prompt: str) -> str:
    client = _Anthropic(api_key=api_key)
    resp = client.messages.create(model=model, max_tokens=900, messages=[{"role": "user", "content": prompt}])
    try:
        return (resp.content[0].text or "").strip()
    except Exception:
        return ""

def run_gemini(model: str, api_key: str, prompt: str) -> str:
    _genai.configure(api_key=api_key)
    g = _genai.GenerativeModel(model)
    r = g.generate_content(prompt)
    return (getattr(r, "text", "") or "").strip()


# =========================
# Prompt helpers
# =========================
def serialize_context(rows: List[Dict[str, str]]) -> str:
    parts = []
    for r in rows:
        m = r["meta"]
        title = m.get("title", "") or "(no title)"
        url = m.get("url", "")
        date = m.get("date", "")
        src = m.get("source", "")
        snippet = r["text"][:800]
        parts.append(f"[Title] {title}\n[URL] {url}\n[Source] {src}\n[Date] {date}\n[Text] {snippet}\n---")
    return "\n".join(parts)

def short_memory_text(history: List[Dict[str, str]], max_pairs: int = 5) -> str:
    dq = deque(history, maxlen=max_pairs * 2)
    out = []
    for msg in dq:
        role = "User" if msg["role"] == "user" else "Assistant"
        out.append(f"{role}: {msg['content']}")
    return "\n".join(out)

def qa_prompt(memory: str, context: str, question: str) -> str:
    return (
        "You are a news analysis assistant for a global law firm.\n"
        "Use ONLY the provided Context when answering.\n"
        "If the Context does not contain the answer, say exactly:\n"
        "\"I could not find this in the news dataset.\"\n\n"
        f"Conversation (recent Q/A):\n{memory}\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer clearly and cite the specific titles you used."
    )

def ranking_prompt(context_list_text: str, task_desc: str) -> str:
    return (
        "You are ranking news for a global law firm. Prioritize:\n"
        "- Legal/regulatory exposure (lawsuits, regulators, enforcement, compliance)\n"
        "- Monetary impact (fines, settlements, M&A scale)\n"
        "- Jurisdictional significance (US/EU/UK/major markets)\n"
        "- Recency and strategic risk\n"
        "Return a numbered list with brief rationale and include title + url for each.\n\n"
        f"Task: {task_desc}\n\n"
        f"Candidates:\n{context_list_text}\n\n"
        "Now output the ranked list."
    )


# =========================
# App (Chat + Eval + Ranking)
# =========================
def render():
    st.header("HW7 – News RAG & Ranking Bot (ChromaDB)")
    st.caption("Auto-loads ./news_data/news_dataset.csv, builds a persistent Chroma vector store, "
               "supports topic search, 'most interesting' ranking for a global law firm, "
               "and compares Advanced vs Lite models per vendor.")

    # --- Keys from .streamlit/secrets.toml
    KEYS = {
        "OPENAI_API_KEY": st.secrets.get("OPENAI_API_KEY"),
        "ANTHROPIC_API_KEY": st.secrets.get("ANTHROPIC_API_KEY"),
        "GEMINI_API_KEY": st.secrets.get("GEMINI_API_KEY"),
    }
    if not KEYS["OPENAI_API_KEY"]:
        st.error("Missing OPENAI_API_KEY in .streamlit/secrets.toml")
        return
    if not KEYS["ANTHROPIC_API_KEY"]:
        st.warning("ANTHROPIC_API_KEY not set; Anthropic runs will fail.")
    if not KEYS["GEMINI_API_KEY"]:
        st.warning("GEMINI_API_KEY not set; Gemini runs will fail.")

    # --- Sidebar controls
    st.sidebar.header("⚙️ Settings")
    provider = st.sidebar.selectbox("Provider", ["OpenAI", "Anthropic", "Google Gemini"])
    mode = st.sidebar.radio("Mode", ["Chatbot", "Most Interesting", "Topic Search", "Evaluation"], horizontal=False)

    # Advanced vs Lite models (non-reasoning)
    model_map = {
        "OpenAI": { "Advanced": "gpt-4.1", "Lite": "gpt-5-chat-latest" },
        "Anthropic": { "Advanced": "claude-opus-4-1", "Lite": "claude-3-5-haiku-latest" },
        "Google Gemini": { "Advanced": "gemini-2.5-pro", "Lite": "gemini-2.5-flash-lite" },
    }
    adv_model = model_map[provider]["Advanced"]
    lite_model = model_map[provider]["Lite"]
    st.sidebar.write(f"**Advanced:** {adv_model}")
    st.sidebar.write(f"**Lite:** {lite_model}")

    # --- Load CSV & build/load Chroma (one-time)
    CSV_PATH = "./news_data/news_dataset.csv"
    data = load_news_csv(CSV_PATH)
    if not data:
        st.error(f"Could not load any rows from {CSV_PATH}.")
        return

    collection = build_or_load_news_collection(
        items=data,
        persist_dir="./vectorstore/news",
        openai_key=KEYS["OPENAI_API_KEY"],
        collection_name="news_items",
    )

    # =====================
    # Helper: call model
    # =====================
    def call_model(model_name: str, prompt: str) -> str:
        if provider == "OpenAI":
            return run_openai(model_name, KEYS["OPENAI_API_KEY"], prompt)
        elif provider == "Anthropic":
            return run_anthropic(model_name, KEYS["ANTHROPIC_API_KEY"], prompt)
        else:
            return run_gemini(model_name, KEYS["GEMINI_API_KEY"], prompt)

    # =====================
    # CHATBOT (RAG Q&A)
    # =====================
    if mode == "Chatbot":
        tier = st.sidebar.radio("Chat Model", ["Advanced", "Lite"], horizontal=True)
        chat_model = adv_model if tier == "Advanced" else lite_model

        # Reset chat when switching model/provider
        chat_key = f"{provider}:{tier}"
        if "hw7_chat_key" not in st.session_state or st.session_state.hw7_chat_key != chat_key:
            st.session_state.hw7_chat_key = chat_key
            st.session_state.hw7_chat = []
            st.sidebar.info(f"🔄 Chat reset for {chat_key}")

        # Show history
        for m in st.session_state.hw7_chat:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

        user_q = st.chat_input("Ask about the news… (e.g., 'Summarize enforcement actions on data privacy')")

        if user_q:
            st.session_state.hw7_chat.append({"role": "user", "content": user_q})
            with st.chat_message("user"):
                st.markdown(user_q)

            # Retrieve top 5 related stories
            rows, _ = retrieve_similar(collection, user_q, k=5)
            context = serialize_context(rows)
            memory = short_memory_text(st.session_state.hw7_chat, max_pairs=5)
            prompt = qa_prompt(memory, context, user_q)

            try:
                ans = call_model(chat_model, prompt)
                if not ans.strip():
                    ans = "I could not find this in the news dataset."
            except Exception as e:
                ans = f"⚠️ {provider} / {chat_model} failed: {e}"

            with st.chat_message("assistant"):
                st.markdown(ans)
            st.session_state.hw7_chat.append({"role": "assistant", "content": ans})

            # Sources
            with st.expander("📂 Sources used (this turn)"):
                for r in rows:
                    m = r["meta"]
                    st.write(f"- {m.get('title','(no title)')} — {m.get('url','')}")

    # =====================
    # MOST INTERESTING
    # =====================
    elif mode == "Most Interesting":
        st.subheader("Rank: Most Interesting News for a Global Law Firm")
        k = st.slider("How many top stories?", min_value=3, max_value=15, value=7, step=1)

        # Compute heuristic score for all items
        scored = []
        for row in data:
            s = interestingness(row)
            scored.append((s, row))
        scored.sort(key=lambda x: x[0], reverse=True)
        top = [r for _, r in scored[:max(20, k*2)]]  # give LLM more candidates to re-rank

        # Turn candidates into context for LLM ranking
        candidate_blocks = []
        for i, r in enumerate(top, 1):
            title = r.get("title", "") or "(no title)"
            url = r.get("url", "")
            src = r.get("source", "")
            date = r.get("date", "") or r.get("published", "")
            snippet = r.get("full_text", "")[:600]
            candidate_blocks.append(f"[{i}] {title}\nURL: {url}\nSource: {src}\nDate: {date}\nText: {snippet}\n---")
        candidates_text = "\n".join(candidate_blocks)

        prompt = ranking_prompt(
            candidates_text,
            task_desc=f"Select and rank the **top {k} most interesting** items for partners at a global law firm."
        )
        # Use Advanced by default for final re-rank (or let user choose)
        tier = st.sidebar.radio("Ranking Model", ["Advanced", "Lite"], horizontal=True, key="rank_tier")
        model_for_rank = adv_model if tier == "Advanced" else lite_model

        try:
            out = call_model(model_for_rank, prompt)
        except Exception as e:
            out = f"⚠️ {provider} / {model_for_rank} failed: {e}"

        st.markdown("### Final Ranked List")
        st.write(out)

    # =====================
    # TOPIC SEARCH
    # =====================
    elif mode == "Topic Search":
        topic = st.text_input("Find news about…", value="antitrust enforcement in the EU")
        k = st.slider("How many results?", min_value=3, max_value=20, value=8, step=1)

        if st.button("Search"):
            rows, _ = retrieve_similar(collection, topic, k=k)
            # Present simple list + have LLM summarize/cluster
            context = serialize_context(rows)
            prompt = (
                "You are a news search assistant for a global law firm.\n"
                "Given the following retrieved articles, list the matching items with titles and URLs, "
                "and write a short synthesis (3–5 bullets) highlighting why they matter to legal practitioners.\n\n"
                f"{context}"
            )
            tier = st.sidebar.radio("Topic Model", ["Advanced", "Lite"], horizontal=True, key="topic_tier")
            model_for_topic = adv_model if tier == "Advanced" else lite_model

            try:
                out = call_model(model_for_topic, prompt)
            except Exception as e:
                out = f"⚠️ {provider} / {model_for_topic} failed: {e}"

            st.markdown("### Results")
            st.write(out)

    # =====================
    # EVALUATION
    # =====================
    else:
        st.subheader("Evaluation – Compare Advanced vs Lite (selected provider)")

        QUESTIONS = [
            "Find the most consequential enforcement actions (fines/settlements) in the last month.",
            "Summarize major M&A items with potential antitrust scrutiny.",
            "Which stories involve cross-border data privacy or GDPR risks?",
            "Identify litigation that could materially impact a company's balance sheet.",
            "What trends should partners brief clients on this week?",
        ]

        st.write("We retrieve a common context per question, then ask both models the same prompt.")

        if st.button("Run Evaluation"):
            for i, q in enumerate(QUESTIONS, 1):
                st.markdown(f"### Q{i}. {q}")
                # Retrieve more articles for robust eval
                rows, _ = retrieve_similar(collection, q, k=8)
                ctx = serialize_context(rows)

                eval_prompt = (
                    "Answer for a global law firm partner. Use only the Context. "
                    "Return a concise, structured answer and cite titles.\n\n"
                    f"Context:\n{ctx}\n\nQuestion: {q}"
                )

                # Advanced
                try:
                    adv_out = call_model(adv_model, eval_prompt)
                    if not adv_out.strip():
                        adv_out = "(no answer)"
                except Exception as e:
                    adv_out = f"⚠️ {provider} / {adv_model} failed: {e}"

                # Lite
                try:
                    lite_out = call_model(lite_model, eval_prompt)
                    if not lite_out.strip():
                        lite_out = "(no answer)"
                except Exception as e:
                    lite_out = f"⚠️ {provider} / {lite_model} failed: {e}"

                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"**Advanced – {adv_model}**")
                    st.write(adv_out)
                with c2:
                    st.markdown(f"**Lite – {lite_model}**")
                    st.write(lite_out)

                with st.expander(f"📂 Sources for Q{i}"):
                    for r in rows:
                        m = r["meta"]
                        st.write(f"- {m.get('title','(no title)')} — {m.get('url','')}")
                st.divider()
