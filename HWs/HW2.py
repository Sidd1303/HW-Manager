import time
import requests
from bs4 import BeautifulSoup
import streamlit as st

# --------- Provider SDKs ----------
from openai import OpenAI
from anthropic import Anthropic
import google.generativeai as genai

# ------------------ Utility: URL Reader ------------------
def read_url_content(url: str) -> str | None:
    try:
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")

        # Remove script/style/noscript
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        # Normalize whitespace
        text = " ".join(soup.get_text(separator=" ").split())
        return text if text.strip() else None
    except requests.RequestException as e:
        st.error(f"Error reading {url}: {e}")
        return None

# ------------------ Prompt Builder ------------------
def build_prompt(page_text: str, summary_type: str, language: str) -> str:
    style_instructions = {
        "100 words": "Write ~100 words, one compact paragraph, precise and neutral.",
        "Two paragraphs": "Write exactly two connected paragraphs (~4-6 sentences each).",
        "Five bullet points": "Write exactly five concise bullet points, no extras.",
    }
    return f"""
You are a careful, faithful summarizer.

TASK:
- Summarize the following web page content.
- Output language: {language}.
- Format: {style_instructions[summary_type]}
- Be accurate, grounded strictly in the provided text (no outside facts).
- Avoid speculation. If the page has little text, say so concisely.

CONTENT START
{page_text}
CONTENT END
"""

# ------------------ Provider Runners ------------------
def run_openai(model: str, prompt: str, temperature: float, max_tokens: int):
    client = OpenAI(api_key=st.secrets.get("OPENAI_API_KEY"))
    t0 = time.time()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    latency = time.time() - t0
    text = resp.choices[0].message.content
    return text, latency

def run_anthropic(model: str, prompt: str, temperature: float, max_tokens: int):
    client = Anthropic(api_key=st.secrets.get("ANTHROPIC_API_KEY"))
    t0 = time.time()
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    latency = time.time() - t0
    text = "\n".join(
        blk.text if hasattr(blk, "text") else blk.get("text", "")
        for blk in resp.content
    ).strip()
    return text, latency

def run_gemini(model_name: str, prompt: str, temperature: float, max_tokens: int):
    genai.configure(api_key=st.secrets.get("GEMINI_API_KEY"))
    model = genai.GenerativeModel(model_name=model_name)
    t0 = time.time()
    resp = model.generate_content(
        prompt,
        generation_config={
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        },
    )
    latency = time.time() - t0
    text = getattr(resp, "text", "") or ""
    return text, latency

# ------------------ Provider Registry ------------------
PROVIDERS = {
    "OpenAI": {
        "runner": run_openai,
        "advanced": "gpt-4o",
        "standard": "gpt-4o-mini",
        "secret": "OPENAI_API_KEY",
    },
    "Anthropic (Claude)": {
        "runner": run_anthropic,
        "advanced": "claude-opus-4-1",
        "standard": "claude-sonnet-4",
        "secret": "ANTHROPIC_API_KEY",
    },
    "Google (Gemini)": {
        "runner": run_gemini,
        "advanced": "gemini-2.5-pro",
        "standard": "gemini-2.5-flash-lite",
        "secret": "GEMINI_API_KEY",
    },
}

LANG_OPTIONS = ["English", "French", "Spanish"]
SUMMARY_OPTIONS = ["100 words", "Two paragraphs", "Five bullet points"]

def has_key(secret_name: str) -> bool:
    v = st.secrets.get(secret_name)
    return bool(v and isinstance(v, str) and len(v.strip()) > 0)

# ------------------ Streamlit UI ------------------
def render():
    st.header("HW2 – URL Summarizer for Multiple LLMs")
    st.write("Enter a URL, choose summary type, output language, and provider/model.")

    url = st.text_input("🌐 Web page URL", placeholder="https://example.com/article")

    with st.sidebar:
        st.subheader("Summary Settings")
        summary_type = st.selectbox("Type of summary", SUMMARY_OPTIONS, index=0)
        language = st.selectbox("Output language", LANG_OPTIONS, index=0)

        st.subheader("Model Settings")
        provider_name = st.selectbox("LLM Provider", list(PROVIDERS.keys()), index=0)
        use_advanced = st.checkbox("Use advanced model", value=True)
        temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
        max_tokens = st.number_input("Max output tokens", min_value=128, max_value=4096, value=1000, step=64)

    col1, col2 = st.columns([1, 1])
    with col1:
        go = st.button("Summarize", type="primary")
    with col2:
        compare = st.button("Compare 3 providers")

    if go:
        if not url:
            st.warning("Please enter a URL.")
            return
        with st.spinner("Fetching and parsing page..."):
            page_text = read_url_content(url)
        if not page_text:
            st.error("No readable text found at this URL.")
            return

        prompt = build_prompt(page_text, summary_type, language)
        cfg = PROVIDERS[provider_name]
        model = cfg["advanced"] if use_advanced else cfg["standard"]

        if not has_key(cfg["secret"]):
            st.error(f"Missing {cfg['secret']} in secrets.toml")
            return

        st.write(f"**Provider:** {provider_name}  \n**Model:** `{model}`")
        with st.spinner("Generating summary..."):
            try:
                text, latency = cfg["runner"](model, prompt, temperature, max_tokens)
                st.success(f"Done in {latency:.2f}s")
                st.markdown("### Summary")
                st.write(text)
            except Exception as e:
                st.error(f"{provider_name} failed: {e}")

    if compare:
        if not url:
            st.warning("Please enter a URL.")
            return
        with st.spinner("Fetching and parsing page..."):
            page_text = read_url_content(url)
        if not page_text:
            st.error("No readable text found at this URL.")
            return

        prompt = build_prompt(page_text, summary_type, language)
        picks = ["OpenAI", "Anthropic (Claude)", "Google (Gemini)"]

        cols = st.columns(len(picks))
        for i, pname in enumerate(picks):
            cfg = PROVIDERS[pname]
            model = cfg["advanced"] if use_advanced else cfg["standard"]
            with cols[i]:
                st.write(f"**{pname}**")
                if not has_key(cfg["secret"]):
                    st.error("Missing API key")
                    continue
                try:
                    text, latency = cfg["runner"](model, prompt, temperature, max_tokens)
                    st.success(f"{model} ✓ ({latency:.2f}s)")
                    st.write(text)
                except Exception as e:
                    st.error(f"Error: {e}")