"""
Task 1 — AI Chatbot (Streamlit Web UI)  |  CODSOFT AI Internship
Author: SRIKRISH-S | github.com/SRIKRISH-S/CODSOFT

Wraps the existing ProfessionalChatBot logic in a polished Streamlit UI.
"""

import streamlit as st
import json, random, re, datetime, sys, os
from pathlib import Path

# ── Resolve intents.json path ─────────────────────────────────────────────────
BASE = Path(__file__).parent
INTENTS_FILE = BASE / "intents.json"

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="NexusBot · AI Chatbot", page_icon="🤖",
                   layout="centered", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

*, html, body { font-family: 'Inter', sans-serif; }
.stApp { background: #111318; color: #e2e8f0; }

#MainMenu, footer, header { visibility: hidden; }
[data-testid="stSidebar"] { background: #0d0f14 !important; }

/* Header */
.chat-header {
    display: flex; align-items: center; gap: 1rem;
    padding: 1.2rem 1.5rem; background: #1a1d26;
    border-radius: 16px; margin-bottom: 1.5rem;
    border: 1px solid #252833;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}
.avatar { width: 48px; height: 48px; border-radius: 50%;
          background: linear-gradient(135deg, #00b4d8, #0077b6);
          display: flex; align-items: center; justify-content: center;
          font-size: 1.4rem; flex-shrink: 0; }
.bot-name  { font-size: 1.1rem; font-weight: 700; color: #e2e8f0; }
.bot-status { font-size: 0.75rem; color: #22d3ee; }

/* Messages */
.msg-wrap { margin-bottom: 1rem; }
.msg-user  { display: flex; justify-content: flex-end; }
.msg-bot   { display: flex; justify-content: flex-start; align-items: flex-end; gap: 8px; }

.bubble-user {
    background: linear-gradient(135deg, #0077b6, #00b4d8);
    color: white; padding: 0.7rem 1.1rem;
    border-radius: 18px 18px 4px 18px;
    max-width: 75%; font-size: 0.9rem; line-height: 1.5;
    box-shadow: 0 2px 10px rgba(0,119,182,0.3);
}
.bubble-bot {
    background: #1e2130; color: #c5cde0;
    padding: 0.7rem 1.1rem;
    border-radius: 18px 18px 18px 4px;
    max-width: 75%; font-size: 0.9rem; line-height: 1.5;
    border: 1px solid #2a2f40;
}
.mini-avatar {
    width: 28px; height: 28px; border-radius: 50%;
    background: linear-gradient(135deg, #00b4d8, #0077b6);
    display: flex; align-items: center; justify-content: center;
    font-size: 0.85rem; flex-shrink: 0;
}
.msg-time { font-size: 0.65rem; color: #4a5070; margin-top: 4px; text-align: right; }

/* Input area */
.stTextInput > div > div > input {
    background: #1a1d26 !important; color: #e2e8f0 !important;
    border: 1px solid #252833 !important; border-radius: 12px !important;
    padding: 0.7rem 1rem !important; font-size: 0.9rem !important;
}
.stTextInput > div > div > input:focus { border-color: #00b4d8 !important;
    box-shadow: 0 0 0 3px rgba(0,180,216,0.15) !important; }

.stButton > button {
    background: linear-gradient(135deg, #0077b6, #00b4d8) !important;
    color: white !important; border: none !important;
    border-radius: 12px !important; font-weight: 600 !important;
    padding: 0.65rem 1.4rem !important; transition: all 0.2s !important;
}
.stButton > button:hover { transform: translateY(-1px) !important;
    box-shadow: 0 4px 16px rgba(0,180,216,0.4) !important; }

.quick-chip {
    display: inline-block; background: #1a1d26; border: 1px solid #252833;
    border-radius: 20px; padding: 5px 14px; font-size: 0.78rem;
    color: #94a3b8; cursor: pointer; margin: 3px; transition: all 0.15s;
}
.quick-chip:hover { border-color: #00b4d8; color: #00b4d8; }

.stats-bar {
    background: #1a1d26; border: 1px solid #252833; border-radius: 12px;
    padding: 0.7rem 1.2rem; display: flex; gap: 2rem; margin-bottom: 1rem;
    font-size: 0.8rem;
}
.stat { text-align: center; }
.stat-n { font-weight: 700; color: #00b4d8; font-size: 1rem; }
.stat-l { color: #4a5070; font-size: 0.7rem; }
</style>
""", unsafe_allow_html=True)

# ── Bot Logic ─────────────────────────────────────────────────────────────────
try:
    from fuzzywuzzy import fuzz
    FUZZY = True
except ImportError:
    FUZZY = False

@st.cache_resource
def load_bot():
    try:
        with open(INTENTS_FILE, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("intents", [])
    except Exception:
        return []

intents = load_bot()

def fuzzy_score(a, b):
    if FUZZY:
        return fuzz.token_set_ratio(a, b)
    a, b = a.lower(), b.lower()
    return 100 if a in b or b in a else (70 if any(w in b for w in a.split()) else 0)

def get_response(user_input: str, memory: dict) -> str:
    text = user_input.lower().strip()

    # Skills
    if "my name is" in text:
        m = re.search(r"my name is (\w+)", text)
        if m:
            memory["name"] = m.group(1).capitalize()
            return f"Nice to meet you, {memory['name']}! I'll remember that. 😊"
    if any(q in text for q in ["what is my name", "what's my name"]):
        if memory.get("name"):
            return f"Your name is **{memory['name']}**!"
        return "You haven't told me your name yet. Say 'My name is ...' and I'll remember!"
    if "calculate" in text:
        expr = text.split("calculate", 1)[-1]
        try:
            clean = re.sub(r"[^\d+\-*/().\s]", "", expr)
            if clean.strip():
                return f"The result is **{eval(clean)}**."
        except Exception:
            pass
        return "I couldn't calculate that. Try: 'calculate 25 * 4'"
    if "time" in text:
        return f"The current time is **{datetime.datetime.now().strftime('%I:%M %p')}**."
    if "date" in text:
        return f"Today is **{datetime.datetime.now().strftime('%A, %B %d, %Y')}**."

    # Intent matching
    best, best_score = None, 0
    for intent in intents:
        for pattern in intent["patterns"]:
            score = fuzzy_score(text, pattern.lower())
            if score > best_score:
                best_score, best = score, intent
    if best and best_score > 65:
        resp = random.choice(best["responses"])
        if best.get("tag") == "greeting" and memory.get("name"):
            resp += f" Great to see you, {memory['name']}!"
        return resp

    return "I'm not sure I understand that. Could you rephrase? 🤔"

# ── Session State ─────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "memory" not in st.session_state:
    st.session_state.memory = {}
if "msg_count" not in st.session_state:
    st.session_state.msg_count = 0

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="chat-header">
    <div class="avatar">🤖</div>
    <div>
        <div class="bot-name">NexusBot</div>
        <div class="bot-status">● Online · Rule-Based AI · Fuzzy Matching NLP</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Stats bar
known = len(intents)
msgs  = st.session_state.msg_count
name  = st.session_state.memory.get("name", "—")
st.markdown(
    f'<div class="stats-bar">'
    f'<div class="stat"><div class="stat-n">{known}</div><div class="stat-l">Intents</div></div>'
    f'<div class="stat"><div class="stat-n">{msgs}</div><div class="stat-l">Messages</div></div>'
    f'<div class="stat"><div class="stat-n">{"Fuzzy" if FUZZY else "Regex"}</div><div class="stat-l">NLP Mode</div></div>'
    f'<div class="stat"><div class="stat-n">{name}</div><div class="stat-l">Your Name</div></div>'
    f'</div>',
    unsafe_allow_html=True)

# ── Welcome message ───────────────────────────────────────────────────────────
if not st.session_state.history:
    st.markdown(
        '<div class="msg-wrap"><div class="msg-bot">'
        '<div class="mini-avatar">🤖</div>'
        '<div><div class="bubble-bot">👋 Hi! I\'m <strong>NexusBot</strong>, your AI assistant.<br>'
        'I can answer questions, do math, tell the time, and more.<br>'
        'Try saying <em>"Hello"</em>, <em>"What can you do?"</em>, or <em>"My name is ..."</em></div>'
        '<div class="msg-time">Just now</div></div></div></div>',
        unsafe_allow_html=True)

# ── Chat History ──────────────────────────────────────────────────────────────
for turn in st.session_state.history:
    when = turn.get("time", "")
    if turn["role"] == "user":
        st.markdown(
            f'<div class="msg-wrap"><div class="msg-user">'
            f'<div><div class="bubble-user">{turn["text"]}</div>'
            f'<div class="msg-time">{when}</div></div></div></div>',
            unsafe_allow_html=True)
    else:
        st.markdown(
            f'<div class="msg-wrap"><div class="msg-bot">'
            f'<div class="mini-avatar">🤖</div>'
            f'<div><div class="bubble-bot">{turn["text"]}</div>'
            f'<div class="msg-time">{when}</div></div></div></div>',
            unsafe_allow_html=True)

# ── Quick suggestions ─────────────────────────────────────────────────────────
suggestions = ["Hello! 👋", "What can you do?", "What time is it?", "Calculate 42 * 7", "Tell me a joke"]
st.markdown("**Quick replies:**")
cols = st.columns(len(suggestions))
for col, sug in zip(cols, suggestions):
    with col:
        if st.button(sug, key=f"sug_{sug}", use_container_width=True):
            st.session_state["_pending"] = sug
            st.rerun()

# ── Input ─────────────────────────────────────────────────────────────────────
with st.form("chat_form", clear_on_submit=True):
    col_inp, col_btn = st.columns([5, 1])
    with col_inp:
        user_msg = st.text_input("", placeholder="Type a message…", label_visibility="collapsed",
                                 value=st.session_state.pop("_pending", ""))
    with col_btn:
        submitted = st.form_submit_button("Send", use_container_width=True)

if submitted and user_msg.strip():
    now = datetime.datetime.now().strftime("%I:%M %p")
    response = get_response(user_msg, st.session_state.memory)
    st.session_state.history.append({"role":"user", "text": user_msg, "time": now})
    st.session_state.history.append({"role":"bot",  "text": response, "time": now})
    st.session_state.msg_count += 1

    if user_msg.lower() in ("bye","exit","quit","goodbye"):
        st.session_state.history.append({"role":"bot","text":"👋 Goodbye! Come back anytime!","time":now})

    st.rerun()

# ── Clear ─────────────────────────────────────────────────────────────────────
if st.session_state.history:
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.history = []
        st.session_state.msg_count = 0
        st.rerun()

st.markdown('<div style="text-align:center;color:#252833;font-size:0.72rem;margin-top:1rem;">'
            'CODSOFT AI Internship · Task 1 · '
            '<a href="https://github.com/SRIKRISH-S/CODSOFT" style="color:#00b4d8;">github.com/SRIKRISH-S</a></div>',
            unsafe_allow_html=True)
