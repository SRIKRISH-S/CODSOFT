"""
Task 2 — Tic-Tac-Toe AI  |  CODSOFT AI Internship
Author: SRIKRISH-S | github.com/SRIKRISH-S/CODSOFT
"""

import streamlit as st
import time
from game import Board, AIEngine, GameSession, X, O, EMPTY

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="TacticAI · Tic-Tac-Toe", page_icon="♟️",
                   layout="wide", initial_sidebar_state="expanded")

# ── THEME ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Orbitron:wght@700;900&display=swap');

*, html, body { font-family: 'Space Grotesk', sans-serif; box-sizing: border-box; }

.stApp { background: #0b0c10; color: #c5c6c7; }

/* ── Sidebar ── */
[data-testid="stSidebar"] { background: #0d0e12 !important; border-right: 1px solid #1a1b22; }
[data-testid="stSidebar"] * { color: #c5c6c7 !important; }

/* ── Hide chrome ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Board cell buttons ── */
div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] button {
    width: 100% !important;
    aspect-ratio: 1 !important;
    min-height: 110px !important;
    font-size: 2.4rem !important;
    font-weight: 900 !important;
    font-family: 'Orbitron', sans-serif !important;
    border-radius: 16px !important;
    border: 2px solid #1f2030 !important;
    background: #13141d !important;
    color: #45a29e !important;
    transition: all 0.18s ease !important;
    line-height: 1 !important;
    padding: 0 !important;
}
div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] button:hover {
    border-color: #66fcf1 !important;
    background: #1a2535 !important;
    box-shadow: 0 0 22px rgba(102,252,241,0.3) !important;
    transform: scale(1.04) !important;
}

/* ── Global buttons ── */
.stButton > button {
    background: #1f8ef1 !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    padding: 0.55rem 1.2rem !important;
    transition: all 0.2s !important;
}
.stButton > button:hover { background: #1a7ad4 !important; box-shadow: 0 4px 18px rgba(31,142,241,0.45) !important; transform: translateY(-1px) !important; }

/* ── Custom components ── */
.page-title {
    font-family: 'Orbitron', sans-serif;
    font-size: 2.6rem; font-weight: 900;
    text-align: center;
    background: linear-gradient(90deg, #66fcf1, #1f8ef1, #c3073f);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    letter-spacing: 3px; margin-bottom: 0;
}
.page-sub { text-align: center; color: #4e535a; font-size: 0.85rem; margin-bottom: 1.5rem; letter-spacing: 1px; }

.status-box {
    border-radius: 12px; padding: 0.8rem 1.2rem;
    font-weight: 700; font-size: 1.05rem;
    text-align: center; margin-bottom: 1.2rem;
    border: 1px solid;
}
.s-ongoing { background: rgba(31,142,241,0.1); border-color: rgba(31,142,241,0.35); color: #1f8ef1; }
.s-xwins   { background: rgba(195,7,63,0.1);   border-color: rgba(195,7,63,0.4);   color: #c3073f; }
.s-owins   { background: rgba(102,252,241,0.1); border-color: rgba(102,252,241,0.4); color: #66fcf1; }
.s-draw    { background: rgba(255,189,0,0.08);  border-color: rgba(255,189,0,0.35); color: #ffbd00; }

.card {
    background: #13141d; border: 1px solid #1c1e2a;
    border-radius: 14px; padding: 1.1rem 1.3rem; margin: 0.5rem 0;
}
.row { display: flex; justify-content: space-between; align-items: center;
       padding: 5px 0; border-bottom: 1px solid #1c1e2a; font-size: 0.84rem; }
.row:last-child { border-bottom: none; }
.lbl { color: #4e535a; }
.val { color: #66fcf1; font-weight: 600; }

.score-chip {
    text-align: center; background: #13141d;
    border: 1px solid #1c1e2a; border-radius: 10px; padding: 0.6rem;
}
.chip-icon { font-size: 1.4rem; }
.chip-num  { font-size: 1.5rem; font-weight: 800; }
.chip-lbl  { font-size: 0.65rem; color: #4e535a; letter-spacing: 1px; text-transform: uppercase; }

.hist-item {
    font-size: 0.78rem; padding: 5px 10px;
    border-radius: 7px; margin: 3px 0; border-left: 3px solid;
}
.hist-x { background: rgba(195,7,63,0.08); border-color: #c3073f; color: #e07; }
.hist-o { background: rgba(102,252,241,0.06); border-color: #66fcf1; color: #66fcf1; }

.cell-display {
    border-radius: 16px; border: 2px solid;
    display: flex; align-items: center; justify-content: center;
    min-height: 110px; font-size: 2.6rem; font-weight: 900;
    font-family: 'Orbitron', sans-serif;
}
.cell-x { color: #c3073f; border-color: rgba(195,7,63,0.4); background: rgba(195,7,63,0.07); }
.cell-o { color: #66fcf1; border-color: rgba(102,252,241,0.4); background: rgba(102,252,241,0.06); }
.cell-win { border-color: #ffbd00 !important; background: rgba(255,189,0,0.1) !important;
            box-shadow: 0 0 20px rgba(255,189,0,0.25); }
</style>
""", unsafe_allow_html=True)

# ── Session Init ──────────────────────────────────────────────────────────────
def _init():
    defaults = {
        "game": GameSession(),
        "ai": AIEngine(O, "hard"),
        "scores": {"X": 0, "O": 0, "Draw": 0},
        "ai_stats": {},
        "total_nodes": 0,
        "game_count": 0,
        "celebrated": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()
game: GameSession = st.session_state.game
ai:   AIEngine    = st.session_state.ai

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ♟️ TacticAI")
    st.caption("Minimax · Alpha-Beta · Unbeatable")
    st.divider()

    mode = st.selectbox("Game Mode", ["hvai", "hvh", "aivai"],
        format_func=lambda x: {"hvai":"👤 Human vs 🤖 AI","hvh":"👤 vs 👤 Human","aivai":"🤖 AI vs 🤖 AI"}[x])

    if mode == "hvai":
        human_sym = st.radio("You play as", [X, O], horizontal=True)
        ai_sym = O if human_sym == X else X
    else:
        human_sym, ai_sym = X, O

    difficulty = st.select_slider("AI Difficulty", ["easy","medium","hard"],
        value="hard", disabled=(mode=="hvh"))
    diff_label = {"easy":"😊 Easy — Random","medium":"🤔 Medium — Mixed","hard":"💀 Hard — Unbeatable"}[difficulty]
    st.caption(diff_label)

    # Apply setting changes
    if ai.difficulty != difficulty or ai.ai_player != ai_sym:
        st.session_state.ai = AIEngine(ai_sym, difficulty)
        ai = st.session_state.ai
    if game.mode != mode or game.ai_player != ai_sym:
        game.mode, game.ai_player = mode, ai_sym
        game.reset()
        st.session_state.celebrated = False

    st.divider()
    st.markdown("**📊 Scoreboard**")
    c1, c2, c3 = st.columns(3)
    for col, icon, key, color in [(c1,"✗","X","#c3073f"),(c2,"═","Draw","#ffbd00"),(c3,"○","O","#66fcf1")]:
        col.markdown(
            f'<div class="score-chip"><div class="chip-icon" style="color:{color}">{icon}</div>'
            f'<div class="chip-num" style="color:{color}">{st.session_state.scores[key]}</div>'
            f'<div class="chip-lbl">{key}</div></div>', unsafe_allow_html=True)

    st.divider()
    if st.button("🔄 New Game", use_container_width=True):
        game.reset()
        st.session_state.ai_stats = {}
        st.session_state.celebrated = False
        st.rerun()

    st.divider()
    st.markdown('<div style="text-align:center;color:#2d3039;font-size:0.75rem;">CODSOFT AI · Task 2<br>'
                '<a href="https://github.com/SRIKRISH-S/CODSOFT" style="color:#1f8ef1;">github.com/SRIKRISH-S</a></div>',
                unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
def update_scores():
    s = game.status
    if s == "x_wins":   st.session_state.scores["X"] += 1; st.session_state.game_count += 1
    elif s == "o_wins": st.session_state.scores["O"] += 1; st.session_state.game_count += 1
    elif s == "draw":   st.session_state.scores["Draw"] += 1; st.session_state.game_count += 1

def do_ai_move():
    if game.status != "ongoing": return
    move, stats = ai.best_move(game.board)
    if move >= 0:
        game.make_move(move, stats)
        st.session_state.ai_stats = stats
        st.session_state.total_nodes += stats.get("nodes", 0)
        if game.status != "ongoing":
            update_scores()

# AI vs AI auto-play
if game.mode == "aivai" and game.status == "ongoing":
    time.sleep(0.35)
    do_ai_move()
    st.rerun()

# ── Page Header ───────────────────────────────────────────────────────────────
st.markdown('<p class="page-title">TACTIC AI</p>', unsafe_allow_html=True)
st.markdown('<p class="page-sub">MINIMAX · ALPHA-BETA PRUNING · GAME THEORY</p>', unsafe_allow_html=True)

# ── Status ────────────────────────────────────────────────────────────────────
winning_cells = set(game.winning_line) if game.winning_line else set()
is_ai_turn = game.mode == "hvai" and game.current_player == ai_sym

if game.status == "ongoing":
    msg = "🤖 AI is calculating..." if is_ai_turn else f"🎯 Player {'X' if game.current_player == X else 'O'} — your move"
    cls = "s-ongoing"
elif game.status == "x_wins":
    msg = "🏆 X Wins!" + (" — You beat the AI! Impressive!" if mode=="hvai" and human_sym==X else "")
    cls = "s-xwins"
elif game.status == "o_wins":
    msg = "🤖 O Wins! Unbeatable Minimax!" if mode=="hvai" and ai_sym==O else "🏆 O Wins!"
    cls = "s-owins"
else:
    msg = "🤝 Draw — Perfect play from both sides!"
    cls = "s-draw"

st.markdown(f'<div class="status-box {cls}">{msg}</div>', unsafe_allow_html=True)

# ── Layout ────────────────────────────────────────────────────────────────────
col_board, col_info = st.columns([1.1, 1], gap="large")

with col_board:
    if game.status == "ongoing":
        eval_txt = ai.evaluate_position(game.board)
        st.caption(f"💡 Position: {eval_txt}")

    # Board — 3 rows × 3 cols of Streamlit buttons + HTML cells
    for row_idx in range(3):
        cols = st.columns(3, gap="small")
        for col_idx in range(3):
            cell_idx = row_idx * 3 + col_idx
            val = game.board.cells[cell_idx]
            is_win = cell_idx in winning_cells

            with cols[col_idx]:
                if val == EMPTY and game.status == "ongoing" and not is_ai_turn:
                    # Clickable button
                    if st.button("", key=f"b{cell_idx}", use_container_width=True,
                                 help=f"Row {row_idx+1}, Column {col_idx+1}"):
                        game.make_move(cell_idx)
                        if game.status != "ongoing":
                            update_scores()
                        elif game.mode == "hvai" and game.current_player == ai_sym:
                            do_ai_move()
                        st.rerun()
                else:
                    # Static display cell
                    symbol = {"X": "X", "O": "O", " ": ""}[val]
                    if val == X:
                        cls2 = "cell-x" + (" cell-win" if is_win else "")
                    elif val == O:
                        cls2 = "cell-o" + (" cell-win" if is_win else "")
                    else:
                        cls2 = ""
                        symbol = "·"
                    st.markdown(
                        f'<div class="cell-display {cls2}">{symbol}</div>',
                        unsafe_allow_html=True)

    # Trigger AI when it's AI turn in hvai mode
    if game.mode == "hvai" and is_ai_turn and game.status == "ongoing":
        with st.spinner("AI thinking..."):
            do_ai_move()
        st.rerun()

    # Win celebration — only once
    if game.status in ("x_wins", "o_wins") and not st.session_state.celebrated:
        st.session_state.celebrated = True
        st.balloons()

with col_info:
    # ── AI Stats ──────────────────────────────────────────────────────────────
    stats = st.session_state.ai_stats
    st.markdown("**🤖 AI Engine**")
    rows = [
        ("Algorithm", "Minimax + α-β Pruning"),
        ("Difficulty", difficulty.capitalize()),
        ("Last Move Nodes", f"{stats.get('nodes', 0):,}"),
        ("Think Time", f"{stats.get('think_ms', 0):.1f} ms"),
        ("Minimax Score", str(stats.get('score', '—'))),
        ("Session Nodes", f"{st.session_state.total_nodes:,}"),
    ]
    rows_html = "".join(f'<div class="row"><span class="lbl">{l}</span><span class="val">{v}</span></div>' for l, v in rows)
    st.markdown(f'<div class="card">{rows_html}</div>', unsafe_allow_html=True)

    # ── Game Info ─────────────────────────────────────────────────────────────
    st.markdown("**📋 Session**")
    mode_str = {"hvai":"Human vs AI","hvh":"Human vs Human","aivai":"AI vs AI"}[mode]
    g_rows = [
        ("Mode", mode_str),
        ("Moves Made", str(game.move_count)),
        ("Games Played", str(st.session_state.game_count)),
        ("Status", game.status.replace("_"," ").title()),
    ]
    g_html = "".join(f'<div class="row"><span class="lbl">{l}</span><span class="val">{v}</span></div>' for l, v in g_rows)
    st.markdown(f'<div class="card">{g_html}</div>', unsafe_allow_html=True)

    # ── Move History ──────────────────────────────────────────────────────────
    if game.history:
        st.markdown("**📜 Move Log**")
        items = []
        for i, mv in enumerate(reversed(game.history[-9:])):
            n = len(game.history) - i
            cls3 = "hist-x" if mv.player == X else "hist-o"
            ai_tag = f" · {mv.ai_nodes:,} nodes" if mv.ai_nodes else ""
            items.append(f'<div class="hist-item {cls3}">#{n} {mv.player} → R{mv.row+1}C{mv.col+1}{ai_tag}</div>')
        st.markdown('<div class="card" style="padding:0.7rem">' + "".join(items) + "</div>",
                    unsafe_allow_html=True)

    with st.expander("🧠 Minimax Algorithm"):
        st.markdown("""
**Minimax** builds a complete game tree and picks the move that maximises AI's score  
while assuming the human always plays optimally.

| Outcome | Score |
|---------|-------|
| AI wins | `+10 − depth` |
| Human wins | `depth − 10` |
| Draw | `0` |

**Alpha-Beta Pruning** skips branches that are provably worse,  
cutting ~99% of nodes on the first move.
```
if β ≤ α: break   ← prune this branch
```
""")
