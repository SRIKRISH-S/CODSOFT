"""
Task 2 — Tic-Tac-Toe AI (Streamlit)
CODSOFT AI Internship | Author: SRIKRISH-S
"""

import streamlit as st
import time
from game import Board, AIEngine, GameSession, X, O, EMPTY, SIZE, WINS

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TacticAI — Tic-Tac-Toe",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700;900&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: radial-gradient(ellipse at top, #0d0221 0%, #020010 60%, #000000 100%);
    color: #e2e8f0;
}

/* ── Hero ── */
.hero {
    text-align: center;
    padding: 2rem 0 1.5rem;
}
.hero-title {
    font-family: 'Orbitron', sans-serif;
    font-size: 3.2rem;
    font-weight: 900;
    background: linear-gradient(135deg, #a855f7, #6366f1, #38bdf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    letter-spacing: 4px;
    text-shadow: none;
}
.hero-sub { color: #94a3b8; font-size: 1rem; margin-top: 0.4rem; }

/* ── Status Banner ── */
.status-banner {
    border-radius: 14px;
    padding: 0.9rem 1.4rem;
    font-size: 1.1rem;
    font-weight: 700;
    text-align: center;
    margin-bottom: 1rem;
    letter-spacing: 0.5px;
}
.status-ongoing  { background: rgba(99,102,241,0.15); border: 1px solid rgba(99,102,241,0.4); color: #a5b4fc; }
.status-x_wins   { background: rgba(239,68,68,0.15);  border: 1px solid rgba(239,68,68,0.5);  color: #fca5a5; }
.status-o_wins   { background: rgba(16,185,129,0.15); border: 1px solid rgba(16,185,129,0.5); color: #6ee7b7; }
.status-draw     { background: rgba(245,158,11,0.15); border: 1px solid rgba(245,158,11,0.5); color: #fcd34d; }

/* ── Board ── */
.board-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 10px;
    max-width: 380px;
    margin: 0 auto;
}
.cell {
    aspect-ratio: 1;
    border-radius: 16px;
    border: 2px solid rgba(99,102,241,0.25);
    background: rgba(255,255,255,0.03);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 3.2rem;
    font-weight: 900;
    cursor: pointer;
    transition: all 0.2s ease;
    min-height: 100px;
    backdrop-filter: blur(10px);
}
.cell:hover { border-color: rgba(168,85,247,0.7); background: rgba(168,85,247,0.1); transform: scale(1.03); }
.cell.X { color: #f87171; border-color: rgba(248,113,113,0.5); background: rgba(239,68,68,0.08); }
.cell.O { color: #34d399; border-color: rgba(52,211,153,0.5); background: rgba(16,185,129,0.08); }
.cell.winning { border-color: #fbbf24 !important; background: rgba(251,191,36,0.15) !important; animation: pulse-win 0.8s ease-in-out infinite alternate; }
.cell.empty-avail:hover { border-color: rgba(168,85,247,0.8); }
@keyframes pulse-win { from { box-shadow: 0 0 15px rgba(251,191,36,0.3); } to { box-shadow: 0 0 30px rgba(251,191,36,0.7); } }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: rgba(5,0,20,0.98) !important;
    border-right: 1px solid rgba(99,102,241,0.2);
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #6366f1) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important; font-weight: 600 !important;
    transition: all 0.2s !important;
}
.stButton > button:hover { transform: translateY(-2px) !important; box-shadow: 0 6px 20px rgba(124,58,237,0.5) !important; }

/* ── Info Cards ── */
.info-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 12px;
    padding: 0.9rem 1.1rem;
    margin: 0.4rem 0;
}
.stat-row { display: flex; justify-content: space-between; color: #94a3b8; font-size: 0.85rem; padding: 3px 0; }
.stat-val { color: #a5b4fc; font-weight: 600; }

/* ── Move History ── */
.move-item {
    padding: 6px 12px;
    border-radius: 8px;
    margin: 3px 0;
    font-size: 0.78rem;
    border-left: 3px solid;
}
.move-x { background: rgba(239,68,68,0.08); border-color: #f87171; color: #fca5a5; }
.move-o { background: rgba(16,185,129,0.08); border-color: #34d399; color: #6ee7b7; }

.metric-mini {
    background: rgba(99,102,241,0.08);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 10px;
    padding: 0.7rem;
    text-align: center;
}

#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Session State Init ────────────────────────────────────────────────────────
def init_state():
    if "game" not in st.session_state:
        st.session_state.game = GameSession()
    if "ai" not in st.session_state:
        st.session_state.ai = AIEngine(O, "hard")
    if "scores" not in st.session_state:
        st.session_state.scores = {"X": 0, "O": 0, "Draw": 0}
    if "last_ai_stats" not in st.session_state:
        st.session_state.last_ai_stats = {}
    if "game_count" not in st.session_state:
        st.session_state.game_count = 0
    if "total_nodes" not in st.session_state:
        st.session_state.total_nodes = 0

init_state()

game: GameSession = st.session_state.game
ai:   AIEngine    = st.session_state.ai


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎮 TacticAI Settings")
    st.markdown("---")

    mode = st.selectbox(
        "Game Mode",
        ["hvai", "hvh", "aivai"],
        format_func=lambda x: {
            "hvai":  "👤 Human vs 🤖 AI",
            "hvh":   "👤 Human vs 👤 Human",
            "aivai": "🤖 AI vs 🤖 AI",
        }[x],
    )
    
    if mode == "hvai":
        human_sym = st.radio("You play as", [X, O], horizontal=True)
        ai_sym = O if human_sym == X else X
    else:
        human_sym = X
        ai_sym    = O

    difficulty = st.selectbox(
        "AI Difficulty",
        ["easy", "medium", "hard"],
        index=2,
        format_func=lambda x: {
            "easy":   "😊 Easy (Random)",
            "medium": "🤔 Medium (Mixed)",
            "hard":   "💀 Hard (Unbeatable Minimax)",
        }[x],
        disabled=(mode == "hvh"),
    )

    st.markdown("---")
    st.markdown("### 📊 Scoreboard")
    col_x, col_d, col_o = st.columns(3)
    col_x.markdown(
        f'<div class="metric-mini"><div style="font-size:1.3rem;color:#f87171">✗</div>'
        f'<div style="font-size:1.4rem;font-weight:700;color:#fca5a5">{st.session_state.scores["X"]}</div>'
        f'<div style="font-size:0.7rem;color:#94a3b8">X Wins</div></div>',
        unsafe_allow_html=True,
    )
    col_d.markdown(
        f'<div class="metric-mini"><div style="font-size:1.3rem;color:#fbbf24">═</div>'
        f'<div style="font-size:1.4rem;font-weight:700;color:#fcd34d">{st.session_state.scores["Draw"]}</div>'
        f'<div style="font-size:0.7rem;color:#94a3b8">Draws</div></div>',
        unsafe_allow_html=True,
    )
    col_o.markdown(
        f'<div class="metric-mini"><div style="font-size:1.3rem;color:#34d399">○</div>'
        f'<div style="font-size:1.4rem;font-weight:700;color:#6ee7b7">{st.session_state.scores["O"]}</div>'
        f'<div style="font-size:0.7rem;color:#94a3b8">O Wins</div></div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # Reconfigure AI if settings changed
    if ai.difficulty != difficulty or ai.ai_player != ai_sym:
        st.session_state.ai = AIEngine(ai_sym, difficulty)
        ai = st.session_state.ai

    if game.mode != mode or game.ai_player != ai_sym:
        game.mode = mode
        game.ai_player = ai_sym
        game.reset()

    new_game_btn = st.button("🔄 New Game", use_container_width=True)
    if new_game_btn:
        game.reset()
        st.session_state.last_ai_stats = {}
        st.rerun()

    st.markdown("---")
    st.markdown(
        '<div style="color:#6366f1;font-size:0.78rem;text-align:center;">'
        "CODSOFT AI Internship<br>Task 2 — Tic-Tac-Toe AI<br>"
        '<a href="https://github.com/SRIKRISH-S/CODSOFT" style="color:#a5b4fc;">GitHub: SRIKRISH-S</a>'
        "</div>",
        unsafe_allow_html=True,
    )

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-title">⬛ TACTIC AI ⬛</div>
    <div class="hero-sub">Minimax · Alpha-Beta Pruning · Unbeatable Intelligence</div>
</div>
""", unsafe_allow_html=True)

# ── AI Move Trigger ───────────────────────────────────────────────────────────
def trigger_ai_move():
    if game.status != "ongoing":
        return
    move, stats = ai.best_move(game.board)
    if move >= 0:
        game.make_move(move, stats)
        st.session_state.last_ai_stats = stats
        st.session_state.total_nodes += stats.get("nodes", 0)
        if game.status != "ongoing":
            update_scores()

def update_scores():
    if game.status == "x_wins":
        st.session_state.scores["X"] += 1
        st.session_state.game_count += 1
    elif game.status == "o_wins":
        st.session_state.scores["O"] += 1
        st.session_state.game_count += 1
    elif game.status == "draw":
        st.session_state.scores["Draw"] += 1
        st.session_state.game_count += 1

# AI vs AI: auto-play
if game.mode == "aivai" and game.status == "ongoing":
    time.sleep(0.4)
    trigger_ai_move()
    st.rerun()

# ── Status Banner ─────────────────────────────────────────────────────────────
STATUS_MSGS = {
    "ongoing": lambda: f"{'🤖 AI is thinking...' if game.is_ai_turn and mode == 'hvai' else f'🎯 {game.current_player}\'s turn'}",
    "x_wins":  lambda: "🎉 X Wins! Excellent play!",
    "o_wins":  lambda: "🤖 O Wins! AI is unbeatable!" if mode == "hvai" and ai_sym == O else "🎉 O Wins!",
    "draw":    lambda: "🤝 It's a Draw — Perfect game!",
}
status_msg = STATUS_MSGS[game.status]()
st.markdown(
    f'<div class="status-banner status-{game.status}">{status_msg}</div>',
    unsafe_allow_html=True,
)

# ── Main Layout ───────────────────────────────────────────────────────────────
col_board, col_info = st.columns([1.2, 1])

with col_board:
    # Board evaluation hint
    if game.status == "ongoing":
        eval_txt = ai.evaluate_position(game.board)
        st.markdown(
            f'<div style="text-align:center;color:#6366f1;font-size:0.8rem;margin-bottom:8px;">💡 {eval_txt}</div>',
            unsafe_allow_html=True,
        )

    # Render board as 3×3 Streamlit button grid
    winning_cells = set(game.winning_line) if game.winning_line else set()
    
    board_rows = [st.columns(3) for _ in range(3)]
    
    for idx in range(9):
        row, col = divmod(idx, 3)
        cell_val  = game.board.cells[idx]
        is_win    = idx in winning_cells
        
        with board_rows[row][col]:
            if cell_val == EMPTY and game.status == "ongoing" and not game.is_ai_turn:
                # Clickable empty cell
                if st.button(
                    "·",
                    key=f"cell_{idx}",
                    use_container_width=True,
                    help=f"Play at position {row+1},{col+1}",
                ):
                    game.make_move(idx)
                    if game.status != "ongoing":
                        update_scores()
                    elif game.is_ai_turn:
                        trigger_ai_move()
                    st.rerun()
            else:
                # Display cell value
                symbol  = {"X": "✗", "O": "○", " ": "·"}[cell_val]
                color   = {"X": "#f87171", "O": "#34d399", " ": "#334155"}[cell_val]
                bg      = "rgba(251,191,36,0.12)" if is_win else "rgba(255,255,255,0.02)"
                border  = "2px solid #fbbf24" if is_win else "1px solid rgba(99,102,241,0.2)"
                st.markdown(
                    f'<div style="aspect-ratio:1;border-radius:16px;border:{border};'
                    f'background:{bg};display:flex;align-items:center;justify-content:center;'
                    f'font-size:3rem;font-weight:900;color:{color};min-height:95px;'
                    f'{"animation:pulse-win 0.8s ease-in-out infinite alternate;" if is_win else ""}">'
                    f'{symbol}</div>',
                    unsafe_allow_html=True,
                )

    # Trigger AI move if it's AI's turn (for hvai mode at game start)
    if game.mode == "hvai" and game.is_ai_turn and game.status == "ongoing":
        with st.spinner("🤖 AI is calculating..."):
            trigger_ai_move()
        st.rerun()

with col_info:
    # ── AI Stats ──────────────────────────────────────────────────────────────
    st.markdown("### 🤖 AI Engine Stats")
    stats = st.session_state.last_ai_stats
    
    st.markdown(
        f"""
        <div class="info-card">
            <div class="stat-row"><span>Algorithm</span><span class="stat-val">Minimax + α-β Pruning</span></div>
            <div class="stat-row"><span>Difficulty</span><span class="stat-val">{difficulty.capitalize()}</span></div>
            <div class="stat-row"><span>Nodes Evaluated</span><span class="stat-val">{stats.get('nodes', '—'):,}</span></div>
            <div class="stat-row"><span>Think Time</span><span class="stat-val">{stats.get('think_ms', '—')} ms</span></div>
            <div class="stat-row"><span>Best Score</span><span class="stat-val">{stats.get('score', '—')}</span></div>
            <div class="stat-row"><span>Total Nodes (session)</span><span class="stat-val">{st.session_state.total_nodes:,}</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Game Info ─────────────────────────────────────────────────────────────
    st.markdown("### 📋 Game Info")
    st.markdown(
        f"""
        <div class="info-card">
            <div class="stat-row"><span>Mode</span><span class="stat-val">{'Human vs AI' if mode=='hvai' else 'Human vs Human' if mode=='hvh' else 'AI vs AI'}</span></div>
            <div class="stat-row"><span>Move #</span><span class="stat-val">{game.move_count}</span></div>
            <div class="stat-row"><span>Games Played</span><span class="stat-val">{st.session_state.game_count}</span></div>
            <div class="stat-row"><span>Status</span><span class="stat-val">{game.status.replace('_', ' ').title()}</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Move History ──────────────────────────────────────────────────────────
    if game.history:
        st.markdown("### 📜 Move History")
        for move in reversed(game.history[-8:]):
            ai_note = ""
            if move.ai_nodes:
                ai_note = f" · 🤖 {move.ai_nodes:,} nodes"
            st.markdown(
                f'<div class="move-{"x" if move.player == X else "o"}">'
                f'Move {game.history.index(move)+1}: <strong>{move.player}</strong> → '
                f'Row {move.row+1}, Col {move.col+1}{ai_note}</div>',
                unsafe_allow_html=True,
            )

    # ── Algorithm Explainer ───────────────────────────────────────────────────
    with st.expander("🧠 How Minimax Works"):
        st.markdown("""
**Minimax** explores all possible future game states to find the optimal move.

- **Maximizer (AI)**: Tries to get the highest score (+10 for win)
- **Minimizer (Human)**: Tries to get the lowest score (-10 for human win)  
- **Draw**: Score = 0
- **Depth penalty**: Earlier wins score higher (10-depth)

**Alpha-Beta Pruning** cuts branches that can't affect the result, making the search ~6× faster.

```
minimax(board, is_max, α, β):
  if terminal: return score
  for each move:
    score = minimax(next_board, !is_max, α, β)
    if β ≤ α: break  ← pruning!
```
        """)

# ── Win Celebration ───────────────────────────────────────────────────────────
if game.status == "x_wins" or game.status == "o_wins":
    st.balloons()
