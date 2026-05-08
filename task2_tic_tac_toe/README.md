# 🎮 TacticAI — Tic-Tac-Toe with Minimax AI

> **CODSOFT AI Internship | Task 2**  
> Author: [SRIKRISH-S](https://github.com/SRIKRISH-S)

[![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red?logo=streamlit)](https://streamlit.io)
[![Algorithm](https://img.shields.io/badge/Algorithm-Minimax+α--β%20Pruning-purple)](https://en.wikipedia.org/wiki/Alpha%E2%80%93beta_pruning)

---

## 📋 Overview

**TacticAI** is an unbeatable Tic-Tac-Toe AI powered by **Minimax with Alpha-Beta Pruning**. The AI evaluates every possible future game state and always makes the optimal move.

## 🧠 AI Algorithm

### Minimax with Alpha-Beta Pruning

```
minimax(board, is_maximizing, α, β, depth):
    if terminal state:
        return score (±10 adjusted by depth)
    
    for each available move:
        score = minimax(next_board, !is_maximizing, α, β, depth+1)
        update α or β
        if β ≤ α: break   ← Alpha-Beta pruning (skip bad branches)
    
    return best_score
```

| Concept | Value |
|---------|-------|
| AI win score | `+10 − depth` (prefers faster wins) |
| Human win score | `−10 + depth` (prefers longer games) |
| Draw | `0` |
| Max nodes (9 moves) | 255,168 → pruned to ~500-2000 |

### Difficulty Levels

| Mode | Behavior |
|------|----------|
| 😊 Easy | 100% random moves |
| 🤔 Medium | 50% optimal, 50% random |
| 💀 Hard | 100% Minimax — **unbeatable** |

## 🚀 Setup & Run

```bash
cd CODSOFT/task2_tic_tac_toe
pip install -r requirements.txt
streamlit run app.py
```

## 📸 Features

- 🎮 **3 Game Modes**: Human vs AI, Human vs Human, AI vs AI
- 🧠 **Unbeatable AI**: Minimax + Alpha-Beta Pruning
- 📊 **Live AI Stats**: Nodes evaluated, think time, score
- 📜 **Move History**: Full game log with AI analytics
- 🏆 **Scoreboard**: Tracks wins/draws across sessions
- 💡 **Position Evaluator**: Real-time assessment of game state
- 🎨 **Premium Dark UI**: Orbitron font, glassmorphism cells, win animations

## 🗂️ Project Structure

```
task2_tic_tac_toe/
├── app.py          # Streamlit UI
├── game.py         # Minimax AI engine + game logic
├── requirements.txt
└── README.md
```
