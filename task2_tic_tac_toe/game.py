"""
Tic-Tac-Toe Game Engine — Task 2 (CODSOFT AI Internship)
Implements unbeatable AI using Minimax with Alpha-Beta Pruning.
"""

from __future__ import annotations
import math
import time
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict


# ── Board Constants ───────────────────────────────────────────────────────────
EMPTY  = " "
X      = "X"
O      = "O"
SIZE   = 3
WINS   = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),   # rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),   # cols
    (0, 4, 8), (2, 4, 6),              # diagonals
]


# ── Board State ───────────────────────────────────────────────────────────────

class Board:
    """Represents a 3×3 Tic-Tac-Toe board."""

    def __init__(self, cells: Optional[List[str]] = None):
        self.cells: List[str] = cells.copy() if cells else [EMPTY] * 9

    def copy(self) -> "Board":
        return Board(self.cells)

    def make_move(self, idx: int, player: str) -> "Board":
        b = self.copy()
        b.cells[idx] = player
        return b

    def available_moves(self) -> List[int]:
        return [i for i, c in enumerate(self.cells) if c == EMPTY]

    def is_full(self) -> bool:
        return EMPTY not in self.cells

    def winner(self) -> Optional[str]:
        for a, b, c in WINS:
            if self.cells[a] != EMPTY and self.cells[a] == self.cells[b] == self.cells[c]:
                return self.cells[a]
        return None

    def is_terminal(self) -> bool:
        return self.winner() is not None or self.is_full()

    def winning_line(self) -> Optional[Tuple[int, int, int]]:
        for combo in WINS:
            a, b, c = combo
            if self.cells[a] != EMPTY and self.cells[a] == self.cells[b] == self.cells[c]:
                return combo
        return None

    def get_cell(self, row: int, col: int) -> str:
        return self.cells[row * SIZE + col]

    def __repr__(self) -> str:
        rows = []
        for r in range(SIZE):
            rows.append(" | ".join(self.cells[r*SIZE:(r+1)*SIZE]))
        return "\n---------\n".join(rows)


# ── AI Engine (Minimax + Alpha-Beta Pruning) ──────────────────────────────────

class AIEngine:
    """
    Unbeatable Tic-Tac-Toe AI using Minimax with Alpha-Beta Pruning.

    Difficulty modes:
      - easy   → random moves
      - medium → 50% optimal, 50% random
      - hard   → 100% Minimax (unbeatable)
    """

    def __init__(self, ai_player: str = O, difficulty: str = "hard"):
        self.ai_player  = ai_player
        self.human_player = X if ai_player == O else O
        self.difficulty = difficulty
        self.nodes_evaluated = 0
        self.last_think_time = 0.0

    def best_move(self, board: Board) -> Tuple[int, Dict]:
        """Return the best move index and stats dict."""
        import random

        self.nodes_evaluated = 0
        start = time.perf_counter()

        moves = board.available_moves()
        if not moves:
            return -1, {}

        if self.difficulty == "easy":
            chosen = random.choice(moves)
            score = 0
        elif self.difficulty == "medium":
            if random.random() < 0.5:
                chosen = random.choice(moves)
                score = 0
            else:
                chosen, score = self._minimax_root(board)
        else:  # hard
            chosen, score = self._minimax_root(board)

        self.last_think_time = (time.perf_counter() - start) * 1000

        stats = {
            "move": chosen,
            "score": score,
            "nodes": self.nodes_evaluated,
            "think_ms": round(self.last_think_time, 2),
            "difficulty": self.difficulty,
        }
        return chosen, stats

    def _minimax_root(self, board: Board) -> Tuple[int, int]:
        """Run minimax from root; return (best_idx, best_score)."""
        best_score = -math.inf
        best_move  = -1

        for idx in board.available_moves():
            new_board = board.make_move(idx, self.ai_player)
            score = self._minimax(new_board, False, -math.inf, math.inf, 0)
            if score > best_score:
                best_score = score
                best_move  = idx

        return best_move, best_score

    def _minimax(self, board: Board, is_maximizing: bool,
                 alpha: float, beta: float, depth: int) -> int:
        self.nodes_evaluated += 1

        winner = board.winner()
        if winner == self.ai_player:
            return 10 - depth
        if winner == self.human_player:
            return depth - 10
        if board.is_full():
            return 0

        if is_maximizing:
            best = -math.inf
            for idx in board.available_moves():
                child = board.make_move(idx, self.ai_player)
                val   = self._minimax(child, False, alpha, beta, depth + 1)
                best  = max(best, val)
                alpha = max(alpha, best)
                if beta <= alpha:
                    break   # β-cutoff
            return best
        else:
            best = math.inf
            for idx in board.available_moves():
                child = board.make_move(idx, self.human_player)
                val   = self._minimax(child, True, alpha, beta, depth + 1)
                best  = min(best, val)
                beta  = min(beta, best)
                if beta <= alpha:
                    break   # α-cutoff
            return best

    def evaluate_position(self, board: Board) -> str:
        """Return a human-readable assessment of the current position."""
        winner = board.winner()
        if winner:
            return f"{'AI' if winner == self.ai_player else 'Human'} wins!"
        if board.is_full():
            return "Draw — perfect play on both sides."

        score = self._minimax(board, True, -math.inf, math.inf, 0)
        if score > 0:
            return f"AI is winning (score: +{score})"
        elif score < 0:
            return f"Human is winning (score: {score})"
        else:
            return "Draw with optimal play from both sides."


# ── Game Session ──────────────────────────────────────────────────────────────

@dataclass
class MoveRecord:
    player:    str
    cell:      int
    row:       int
    col:       int
    board_snapshot: List[str]
    ai_nodes:  Optional[int]   = None
    ai_ms:     Optional[float] = None


@dataclass
class GameSession:
    """Manages a complete game session including history and stats."""

    mode:        str   = "hvai"     # hvai | hvh | aivai
    ai_player:   str   = O
    difficulty:  str   = "hard"
    current_player: str = X
    board:       Board = field(default_factory=Board)
    history:     List[MoveRecord] = field(default_factory=list)
    status:      str   = "ongoing"  # ongoing | x_wins | o_wins | draw
    winner:      Optional[str] = None
    winning_line: Optional[Tuple] = None

    def reset(self):
        self.board = Board()
        self.history = []
        self.status = "ongoing"
        self.winner = None
        self.winning_line = None
        self.current_player = X

    def make_move(self, idx: int, ai_stats: Optional[Dict] = None) -> bool:
        if self.board.cells[idx] != EMPTY or self.status != "ongoing":
            return False

        row, col = divmod(idx, SIZE)
        self.board = self.board.make_move(idx, self.current_player)

        self.history.append(MoveRecord(
            player=self.current_player,
            cell=idx,
            row=row,
            col=col,
            board_snapshot=self.board.cells.copy(),
            ai_nodes=ai_stats.get("nodes") if ai_stats else None,
            ai_ms=ai_stats.get("think_ms") if ai_stats else None,
        ))

        # Check terminal state
        w = self.board.winner()
        if w:
            self.status = "x_wins" if w == X else "o_wins"
            self.winner = w
            self.winning_line = self.board.winning_line()
        elif self.board.is_full():
            self.status = "draw"

        if self.status == "ongoing":
            self.current_player = O if self.current_player == X else X

        return True

    @property
    def is_ai_turn(self) -> bool:
        if self.mode == "hvh":
            return False
        if self.mode == "aivai":
            return True
        return self.current_player == self.ai_player

    @property
    def move_count(self) -> int:
        return len(self.history)
