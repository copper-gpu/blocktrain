
"""
Minimal but functional Tetris Gymnasium environment.
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

BOARD_HEIGHT = 20
BOARD_WIDTH = 10

PIECES = {
    0: np.array([[1, 1, 1, 1]], dtype=np.int8),                  # I
    1: np.array([[1, 1], [1, 1]], dtype=np.int8),                # O
    2: np.array([[0, 1, 0], [1, 1, 1]], dtype=np.int8),          # T
    3: np.array([[0, 1, 1], [1, 1, 0]], dtype=np.int8),          # S
    4: np.array([[1, 1, 0], [0, 1, 1]], dtype=np.int8),          # Z
    5: np.array([[1, 0, 0], [1, 1, 1]], dtype=np.int8),          # J
    6: np.array([[0, 0, 1], [1, 1, 1]], dtype=np.int8),          # L
    7: np.array([[1]], dtype=np.int8),                           # Dot (filler)
}

NO_OP, LEFT, RIGHT, ROTATE, SOFT_DROP, HARD_DROP = range(6)
Action = int


class TetrisEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, seed: int | None = None):
        super().__init__()
        self.observation_space = spaces.Dict(
            {
                "board": spaces.Box(
                    low=0,
                    high=1,
                    shape=(BOARD_HEIGHT, BOARD_WIDTH),
                    dtype=np.int8,
                ),
                "current": spaces.Discrete(8),
            }
        )
        self.action_space = spaces.Discrete(6)

        self._rng = np.random.default_rng(seed)
        self.board = np.zeros((BOARD_HEIGHT, BOARD_WIDTH), dtype=np.int8)
        self.current_id = 0
        self.piece = PIECES[0]
        self.pos = (0, 3)
        self._lines_total = 0

    # ------------------------------------------------------------------ #
    # Gymnasium API                                                      #
    # ------------------------------------------------------------------ #
    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        self.board.fill(0)
        self._lines_total = 0
        self._spawn_new_piece()
        return self._get_obs(), {"lines_cleared": 0}

    def step(self, action: Action):
        if action == LEFT:
            self._move(dx=-1)
        elif action == RIGHT:
            self._move(dx=1)
        elif action == ROTATE:
            self._rotate()
        elif action == SOFT_DROP:
            self._move(dy=1)
        elif action == HARD_DROP:
            self._hard_drop()

        moved = self._move(dy=1)
        if not moved:
            self._lock_piece()
            lines = self._clear_lines()
            self._lines_total += lines
            reward = float(lines)
            terminated = not self._spawn_new_piece()
        else:
            reward, terminated = 0.0, False

        obs = self._get_obs()
        return obs, reward, terminated, False, {"lines_cleared": self._lines_total}

    # ----------------------- helpers ---------------------------------- #
    def _get_obs(self):
        brd = self.board.copy()
        r, c = self.pos
        h, w = self.piece.shape
        brd[r : r + h, c : c + w] |= self.piece
        return {"board": brd, "current": np.array(self.current_id, dtype=np.int8)}

    def _spawn_new_piece(self):
        self.current_id = int(self._rng.integers(0, 7))
        self.piece = PIECES[self.current_id]
        self.pos = (0, (BOARD_WIDTH - self.piece.shape[1]) // 2)
        return not self._collides(self.pos)

    def _move(self, dx=0, dy=0):
        nr, nc = self.pos[0] + dy, self.pos[1] + dx
        if not self._collides((nr, nc)):
            self.pos = (nr, nc)
            return True
        return False

    def _rotate(self):
        rot = np.rot90(self.piece)
        if not self._collides(self.pos, rot):
            self.piece = rot

    def _hard_drop(self):
        while self._move(dy=1):
            pass

    def _collides(self, pos, piece=None):
        if piece is None:
            piece = self.piece
        r, c = pos
        h, w = piece.shape
        if c < 0 or c + w > BOARD_WIDTH or r + h > BOARD_HEIGHT:
            return True
        return np.any(self.board[r : r + h, c : c + w] & piece)

    def _lock_piece(self):
        r, c = self.pos
        h, w = self.piece.shape
        self.board[r : r + h, c : c + w] |= self.piece

    def _clear_lines(self):
        full = np.all(self.board == 1, axis=1)
        n = int(full.sum())
        if n:
            self.board = np.vstack(
                [np.zeros((n, BOARD_WIDTH), dtype=np.int8), self.board[~full]]
            )
        return n
