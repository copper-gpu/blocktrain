
"""
Pytest sanity checks for TetrisEnv.
"""
import numpy as np
from env.tetris_env import (
    TetrisEnv,
    PIECES,
    ROTATE,
    HARD_DROP,
)


def test_reset_lines_zero():
    env = TetrisEnv(seed=42)
    _, info = env.reset()
    assert info["lines_cleared"] == 0


def test_step_board_shape():
    env = TetrisEnv(seed=123)
    obs, _ = env.reset()
    obs, *_ = env.step(env.action_space.sample())
    assert obs["board"].shape == (20, 10)


def test_rotate_without_collision():
    env = TetrisEnv(seed=0)
    env.reset()
    env.piece = PIECES[0].copy()
    env.current_id = 0
    env.pos = (0, 3)
    original = env.piece.copy()
    env.step(ROTATE)
    assert np.array_equal(env.piece, np.rot90(original))


def test_line_clearing():
    env = TetrisEnv(seed=0)
    env.reset()
    env.board[-1, 1:] = 1
    env.piece = PIECES[7].copy()
    env.current_id = 7
    env.pos = (0, 0)
    obs, reward, terminated, truncated, info = env.step(HARD_DROP)
    assert reward == 1.0
    assert not terminated
    assert info["lines_cleared"] == 1
    assert env.board[-1].sum() == 0


def test_game_over_when_spawn_blocked():
    env = TetrisEnv(seed=0)
    env.reset()
    env.board[0:4, 3:7] = 1
    _, _, terminated, _, _ = env.step(HARD_DROP)
    assert terminated
