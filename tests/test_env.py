
"""
Pytest sanity checks for TetrisEnv.
"""
import numpy as np
from env.tetris_env import TetrisEnv


def test_reset_lines_zero():
    env = TetrisEnv(seed=42)
    _, info = env.reset()
    assert info["lines_cleared"] == 0


def test_step_board_shape():
    env = TetrisEnv(seed=123)
    obs, _ = env.reset()
    obs, *_ = env.step(env.action_space.sample())
    assert obs["board"].shape == (20, 10)
