"""
Real-time Pygame viewer for the TetrisEnv.

Run:

    python viewer/live_view.py                 # random agent
    python viewer/live_view.py --model <zip>   # play with trained PPO

Press ESC or close the window to quit.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import pygame
import numpy as np

from env.tetris_env import TetrisEnv

# Optional Stable-Baselines3 import (only if a model is requested)
try:
    from stable_baselines3 import PPO
except ImportError:  # pragma: no cover
    PPO = None  # type: ignore


CELL = 24                    # square size in pixels
GRID_COLOR = (40, 40, 40)
FILLED_COLOR = (0, 200, 255)
BG_COLOR = (18, 18, 18)
FPS = 30


def draw_board(screen: pygame.Surface, board: np.ndarray) -> None:
    h, w = board.shape
    for r in range(h):
        for c in range(w):
            rect = pygame.Rect(c * CELL, r * CELL, CELL, CELL)
            pygame.draw.rect(screen, GRID_COLOR, rect, width=1)
            if board[r, c]:
                pygame.draw.rect(screen, FILLED_COLOR, rect.inflate(-2, -2))


def play(model_path: Path | None = None) -> None:
    env = TetrisEnv()
    model = PPO.load(model_path) if (model_path and PPO) else None

    pygame.init()
    surface = pygame.display.set_mode(
        (env.board.shape[1] * CELL, env.board.shape[0] * CELL)
    )
    pygame.display.set_caption("Tetris RL Viewer")
    clock = pygame.time.Clock()

    obs, _ = env.reset()
    running = True
    while running:
        # ----- handle events ---------------------------------------------
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (
                event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
            ):
                running = False

        # ----- agent action ----------------------------------------------
        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = env.action_space.sample()

        obs, _, terminated, _, _ = env.step(action)
        if terminated:
            obs, _ = env.reset()

        # ----- render ----------------------------------------------------
        surface.fill(BG_COLOR)
        draw_board(surface, obs["board"])
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live viewer for TetrisEnv.")
    parser.add_argument(
        "--model",
        type=Path,
        help="Path to a trained PPO .zip (optional).",
    )
    args = parser.parse_args()
    play(model_path=args.model)
