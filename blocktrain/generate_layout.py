"""
Bootstrap the canonical directory tree for the **tetris-trainer** project.

Run this from inside the project root, e.g.:

    C:\dev\tetris-trainer> python generate_layout.py

The script is idempotent: existing files are left untouched unless
``--force`` is supplied.
"""
from __future__ import annotations

import argparse
import datetime as _dt
from pathlib import Path
from textwrap import dedent

ROOT = Path.cwd()

# --------------------------------------------------------------------------- #
# Content blobs – minimal but functional code & stubs                         #
# --------------------------------------------------------------------------- #
TETRIS_ENV_CODE = dedent(
    '''
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
    '''
)

TRAIN_CODE = dedent(
    '''
    """
    Train PPO on the custom Tetris environment for 50 000 steps.
    """
    from __future__ import annotations

    import datetime as _dt
    from pathlib import Path

    import gymnasium as gym
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import TensorBoardCallback

    from env.tetris_env import TetrisEnv

    LOG_ROOT = Path("logs") / "tb"
    LOG_ROOT.mkdir(parents=True, exist_ok=True)


    def main() -> None:
        env = TetrisEnv()
        ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = LOG_ROOT / ts

        model = PPO("MlpPolicy", env, tensorboard_log=str(log_dir), verbose=1)
        model.learn(total_timesteps=50_000, callback=TensorBoardCallback(), progress_bar=True)
        model.save(log_dir / "ppo_tetris.zip")
        print("Model saved to", log_dir / "ppo_tetris.zip")


    if __name__ == "__main__":
        main()
    '''
)

TEST_CODE = dedent(
    '''
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
    '''
)

README_MD = dedent(
    '''
    # Tetris Trainer

    A minimal reinforcement-learning playground for learning to stack Tetrominoes
    with PPO & Stable-Baselines3.

    ```
    # quickstart
    python -m venv .env
    .env\\Scripts\\activate
    pip install -r requirements.txt
    pytest
    python scripts/train.py
    ```
    '''
)

REQUIREMENTS_TXT = dedent(
    '''
    gymnasium==0.29.1
    stable-baselines3==2.3.0
    pygame==2.6.1
    numpy==1.26.4
    torch==2.1.2
    '''
)

PYPROJECT_TOML = dedent(
    '''
    [build-system]
    requires = ["setuptools>=61.0"]
    build-backend = "setuptools.build_meta"

    [project]
    name = "tetris-trainer"
    version = "0.1.0"
    requires-python = ">=3.10"
    '''
)

# --------------------------------------------------------------------------- #
# Directory / file spec                                                       #
# --------------------------------------------------------------------------- #
DIRS = [
    "env",
    "scripts",
    "viewer",
    "tests",
    "logs/tb",
]
FILES = {
    "env/__init__.py": '"""Tetris custom Gymnasium env package."""\n',
    "env/tetris_env.py": TETRIS_ENV_CODE,
    "scripts/__init__.py": "",
    "scripts/train.py": TRAIN_CODE,
    "viewer/__init__.py": "",
    "tests/__init__.py": "",
    "tests/test_env.py": TEST_CODE,
    "README.md": README_MD,
    "requirements.txt": REQUIREMENTS_TXT,
    "pyproject.toml": PYPROJECT_TOML,
}


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def write_file(path: Path, content: str, force: bool) -> None:
    if path.exists() and not force:
        print(f"SKIP  {path} (exists)")
        return
    path.write_text(content, encoding="utf-8")
    print(f"WRITE {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate project skeleton.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite existing files",
    )
    args = parser.parse_args()

    # 1. create dirs
    for d in DIRS:
        p = ROOT / d
        p.mkdir(parents=True, exist_ok=True)
        # ensure log subdir keeps git history
        if "logs" in d and not (p / ".gitkeep").exists():
            (p / ".gitkeep").write_text("", encoding="utf-8")

    # 2. files
    for rel, content in FILES.items():
        write_file(ROOT / rel, content, args.force)

    print("\nDone – happy stacking!")


if __name__ == "__main__":
    main()
