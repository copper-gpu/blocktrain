"""
Train PPO on the custom Tetris environment for 150 000 steps.

> python -m scripts.train
"""
from pathlib import Path

import datetime as _dt

from stable_baselines3 import PPO

from env.tetris_env import TetrisEnv

LOG_ROOT = Path("logs") / "tb"
LOG_ROOT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    # 1️⃣ create env
    env = TetrisEnv()

    # 2️⃣ logging dir
    ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = LOG_ROOT / ts

    # 3️⃣ instantiate PPO – **MultiInputPolicy** for Dict obs
    model = PPO(
        policy="MultiInputPolicy",
        env=env,
        tensorboard_log=str(log_dir),
        verbose=1,
    )

    # 4️⃣ learn
    model.learn(total_timesteps=150_000, progress_bar=True)

    # 5️⃣ save
    model_path = log_dir / "ppo_tetris.zip"
    model.save(model_path)
    print(f"Training finished – model saved to {model_path}")


if __name__ == "__main__":
    main()
