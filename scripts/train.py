"""
Train PPO on the custom Tetris environment for 150 000 steps.

Example:

    python scripts/train.py --timesteps 200000 --line-reward 2.0
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import datetime as _dt

from stable_baselines3 import PPO

from env.tetris_env import TetrisEnv

LOG_ROOT = Path("logs") / "tb"
LOG_ROOT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO on Tetris")
    parser.add_argument("--timesteps", type=int, default=150_000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--line-reward", type=float, default=1.0)
    args = parser.parse_args()

    # 1️⃣ create env
    env = TetrisEnv(line_reward=args.line_reward)

    # 2️⃣ logging dir
    ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = LOG_ROOT / ts

    # 3️⃣ instantiate PPO – **MultiInputPolicy** for Dict obs
    model = PPO(
        policy="MultiInputPolicy",
        env=env,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        tensorboard_log=str(log_dir),
        verbose=1,
    )

    # 4️⃣ learn
    model.learn(total_timesteps=args.timesteps, progress_bar=True)

    # 5️⃣ save
    model_path = log_dir / "ppo_tetris.zip"
    model.save(model_path)
    print(f"Training finished – model saved to {model_path}")


if __name__ == "__main__":
    main()
