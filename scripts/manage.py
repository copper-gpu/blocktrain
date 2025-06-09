"""CLI utility for training and managing Tetris RL agents."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
import datetime as dt

from stable_baselines3 import PPO

from env.tetris_env import TetrisEnv

LOG_ROOT = Path("logs") / "tb"


def setup_env(_: argparse.Namespace) -> None:
    """Create virtualenv, install requirements and prepare logs folder."""
    env_dir = Path(".env")
    if not env_dir.exists():
        subprocess.run([sys.executable, "-m", "venv", str(env_dir)], check=True)
    if os.name == "nt":
        pip = env_dir / "Scripts" / "pip"
    else:
        pip = env_dir / "bin" / "pip"
    subprocess.run([str(pip), "install", "-r", "requirements.txt"], check=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    print("Environment setup complete. Activate the virtualenv and start training.")


def train(args: argparse.Namespace) -> None:
    """Train a new PPO agent."""
    env = TetrisEnv()
    ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = LOG_ROOT / ts
    model = PPO(
        policy="MultiInputPolicy",
        env=env,
        tensorboard_log=str(log_dir),
        verbose=1,
    )
    model.learn(total_timesteps=args.timesteps, progress_bar=True)
    model_path = log_dir / "ppo_tetris.zip"
    model.save(model_path)
    print(f"Training finished – model saved to {model_path}")


def resume(args: argparse.Namespace) -> None:
    """Resume training from a saved model."""
    env = TetrisEnv()
    model = PPO.load(args.model, env=env)
    model.learn(total_timesteps=args.timesteps, progress_bar=True, reset_num_timesteps=False)
    model.save(args.model)
    print(f"Resumed training – model saved to {args.model}")


def tensorboard(_: argparse.Namespace) -> None:
    """Launch TensorBoard on the log directory."""
    subprocess.run(["tensorboard", "--logdir", str(LOG_ROOT)], check=True)


COMMANDS = {
    "setup": setup_env,
    "train": train,
    "resume": resume,
    "tensorboard": tensorboard,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Tetris Trainer management CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    setup_parser = subparsers.add_parser("setup", help="Prepare environment")
    setup_parser.set_defaults(func=setup_env)

    train_parser = subparsers.add_parser("train", help="Start a new training run")
    train_parser.add_argument("--timesteps", type=int, default=150_000)
    train_parser.set_defaults(func=train)

    resume_parser = subparsers.add_parser("resume", help="Resume training from a model")
    resume_parser.add_argument("--model", type=Path, required=True)
    resume_parser.add_argument("--timesteps", type=int, default=50_000)
    resume_parser.set_defaults(func=resume)

    tb_parser = subparsers.add_parser("tensorboard", help="Launch TensorBoard")
    tb_parser.set_defaults(func=tensorboard)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
