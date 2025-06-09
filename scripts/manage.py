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
    env = TetrisEnv(line_reward=args.line_reward)
    ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = LOG_ROOT / ts
    model = PPO(
        policy="MultiInputPolicy",
        env=env,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        tensorboard_log=str(log_dir),
        verbose=1,
    )
    model.learn(total_timesteps=args.timesteps, progress_bar=True)
    model_path = log_dir / "ppo_tetris.zip"
    model.save(model_path)
    print(f"Training finished – model saved to {model_path}")


def resume(args: argparse.Namespace) -> None:
    """Resume training from a saved model."""
    env = TetrisEnv(line_reward=args.line_reward)
    model = PPO.load(
        args.model,
        env=env,
        custom_objects={"learning_rate": args.learning_rate, "gamma": args.gamma},
    )
    model.learn(total_timesteps=args.timesteps, progress_bar=True, reset_num_timesteps=False)
    model.save(args.model)
    print(f"Resumed training – model saved to {args.model}")


def tensorboard(_: argparse.Namespace) -> None:
    """Launch TensorBoard on the log directory."""
    subprocess.run(["tensorboard", "--logdir", str(LOG_ROOT)], check=True)


# map command names to handler and description
COMMANDS = {
    "setup": (setup_env, "Prepare environment"),
    "train": (train, "Start a new training run"),
    "resume": (resume, "Resume training from a model"),
    "tensorboard": (tensorboard, "Launch TensorBoard"),
}


def interactive_menu() -> None:
    """Present a simple numbered menu for common tasks."""
    options = list(COMMANDS.items())
    print("Tetris Trainer – choose an option:")
    for i, (_, (_, desc)) in enumerate(options, start=1):
        print(f"{i}. {desc}")
    print(f"{len(options) + 1}. Exit")

    choice = input("Enter number: ").strip()
    if not choice.isdigit():
        print("Invalid selection")
        return

    idx = int(choice) - 1
    if idx == len(options):
        return
    if not (0 <= idx < len(options)):
        print("Invalid selection")
        return

    cmd, (func, _) = options[idx]
    if cmd == "train":
        ts = input("Total timesteps [150000]: ").strip()
        lr_in = input("Learning rate [0.0003]: ").strip()
        g_in = input("Gamma [0.99]: ").strip()
        lrw_in = input("Line reward [1.0]: ").strip()
        timesteps = int(ts) if ts else 150_000
        lr = float(lr_in) if lr_in else 3e-4
        gamma = float(g_in) if g_in else 0.99
        line_reward = float(lrw_in) if lrw_in else 1.0
        func(
            argparse.Namespace(
                timesteps=timesteps,
                learning_rate=lr,
                gamma=gamma,
                line_reward=line_reward,
            )
        )
    elif cmd == "resume":
        model = input("Model path: ").strip()
        ts = input("Total timesteps [50000]: ").strip()
        lr_in = input("Learning rate [0.0003]: ").strip()
        g_in = input("Gamma [0.99]: ").strip()
        lrw_in = input("Line reward [1.0]: ").strip()
        timesteps = int(ts) if ts else 50_000
        lr = float(lr_in) if lr_in else 3e-4
        gamma = float(g_in) if g_in else 0.99
        line_reward = float(lrw_in) if lrw_in else 1.0
        func(
            argparse.Namespace(
                model=Path(model),
                timesteps=timesteps,
                learning_rate=lr,
                gamma=gamma,
                line_reward=line_reward,
            )
        )
    else:
        func(argparse.Namespace())


def main() -> None:
    parser = argparse.ArgumentParser(description="Tetris Trainer management CLI")
    subparsers = parser.add_subparsers(dest="command", required=False)

    setup_parser = subparsers.add_parser("setup", help="Prepare environment")
    setup_parser.set_defaults(func=setup_env)

    train_parser = subparsers.add_parser("train", help="Start a new training run")
    train_parser.add_argument("--timesteps", type=int, default=150_000)
    train_parser.add_argument("--learning-rate", type=float, default=3e-4)
    train_parser.add_argument("--gamma", type=float, default=0.99)
    train_parser.add_argument("--line-reward", type=float, default=1.0)
    train_parser.set_defaults(func=train)

    resume_parser = subparsers.add_parser("resume", help="Resume training from a model")
    resume_parser.add_argument("--model", type=Path, required=True)
    resume_parser.add_argument("--timesteps", type=int, default=50_000)
    resume_parser.add_argument("--learning-rate", type=float, default=3e-4)
    resume_parser.add_argument("--gamma", type=float, default=0.99)
    resume_parser.add_argument("--line-reward", type=float, default=1.0)
    resume_parser.set_defaults(func=resume)

    tb_parser = subparsers.add_parser("tensorboard", help="Launch TensorBoard")
    tb_parser.set_defaults(func=tensorboard)

    args = parser.parse_args()
    if args.command is None:
        interactive_menu()
    else:
        args.func(args)


if __name__ == "__main__":
    main()
