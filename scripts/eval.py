"""
Evaluate a trained PPO agent on TetrisEnv.

Usage:

    python scripts/eval.py --model logs/tb/20250607-120000/ppo_tetris.zip --episodes 100
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from stable_baselines3 import PPO

from env.tetris_env import TetrisEnv


def run_episode(model: PPO, render: bool = False) -> int:
    env = TetrisEnv()
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, info = env.step(action)
        if render:
            print("\033[H\033[J")  # rudimentary console clear
            print(obs["board"].astype(int))
    return info["lines_cleared"]  # total lines this episode


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PPO agent.")
    parser.add_argument("--model", required=True, type=Path, help="Path to PPO .zip")
    parser.add_argument(
        "--episodes", type=int, default=100, help="Number of evaluation episodes"
    )
    parser.add_argument("--render", action="store_true", help="ASCII board print-out")
    args = parser.parse_args()

    model = PPO.load(args.model)
    scores = np.array([run_episode(model, render=args.render) for _ in range(args.episodes)])

    print(f"Evaluated {args.episodes} episodes")
    print(f"Mean lines cleared   : {scores.mean():.2f}")
    print(f"Median lines cleared : {np.median(scores):.1f}")
    print(f"Max lines cleared    : {scores.max()}")

    # Simple success criterion
    solved = (scores.mean() >= 10)  # arbitrary threshold
    print("Solved?" , "YES" if solved else "no")


if __name__ == "__main__":
    main()
