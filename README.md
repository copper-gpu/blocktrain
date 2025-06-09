
# Tetris Trainer

A minimal reinforcement-learning playground for learning to stack Tetrominoes
with PPO & Stable-Baselines3.

```
# quickstart
python -m venv .env
# Windows
.env\Scripts\activate
# POSIX
source .env/bin/activate
pip install -r requirements.txt
# requirements include stable-baselines3 extras and tensorboard for training logs
pytest
# train an agent
# train an agent
python scripts/train.py --timesteps 200000 --line-reward 2.0 --tetris-bonus 20
# evaluate a saved model
python scripts/eval.py --model path/to/model.zip
# or use the management script
python scripts/manage.py train
```

On Windows you can alternatively run the provided `quickstart_win.bat`
script to automatically set up the virtual environment and install all
dependencies. Make sure to run it from a `cmd.exe` prompt (or via
`cmd /k quickstart_win.bat`) so that the virtual environment remains
active. If you invoke it from PowerShell, activate the environment
afterwards with `./.env/Scripts/activate` before running the training
command.

Run the evaluation or viewer scripts directly:

```bash
python scripts/eval.py --model path/to/model.zip
python viewer/live_view.py --model path/to/model.zip
```

## Management script

Common tasks can also be run through a simple command line interface. The
`scripts/manage.py` helper provides an interactive menu when run without
arguments:

```bash
python scripts/manage.py
```

Subcommands can be called directly:

```bash
# create a virtual environment and install requirements
python scripts/manage.py setup

# start a new training run
python scripts/manage.py train --timesteps 200000 --line-reward 2.0 --tetris-bonus 20

# resume training from an existing model
python scripts/manage.py resume --model logs/tb/20240101-120000/ppo_tetris.zip --timesteps 50000 --line-reward 2.0 --tetris-bonus 20

# launch TensorBoard to monitor progress
python scripts/manage.py tensorboard
```

Both the training script and management helper expose common PPO parameters
such as `--learning-rate`, `--gamma`, `--line-reward` and the optional
`--tetris-bonus` which controls how valuable clearing four lines at once is to
the agent. Adjust these flags to experiment with different behaviours.

## License

This project is released under the [MIT License](LICENSE).
