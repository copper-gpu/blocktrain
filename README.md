
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
python -m scripts.train --timesteps 200000 --line-reward 2.0
# evaluate a saved model
python -m scripts.eval --model path/to/model.zip
# or use the management CLI
tetris-trainer train
```

On Windows you can alternatively run the provided `quickstart_win.bat`
script to automatically set up the virtual environment and install all
dependencies. Make sure to run it from a `cmd.exe` prompt (or via
`cmd /k quickstart_win.bat`) so that the virtual environment remains
active. If you invoke it from PowerShell, activate the environment
afterwards with `./.env/Scripts/activate` before running the training
command.

Run the training script as a module (with `-m`) so that package imports
resolve correctly. The same applies to the evaluation and viewer
scripts:

```bash
python -m scripts.eval --model path/to/model.zip
python -m viewer.live_view --model path/to/model.zip
```

## Management CLI

Common tasks can also be run through a simple command line interface. When the
project is installed (e.g. via `pip install -e .`), the `tetris-trainer` command
provides direct access to these utilities. Running `tetris-trainer` without any
arguments will show an interactive menu of available actions:

```bash
tetris-trainer
```

You can also call the subcommands directly:

```bash
# create a virtual environment and install requirements
tetris-trainer setup

# start a new training run
tetris-trainer train --timesteps 200000 --line-reward 2.0

# resume training from an existing model
tetris-trainer resume --model logs/tb/20240101-120000/ppo_tetris.zip --timesteps 50000 --line-reward 2.0

# launch TensorBoard to monitor progress
tetris-trainer tensorboard
```

Both the training script and CLI expose common PPO parameters such as
`--learning-rate`, `--gamma` and a custom `--line-reward` that controls how
valuable clearing a line is to the agent. Adjust these flags to experiment with
different behaviours.

When using the `-m` flag, give the module name with dots rather than a
filesystem path. For instance run `python -m viewer.live_view`, not
`python -m ./viewer/live_view`.

## License

This project is released under the [MIT License](LICENSE).
