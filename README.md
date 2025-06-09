
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
python -m scripts.train
# evaluate a saved model
python -m scripts.eval --model path/to/model.zip
# or use the management CLI
python -m scripts.manage train
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

Common tasks can also be run through a simple command line interface:

```bash
# create a virtual environment and install requirements
python -m scripts.manage setup

# start a new training run
python -m scripts.manage train --timesteps 200000

# resume training from an existing model
python -m scripts.manage resume --model logs/tb/20240101-120000/ppo_tetris.zip --timesteps 50000

# launch TensorBoard to monitor progress
python -m scripts.manage tensorboard
```

When using the `-m` flag, give the module name with dots rather than a
filesystem path. For instance run `python -m viewer.live_view`, not
`python -m ./viewer/live_view`.

## License

This project is released under the [MIT License](LICENSE).
