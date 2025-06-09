
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

## License

This project is released under the [MIT License](LICENSE).
