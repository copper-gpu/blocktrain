
# Tetris Trainer

A minimal reinforcement-learning playground for learning to stack Tetrominoes
with PPO & Stable-Baselines3.

```
# quickstart
codex/remove-sys.path-import-in-train.py
python -m venv .env
.env\Scripts\activate
pip install -r requirements.txt
pytest
python -m scripts.train
```

Run the training script as a module (with `-m`) so that package imports
resolve correctly.

codex/update-readme.md-with-setup-instructions
python -m venv .env
# Windows
.env\Scripts\activate
# POSIX
source .env/bin/activate
pip install -r requirements.txt
pytest
# train an agent
python scripts/train.py
# evaluate a saved model
python scripts/eval.py path/to/model.zip
```

To watch a game in real time, launch the Pygame viewer:

```bash
python viewer/live_view.py --model path/to/model.zip
```

python -m venv .env
.env\Scripts\activate
pip install -r requirements.txt
pytest
python scripts/train.py
```

## License

This project is released under the [MIT License](LICENSE).
main
main
