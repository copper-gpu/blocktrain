
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
pytest
# train an agent
python -m scripts.train
# evaluate a saved model
python scripts/eval.py path/to/model.zip
```

Run the training script as a module (with `-m`) so that package imports
resolve correctly.

To watch a game in real time, launch the Pygame viewer:

```bash
python viewer/live_view.py --model path/to/model.zip
```

## License

This project is released under the [MIT License](LICENSE).
