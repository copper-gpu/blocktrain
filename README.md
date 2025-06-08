
# Tetris Trainer

A minimal reinforcement-learning playground for learning to stack Tetrominoes
with PPO & Stable-Baselines3.

```
# quickstart
python -m venv .env
.env\Scripts\activate
pip install -r requirements.txt
pytest
python -m scripts.train
```

Run the training script as a module (with `-m`) so that package imports
resolve correctly.
