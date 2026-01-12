# Pokemon Showdown RL Bot

A reinforcement learning bot that plays Pokemon Showdown Gen 9 Random Battles using self-play training.

## Quick Start: Play Against the Bot

Play against the trained bot in your browser in 3 terminals:

### Terminal 1: Start Pokemon Showdown Server
```bash
cd ~/pokemon-showdown && node pokemon-showdown start --no-security
```

If you don't have the server installed:
```bash
git clone https://github.com/smogon/pokemon-showdown.git ~/pokemon-showdown
cd ~/pokemon-showdown && npm install
node pokemon-showdown start --no-security
```

### Terminal 2: Start the HTTP Server
```bash
cd ~/showdown-bot
./scripts/start_local_play.sh
```

### Terminal 3: Start the Bot
```bash
cd ~/showdown-bot
source .venv/bin/activate  # or .venv-rocm/bin/activate for ROCm
python scripts/play.py
```

### In Your Browser
1. Open: `http://localhost:8080/play.pokemonshowdown.com/testclient.html?~~localhost:8000`

2. Open browser console (F12 > Console) and login:
   ```javascript
   app.socket.send('|/trn YourName,0,')
   ```

3. Challenge the bot:
   ```javascript
   app.socket.send('|/challenge TrainedBot, gen9randombattle')
   ```

   Or use the UI: Click "Find a user" → type "TrainedBot" → "Challenge" → select "gen9randombattle"

---

## Features

- **PPO-based RL**: Uses Proximal Policy Optimization for stable training
- **Self-play Training**: Learns by playing against previous versions of itself
- **Attention Architecture**: Transformer-based model that handles variable Pokemon team compositions
- **poke-env Integration**: Uses the poke-env library for Showdown protocol handling
- **Browser Coaching**: Chrome extension that suggests optimal moves while you play

## Requirements

- Python 3.11+
- PyTorch 2.0+ (with ROCm for AMD GPUs or CUDA for NVIDIA)
- A running Pokemon Showdown server (for training/evaluation)

## Installation

```bash
# Clone the repository
git clone https://github.com/Charlie-Mcc-Work/showdown-bot.git
cd showdown-bot

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or: .venv\Scripts\activate  # Windows

# Install dependencies
pip install -e ".[dev]"
```

For AMD GPUs with ROCm, use the ROCm-specific PyTorch wheel:
```bash
pip install torch --index-url https://download.pytorch.org/whl/rocm6.0
```

## Setting Up Pokemon Showdown Server

Training requires a local Pokemon Showdown server:

```bash
# Clone Showdown (if not already done)
git clone https://github.com/smogon/pokemon-showdown.git ~/pokemon-showdown
cd ~/pokemon-showdown

# Install and start (with security disabled for local training)
npm install
node pokemon-showdown start --no-security
```

The server runs on `localhost:8000` by default.

---

## Training

### Quick Start

```bash
# Terminal 1: Start Pokemon Showdown server
cd ~/pokemon-showdown && node pokemon-showdown start --no-security

# Terminal 2: Run training
source .venv/bin/activate  # or .venv-rocm/bin/activate for ROCm
python scripts/train.py
```

### Training Commands

| Command | Description |
|---------|-------------|
| `python scripts/train.py` | Start fresh training (default: 10M steps, 8 envs) |
| `python scripts/train.py --resume` | Resume from latest checkpoint |
| `python scripts/train.py --resume path/to/checkpoint.pt` | Resume from specific checkpoint |
| `python scripts/train.py -t 100000` | Train for 100k steps |
| `python scripts/train.py --num-envs 4` | Use 4 parallel environments |
| `python scripts/train.py --no-self-play` | Train against random opponents only |

### Stopping Training

- **Ctrl+C**: Graceful shutdown - saves checkpoint and exits cleanly
- **Kill/crash**: `latest.pt` is updated on every periodic save, so resume will work

### Monitoring

```bash
# View training metrics in TensorBoard
tensorboard --logdir runs/
```

---

## Checkpoints

Checkpoints are saved to `data/checkpoints/`:

| File | Description |
|------|-------------|
| `latest.pt` | Most recent checkpoint (auto-updated on every save) |
| `best_model.pt` | Checkpoint with highest win rate achieved |
| `checkpoint_NNNNNN.pt` | Periodic checkpoints (every ~50k steps) |
| `emergency_checkpoint.pt` | Saved on errors before crash |

### Using Checkpoints

```bash
# Resume training from latest
python scripts/train.py --resume

# Resume from specific checkpoint
python scripts/train.py --resume data/checkpoints/checkpoint_3000312.pt

# Evaluate a checkpoint
python scripts/evaluate.py -c data/checkpoints/best_model.pt -n 100

# Play against a checkpoint
python scripts/play.py -c data/checkpoints/best_model.pt
```

### Inspecting Checkpoints

```python
import torch
checkpoint = torch.load("data/checkpoints/latest.pt", weights_only=False)
print(f"Timesteps: {checkpoint['stats']['total_timesteps']:,}")
print(f"Episodes: {checkpoint['stats']['total_episodes']:,}")
print(f"Best win rate: {checkpoint['stats']['best_win_rate']:.1%}")
if 'self_play' in checkpoint:
    print(f"Skill rating: {checkpoint['self_play']['agent_skill']:.0f}")
```

---

## Clearing Training History

### Full Reset (Start Fresh)

```bash
# Remove all checkpoints and logs
rm -rf data/checkpoints/*
rm -rf runs/*
rm -rf data/self_play/*

# Start fresh training
python scripts/train.py
```

### Keep Best Model Only

```bash
# Backup best model
cp data/checkpoints/best_model.pt /tmp/best_model_backup.pt

# Clear everything
rm -rf data/checkpoints/*
rm -rf runs/*
rm -rf data/self_play/*

# Restore best model (optional - to resume from best)
cp /tmp/best_model_backup.pt data/checkpoints/latest.pt
```

### Clear Opponent Pool Only

```bash
# Remove self-play opponent history (keeps main checkpoints)
rm -rf data/self_play/*
```

---

## Evaluation

```bash
# Evaluate against random opponent (100 games)
python scripts/evaluate.py -c data/checkpoints/best_model.pt -n 100

# Evaluate against max damage heuristic
python scripts/evaluate.py -c data/checkpoints/best_model.pt --opponent max_damage
```

---

## AI Coach (Browser Extension)

Get move suggestions while playing on play.pokemonshowdown.com:

```bash
# 1. Start coach server
python scripts/coach_server.py

# 2. Load extension in Chrome
# chrome://extensions > Developer mode > Load unpacked > select extension/

# 3. Play on https://play.pokemonshowdown.com
# The coach panel shows AI-recommended moves
```

---

## Hyperparameter Tuning

```bash
# Install Optuna
pip install optuna

# Run hyperparameter search (20 trials, 20k steps each)
python scripts/tune.py --trials 20 --timesteps 20000

# Results saved to data/tuning/best_config.py
```

---

## Project Structure

```
showdown-bot/
├── scripts/
│   ├── train.py          # Main training script
│   ├── evaluate.py       # Evaluation script
│   ├── play.py           # Interactive play
│   ├── coach_server.py   # AI coach Flask server
│   └── tune.py           # Hyperparameter tuning
├── src/showdown_bot/
│   ├── config.py         # Training hyperparameters
│   ├── environment/      # Battle wrappers & state encoding
│   ├── models/           # Neural network architecture
│   ├── training/         # PPO, self-play, trainer
│   └── evaluation/       # Skill rating tracking
├── extension/            # Chrome coaching extension
├── data/
│   ├── checkpoints/      # Model checkpoints
│   └── self_play/        # Opponent pool checkpoints
└── runs/                 # TensorBoard logs
```

---

## Current Status

See [CLAUDE.md](CLAUDE.md) for detailed development progress.

**Training Progress:**
- Model: ~1.5M parameters (Transformer-based)
- Current best: ~90% win rate against self-play opponents
- Skill rating: 15,000+ (internal rating system)

## License

MIT
