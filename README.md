# Pokemon Showdown RL Bot

A reinforcement learning bot that plays Pokemon Showdown Gen 9 Random Battles using self-play training.

## Features

- **PPO-based RL**: Uses Proximal Policy Optimization for stable training
- **Self-play Training**: Learns by playing against previous versions of itself
- **Attention Architecture**: Transformer-based model that handles variable Pokemon team compositions
- **poke-env Integration**: Uses the poke-env library for Showdown protocol handling

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
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -e ".[dev]"
```

## Setting Up Pokemon Showdown

Training requires a local Pokemon Showdown server:

```bash
# Clone Showdown
git clone https://github.com/smogon/pokemon-showdown.git
cd pokemon-showdown

# Install and start (with security disabled for local training)
npm install
node pokemon-showdown start --no-security
```

The server will run on `localhost:8000` by default.

## Usage

### Training

```bash
python scripts/train.py
```

### Evaluation

```bash
# Evaluate against random opponent
python scripts/evaluate.py

# Evaluate with a trained checkpoint
python scripts/evaluate.py --checkpoint data/checkpoints/model.pt
```

### Play Against the Bot

```bash
python scripts/play.py --checkpoint data/checkpoints/model.pt
```

## Project Status

This project is under active development. See [CLAUDE.md](CLAUDE.md) for detailed progress tracking.

### Current Phase: Foundation
- [x] Project structure and dependencies
- [x] State encoder for battle representation
- [x] Neural network architecture
- [x] Basic training script skeleton
- [ ] PPO implementation
- [ ] Self-play training loop

## Architecture

```
State Encoding:
  - Pokemon: HP, types, status, boosts, moves
  - Field: weather, terrain, hazards, screens

Neural Network:
  - Team encoder (Transformer attention)
  - Cross-attention between active Pokemon
  - Policy head (action probabilities)
  - Value head (state value)

Training:
  - PPO algorithm
  - Self-play against historical checkpoints
  - Reward shaping (HP differential + knockouts)
```

## License

MIT
