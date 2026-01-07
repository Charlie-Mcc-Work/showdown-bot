# Pokemon Showdown RL Bot

A reinforcement learning bot that plays Pokemon Showdown Gen 9 Random Battles using self-play training.

## Project Status

**Current Phase**: Phase 1 - Foundation
**Last Updated**: 2026-01-07

### Progress Tracker

#### Phase 1: Foundation
- [x] Project planning and architecture design
- [x] Initialize repository with proper structure
- [x] Set up dependencies (PyTorch, poke-env, etc.)
- [x] Create CLAUDE.md with project documentation
- [x] Basic state encoder implementation
- [x] Neural network architecture (PolicyValueNetwork with attention)
- [x] Battle environment wrappers (RLPlayer, NeuralNetworkPlayer, MaxDamagePlayer)
- [x] Training/evaluation/play scripts
- [x] Push to GitHub

#### Phase 2: Core ML Pipeline
- [x] Implement neural network architecture
- [ ] Implement PPO algorithm
- [ ] Create experience buffer
- [ ] Basic training loop (agent vs random opponent)
- [ ] Logging and metrics (TensorBoard)

#### Phase 3: Self-Play System
- [ ] Implement self-play manager
- [ ] Opponent pool with historical checkpoints
- [ ] Elo rating tracking
- [ ] Parallel environment support
- [ ] Curriculum learning

#### Phase 4: Training & Optimization
- [ ] Hyperparameter tuning
- [ ] Extended training runs
- [ ] Action masking for illegal moves
- [ ] Auxiliary prediction heads

#### Phase 5: Evaluation & Ladder Play
- [ ] Evaluate against Showdown ladder
- [ ] Analyze failure modes
- [ ] Fine-tune based on performance

#### Phase 6: Browser Integration (Future)
- [ ] Screen capture / DOM parsing
- [ ] Real-time state extraction
- [ ] Move suggestion overlay

#### Phase 7: OU Expansion (Future)
- [ ] Team building module
- [ ] Metagame analysis
- [ ] Standard battle adaptation

---

## Architecture Overview

### Technology Stack
- **Language**: Python 3.11+
- **ML Framework**: PyTorch (ROCm for AMD GPU)
- **Showdown Integration**: poke-env
- **Training Hardware**: AMD RX 7900 XTX (24GB VRAM)

### RL Approach
- **Algorithm**: PPO (Proximal Policy Optimization)
- **Training**: Self-play against historical checkpoints
- **Model**: Transformer-based encoder with attention over Pokemon/moves

### Key Design Decisions
1. **PPO over DQN**: More stable for complex games, handles large action spaces better
2. **Self-play**: Avoids overfitting to specific opponent strategies
3. **Attention mechanism**: Handles variable Pokemon team compositions naturally
4. **Reward shaping**: Win/loss + HP differential + knockout bonuses for faster learning

---

## Project Structure

```
showdown-bot/
├── CLAUDE.md                    # This file - project documentation
├── pyproject.toml               # Dependencies and build config
├── .gitignore
│
├── src/showdown_bot/
│   ├── __init__.py
│   ├── config.py                # Training hyperparameters
│   ├── environment/             # Battle environment wrappers
│   ├── models/                  # Neural network architecture
│   ├── training/                # PPO, self-play, buffers
│   ├── evaluation/              # Elo tracking, metrics
│   └── browser/                 # Future browser integration
│
├── scripts/
│   ├── train.py                 # Main training script
│   ├── evaluate.py              # Ladder evaluation
│   └── play.py                  # Interactive play
│
├── data/
│   ├── pokemon_data/            # Static Pokemon data
│   └── checkpoints/             # Model checkpoints
│
└── tests/                       # Unit tests
```

---

## Quick Reference

### Running Training
```bash
# Install dependencies
pip install -e ".[dev]"

# Start training
python scripts/train.py

# Monitor with TensorBoard
tensorboard --logdir runs/
```

### Key Files
- `src/showdown_bot/config.py` - All hyperparameters
- `src/showdown_bot/models/network.py` - Neural network
- `src/showdown_bot/training/ppo.py` - PPO implementation
- `src/showdown_bot/environment/state_encoder.py` - State encoding

### Environment Variables
- `SHOWDOWN_SERVER` - Pokemon Showdown server URL (default: localhost:8000)
- `WANDB_API_KEY` - Optional Weights & Biases logging

---

## Notes for Development

### State Encoding
The state encoder converts battle state to tensors:
- Per-Pokemon: species, HP%, status, types, boosts, ability, item, moves
- Field: weather, terrain, hazards, screens, trick room, tailwind
- Handle partial observability for opponent's unrevealed info

### Action Space
- 4 move actions (indices 0-3)
- 5 switch actions (indices 4-8)
- Special: Mega/Z-Move/Tera/Dynamax when available
- Use action masking for illegal moves

### Reward Function
```python
reward = 0.0
if won: reward = 1.0
elif lost: reward = -1.0
else:
    reward += 0.01 * (our_hp - opp_hp)  # HP differential
    reward += 0.05 * (opp_fainted - our_fainted)  # KO bonus
```
