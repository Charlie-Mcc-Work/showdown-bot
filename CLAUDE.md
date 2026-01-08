# Pokemon Showdown RL Bot

A reinforcement learning bot that plays Pokemon Showdown Gen 9 Random Battles using self-play training.

## Workflow Reminders

**IMPORTANT**: After completing each task, always tell the user what the next task is.

Current task queue:
1. ~~Hyperparameter tuning~~ (complete)
2. ~~Parallel environments~~ (complete - tested 4 envs at ~360 it/s)
3. ~~Websocket error fix~~ (complete - auto-retry with reconnection)
4. ~~Browser coaching extension~~ (COMPLETE)
5. Extended training & evaluation (next)

### Browser Coaching Extension (Complete)
Chrome extension that coaches you on play.pokemonshowdown.com:
- `extension/manifest.json` - Extension manifest (v3)
- `extension/content.js` - Reads battle state from PS, sends to local server
- `extension/styles.css` - Coach panel styling
- `extension/icon48.png` & `icon128.png` - Extension icons
- `scripts/coach_server.py` - Flask server with model inference
- `scripts/create_icons.py` - Icon generation script
- `tests/test_coach.py` - 23 comprehensive tests

**Features:**
- Full model integration with BrowserStateEncoder
- Fallback heuristic recommendations when model unavailable
- Status move prioritization, setup move detection, priority move awareness
- Health/recommend API endpoints

**To use:**
1. Start server: `uv run python scripts/coach_server.py`
2. Load extension in Chrome: chrome://extensions > Developer mode > Load unpacked > select extension/
3. Go to https://play.pokemonshowdown.com and start a battle
4. Coach panel shows AI-powered move recommendations

### Known Issues (2026-01-07) - RESOLVED
1. **Training websocket error**: FIXED - Added automatic retry with connection recovery.
   The trainer now catches websocket errors, recreates players, and resumes training.
   See `collect_rollout_with_retry()` in `trainer.py`.

2. **Browser auth**: WORKAROUND - Use console login command.
   The testclient's auth can be bypassed using the browser console:
   ```javascript
   app.socket.send('|/trn YourName,0,')
   ```
   See `browser/local-client.html` for full setup instructions and a bookmarklet.

## Project Status

**Current Phase**: Phase 4 - Training & Optimization (In Progress)
**Last Updated**: 2026-01-08

### Recent Changes
- **Browser coaching extension complete**: Full model integration with BrowserStateEncoder
- **Extension icons**: Auto-generated Pokeball-style icons (scripts/create_icons.py)
- **Comprehensive tests**: 23 new tests for coaching server (tests/test_coach.py)
- **Coach server improvements**: Model inference, heuristic fallback, action masking
- **Dependencies updated**: Added pytest, flask-cors to dev dependencies
- **Websocket error fix**: Added automatic retry with connection recovery in trainer
- **Training note**: Training REQUIRES local Pokemon Showdown server running on port 8000
- Implemented Optuna-based hyperparameter tuning system (scripts/tune.py)
- Added parallel environment support (--num-envs flag, ~360 it/s with 4 envs)
- Trainer now accepts injectable TrainingConfig for tuning
- Integrated self-play training into main trainer
- Added graceful Ctrl+C shutdown with checkpoint saving
- Added `--resume` flag for easy training continuation
- Verified GPU training with parallel environments

### Progress Tracker

#### Phase 1: Foundation - COMPLETE
- [x] Project planning and architecture design
- [x] Initialize repository with proper structure
- [x] Set up dependencies (PyTorch ROCm, poke-env)
- [x] Create CLAUDE.md with project documentation
- [x] Basic state encoder implementation
- [x] Neural network architecture (PolicyValueNetwork with attention)
- [x] Battle environment wrappers (RLPlayer, NeuralNetworkPlayer, MaxDamagePlayer)
- [x] Training/evaluation/play scripts
- [x] Push to GitHub

#### Phase 2: Core ML Pipeline - COMPLETE
- [x] Implement neural network architecture
- [x] Implement PPO algorithm (with clipped objective, GAE)
- [x] Create experience/rollout buffer
- [x] Basic training loop (agent vs random opponent)
- [x] TensorBoard logging and metrics
- [x] Checkpoint saving/loading
- [x] Graceful shutdown with signal handlers
- [x] Resume from checkpoint support

#### Phase 3: Self-Play System - COMPLETE
- [x] Implement self-play manager
- [x] Opponent pool with historical checkpoints
- [x] Elo rating tracking
- [x] Comprehensive test suite
- [x] Self-play integrated into trainer
- [x] Parallel environment support (moved to Phase 4, now complete)
- [ ] Curriculum learning (Future)

#### Phase 4: Training & Optimization - IN PROGRESS
- [x] Hyperparameter tuning system (Optuna)
- [x] Parallel environments for faster training
- [ ] Extended training runs
- [x] Action masking for illegal moves (already implemented)
- [ ] Auxiliary prediction heads (Future)

#### Phase 5: Evaluation & Ladder Play
- [ ] Evaluate against Showdown ladder
- [ ] Analyze failure modes
- [ ] Fine-tune based on performance

#### Phase 6: Browser Integration - IN PROGRESS
- [x] Chrome extension for play.pokemonshowdown.com
- [x] Real-time state extraction from PS battle client
- [x] Move suggestion overlay (AI Coach panel)
- [x] Model inference server (Flask API)
- [ ] Screen capture integration (future)

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
# Using the ROCm virtual environment
source .venv-rocm/bin/activate

# Start fresh training (default 10M steps, 8 parallel envs)
python scripts/train.py

# Start short training run (20k steps)
python scripts/train.py -t 20000

# Run with 4 parallel environments (faster training)
python scripts/train.py --num-envs 4

# Run with single environment (for debugging)
python scripts/train.py --num-envs 1

# Resume from latest checkpoint
python scripts/train.py --resume

# Resume from specific checkpoint
python scripts/train.py --resume data/checkpoints/best_model.pt

# Train without self-play (random opponents only)
python scripts/train.py --no-self-play

# Monitor with TensorBoard
tensorboard --logdir runs/
```

### Evaluation
```bash
# Evaluate trained model against baselines
python scripts/evaluate.py -c data/checkpoints/best_model.pt -n 100
```

### Requirements
Pokemon Showdown server must be running:
```bash
cd ~/pokemon-showdown && node pokemon-showdown start --no-security
```

### Key Files
- `src/showdown_bot/config.py` - All hyperparameters
- `src/showdown_bot/models/network.py` - Neural network
- `src/showdown_bot/training/ppo.py` - PPO implementation
- `src/showdown_bot/training/self_play.py` - Self-play manager and opponent pool
- `src/showdown_bot/evaluation/elo.py` - Elo rating system
- `src/showdown_bot/environment/state_encoder.py` - State encoding

### Browser Play (Working)
Play against the trained bot in a web browser.

**Quick Start:**
```bash
# Run the all-in-one setup script:
./scripts/start_local_play.sh
# Then open: http://localhost:8081/local-client.html
```

**Manual Setup:**
```bash
# 1. Start the Pokemon Showdown server
cd ~/pokemon-showdown
node pokemon-showdown start --no-security

# 2. Start the HTTP server for the client
cd ~/pokemon-showdown-client
npx http-server -p 8080 -c-1

# 3. Start the bot
cd ~/Documents/Programming/Charlie-Mcc-Work/showdown-bot
.venv-rocm/bin/python scripts/play.py

# 4. Open browser to:
# http://localhost:8080/play.pokemonshowdown.com/testclient.html?~~localhost:8000

# 5. Login via browser console (F12 > Console):
# app.socket.send('|/trn YourName,0,')

# 6. Challenge "TrainedBot" to a gen9randombattle!
```

**Note**: The testclient requires a console command to login because it tries to verify
authentication signatures. The `/trn username,0,` command bypasses this for local servers.
See `browser/local-client.html` for a bookmarklet that makes this easier.

**Ultimate Goal**: Bot watches user play and suggests optimal moves (Phase 6).

### Running Tests
```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_network.py

# Run with coverage
pytest tests/ --cov=showdown_bot
```

### Environment Variables
- `SHOWDOWN_SERVER` - Pokemon Showdown server URL (default: localhost:8000)
- `WANDB_API_KEY` - Optional Weights & Biases logging

### Hyperparameter Tuning
```bash
# Install tuning dependencies
pip install optuna

# Run hyperparameter search (20 trials, 20k steps each)
python scripts/tune.py --trials 20 --timesteps 20000

# Continue an existing study
python scripts/tune.py --load-if-exists --trials 10

# Results saved to data/tuning/best_config.py
```

**Tunable Hyperparameters:**
| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| learning_rate | 3e-4 | 1e-5 to 1e-3 | Adam learning rate |
| gamma | 0.99 | 0.95 to 0.999 | Discount factor |
| gae_lambda | 0.95 | 0.9 to 0.99 | GAE lambda |
| clip_epsilon | 0.2 | 0.1 to 0.3 | PPO clipping |
| entropy_coef | 0.01 | 0.001 to 0.1 | Exploration bonus |
| value_coef | 0.5 | 0.25 to 1.0 | Value loss weight |
| num_epochs | 4 | 2 to 10 | PPO epochs per update |
| batch_size | 64 | 32/64/128/256 | Minibatch size |
| rollout_steps | 2048 | 512 to 4096 | Steps per rollout |

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
