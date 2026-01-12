# Pokemon Showdown RL Bot

A reinforcement learning bot that plays Pokemon Showdown Gen 9 Random Battles using self-play training.

## Workflow Reminders

**IMPORTANT**: After completing each task, always tell the user what the next task is.

Current task queue:
1. ~~Hyperparameter tuning~~ (complete)
2. ~~Parallel environments~~ (complete - tested 4 envs at ~360 it/s)
3. ~~Websocket error fix~~ (complete - auto-retry with reconnection)
4. ~~Browser coaching extension~~ (COMPLETE)
5. ~~OU Module Foundation~~ (complete - player classes, training script)
6. ~~OU Experience Collection~~ (COMPLETE)
7. ~~OU Self-Play System~~ (COMPLETE)
8. ~~OU Team Preview Integration~~ (COMPLETE - heuristic selector)
9. ~~OU Teambuilder Training Loop~~ (COMPLETE - infrastructure ready)
10. ~~OU Joint Training~~ (COMPLETE - placeholder, awaits TeamGenerator)

**Next steps** for OU completion:
- Implement full TeamGenerator (species selection, move selection, etc.)
- Implement Team → tensor encoding for TeamEvaluator
- Extended training runs with player mode
- Evaluation against Showdown ladder

### OU Development Plan

The OU system trains **player** and **teambuilder** together in a feedback loop:
- Player plays battles with generated teams
- Battle results (win/loss, performance) feed back to teambuilder
- Teambuilder learns which team compositions lead to wins
- Player improves at playing the teams teambuilder generates

#### Step 6: Experience Collection (COMPLETE)
**Goal**: Collect state/action/reward transitions during OU battles for PPO updates.

Implemented:
- [x] `OUTrainablePlayer` class hooks into `choose_move()` to record (state, action, log_prob, value)
- [x] `calculate_ou_reward()` function for HP differential + KO bonus rewards
- [x] Terminal rewards in `_battle_finished_callback()`
- [x] `train_ou.py` collects experiences via `player.get_experiences()` and adds to `OUExperienceBuffer`
- [x] PPO update runs when buffer has enough data

Files modified:
- `ou/player/ou_player.py` - Added `OUTrainablePlayer`, `calculate_ou_reward()`
- `scripts/train_ou.py` - Uses `OUTrainablePlayer`, collects experiences after battles

#### Step 7: OU Self-Play System (COMPLETE)
**Goal**: Train against historical checkpoints instead of just Random/MaxDamage.

Implemented:
- [x] `OUOpponentPool` - Store historical OU player checkpoints
- [x] `OUHistoricalPlayer` - Load and play with historical checkpoints
- [x] `OUSelfPlayManager` - High-level manager with skill-matched sampling
- [x] Elo rating updates after games
- [x] `--no-self-play` CLI flag for disabling self-play
- [x] TensorBoard logging of self-play stats

Files created/modified:
- `ou/training/self_play.py` - OU self-play system (OUOpponentPool, OUSelfPlayManager)
- `scripts/train_ou.py` - Integrated self-play opponents

#### Step 8: Team Preview Integration (COMPLETE)
**Goal**: Use heuristic lead selector to choose leads strategically.

Implemented:
- [x] `_get_team_preview_order()` helper function using `HeuristicLeadSelector`
- [x] Override `teampreview()` in all OU player classes
- [x] Lead selection based on role priorities (hazard setter, blocker, pivot)
- [x] Neural network selector exists but not yet trained

Files modified:
- `ou/player/ou_player.py` - Added `teampreview()` overrides
- `ou/training/self_play.py` - Added `teampreview()` to `OUHistoricalPlayer`

#### Step 9: Teambuilder Training Loop (COMPLETE - Infrastructure)
**Goal**: Train team generator from battle performance feedback.

Implemented:
- [x] `TeambuilderTrainingManager` class in `scripts/train_ou.py`
- [x] `--mode teambuilder` CLI flag for teambuilder training
- [x] Loads frozen player checkpoint for team evaluation
- [x] Generates teams and plays N games per team
- [x] Records battle outcomes in `TeamOutcomeBuffer`
- [x] Checkpoint saving for teambuilder

**Note**: The underlying `TeamGenerator` produces placeholder teams. The training
infrastructure is ready for when the generator is fully implemented.

Files modified:
- `scripts/train_ou.py` - Added `TeambuilderTrainingManager`, `--mode` argument

#### Step 10: Joint Training (COMPLETE - Placeholder)
**Goal**: Alternate between player and teambuilder updates.

Implemented:
- [x] `--mode joint` CLI flag
- [x] Currently falls back to player training (teambuilder not fully functional)
- [x] Architecture documented in ou/README.md

**Future work** (when TeamGenerator is complete):
1. Generate K teams from teambuilder
2. Play M games per team, collecting player experience
3. Update player with PPO
4. Update teambuilder based on team win rates
5. Add player checkpoint to opponent pool
6. Repeat

### Browser Coaching Extension (Complete)
Chrome extension that coaches you on play.pokemonshowdown.com:
- `extension/manifest.json` - Extension manifest (v3 with `world: "MAIN"`)
- `extension/page-script.js` - Runs in page context, extracts battle state from `app.curRoom`
- `extension/content.js` - Handles UI and API communication (isolated world)
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

**Technical notes:**
- Uses `world: "MAIN"` in manifest to access page JavaScript variables
- Battle request is at `app.curRoom.request`, not `app.curRoom.battle.request`
- Two-script architecture: page-script.js (MAIN world) + content.js (isolated world)

**To use:**
1. Start server: `uv run python scripts/coach_server.py`
2. Load extension in Chrome: chrome://extensions > Developer mode > Load unpacked > select extension/
3. Go to https://play.pokemonshowdown.com and start a battle
4. Coach panel shows AI-powered move recommendations

### Known Issues - RESOLVED
1. **Training websocket error**: FIXED - Added automatic retry with connection recovery.
   The trainer now catches websocket errors, recreates players, and resumes training.
   See `collect_rollout_with_retry()` in `trainer.py`.

2. **Memory leak (2026-01-12)**: FIXED - HistoricalPlayer models are now properly released
   after each training loop. The cleanup process:
   - Stops websocket listeners and waits 0.5s for messages to drain
   - Sets `player.model = None` and `player.state_encoder = None` to break references
   - Calls `gc.collect()` to immediately free memory
   - Defensive check in `choose_move()` returns random move if model is None (logs warning)
   See `_cleanup_players()` in `trainer.py`.

3. **File descriptor leak (2026-01-08)**: FIXED - Opponents are now cleaned up after each
   training loop iteration. Previously, HistoricalPlayer websocket connections accumulated
   until hitting the system limit (~1024 open files).

4. **Browser auth**: WORKAROUND - Use console login command.
   The testclient's auth can be bypassed using the browser console:
   ```javascript
   app.socket.send('|/trn YourName,0,')
   ```
   See `browser/local-client.html` for full setup instructions and a bookmarklet.

## Project Status

**Current Phase**: Phase 4 - Training & Optimization (In Progress)
**Last Updated**: 2026-01-12

### Recent Changes
- **OU Training Integration (2026-01-12)**: Added complete training pipeline for Gen 9 OU
  - `scripts/train_ou.py` - Training script with sample teams, evaluation, checkpointing
  - `OURLPlayer` / `OUNeuralNetworkPlayer` - poke-env Player wrappers
  - `action_to_battle_order()` - Converts actions to battle orders (13 actions: 4 moves, 4 tera moves, 5 switches)
  - Run with: `python scripts/train_ou.py`
- **Memory leak fix (2026-01-12)**: Fixed HistoricalPlayer models not being garbage collected
  - Models now set to None after cleanup to break reference cycles
  - Added 0.5s delay for websocket message draining
  - Added gc.collect() after opponent cleanup
  - Defensive check in choose_move() with warning logging
  - Added comprehensive memory management tests (tests/test_memory.py)
- **Quick start guide (2026-01-12)**: Added step-by-step browser play instructions to README
- **Graceful shutdown fix (2026-01-12)**: Suppressed noisy websocket close messages during Ctrl+C
- **Empty buffer handling (2026-01-12)**: PPO update now handles empty buffer during shutdown
- **OU Module (2026-01-08)**: Added Gen 9 OU expansion module with teambuilding and player components
  - See `src/showdown_bot/ou/README.md` for full architecture
  - Shared embeddings for Pokemon, moves, items, abilities
  - Autoregressive team generator with evaluator
  - OU-specific state encoder with opponent prediction
  - Team preview lead selection
- **File descriptor leak fix (2026-01-08)**: Fixed opponents not being cleaned up after training loops
- **Browser coaching extension complete**: Full model integration with BrowserStateEncoder
- **Websocket error fix**: Added automatic retry with connection recovery in trainer
- Implemented Optuna-based hyperparameter tuning system (scripts/tune.py)
- Added parallel environment support (--num-envs flag, ~360 it/s with 4 envs)
- Added graceful Ctrl+C shutdown with checkpoint saving

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
- [x] Skill rating tracking (Elo algorithm)
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

#### Phase 7: OU Expansion - TEAM GENERATION COMPLETE
- [x] Module architecture design (see `src/showdown_bot/ou/README.md`)
- [x] Shared embeddings (Pokemon, moves, items, abilities)
- [x] Team representation and serialization
- [x] Full TeamGenerator implementation:
  - [x] UsageBasedGenerator (generates teams from Smogon usage stats)
  - [x] Species selection weighted by usage rate + teammate correlations
  - [x] Move/item/ability selection from common sets
  - [x] Nature/EV spreads from usage data
  - [x] Proper name formatting (format_name helper)
- [x] Team evaluator network (placeholder architecture)
- [x] OU state encoder with opponent prediction
- [x] Team preview lead selector (heuristic)
- [x] OURLPlayer / OUNeuralNetworkPlayer (poke-env integration)
- [x] OUTrainablePlayer with experience collection
- [x] Training script (`scripts/train_ou.py`) with player, teambuilder, joint modes
- [x] Self-play for OU (OUSelfPlayManager, OUOpponentPool)
- [x] Teambuilder training loop infrastructure
- [x] Usage stats integration (UsageStatsLoader downloads from Smogon)
- [ ] Team → tensor encoding for evaluator
- [ ] Metagame analysis

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
│   ├── evaluation/              # Skill rating tracking, metrics
│   ├── browser/                 # Future browser integration
│   └── ou/                      # Gen 9 OU expansion module
│       ├── README.md            # OU module documentation
│       ├── shared/              # Shared embeddings and encoders
│       │   ├── embeddings.py    # Pokemon/move/item embeddings
│       │   ├── encoders.py      # Team and Pokemon encoders
│       │   └── data_loader.py   # Data loading utilities
│       ├── teambuilder/         # Autoregressive team generator
│       │   ├── team_repr.py     # Team representation
│       │   ├── generator.py     # Team generation network
│       │   └── evaluator.py     # Team quality predictor
│       └── player/              # OU battle player
│           ├── state_encoder.py # OU-specific state encoding
│           ├── network.py       # Policy-value network
│           └── team_preview.py  # Lead selection
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

### Running OU Training
```bash
# Train OU player with sample teams
python scripts/train_ou.py

# Short training run
python scripts/train_ou.py -t 100000

# Resume from checkpoint
python scripts/train_ou.py --resume

# Use parallel environments
python scripts/train_ou.py --num-envs 4

# Monitor with TensorBoard
tensorboard --logdir runs/ou/
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
- `src/showdown_bot/evaluation/elo.py` - Skill rating system (uses Elo algorithm internally, displayed as "skill" not "elo" to avoid confusion with Showdown's ladder)
- `src/showdown_bot/environment/state_encoder.py` - State encoding

### Browser Play (Working)
Play against the trained bot in a web browser using 3 terminals:

**Terminal 1: Start Pokemon Showdown Server**
```bash
cd ~/pokemon-showdown && node pokemon-showdown start --no-security
```

**Terminal 2: Start HTTP Server**
```bash
cd ~/showdown-bot && ./scripts/start_local_play.sh
```

**Terminal 3: Start the Bot**
```bash
cd ~/showdown-bot
source .venv/bin/activate  # or .venv-rocm/bin/activate
python scripts/play.py
```

**In Browser:**
1. Open: `http://localhost:8080/play.pokemonshowdown.com/testclient.html?~~localhost:8000`
2. Open console (F12) and login: `app.socket.send('|/trn YourName,0,')`
3. Challenge bot: `app.socket.send('|/challenge TrainedBot, gen9randombattle')`

Or use UI: Click "Find a user" → "TrainedBot" → "Challenge" → "gen9randombattle"

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
