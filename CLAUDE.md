# Pokemon Showdown RL Bot

A reinforcement learning bot for Pokemon Showdown using PPO and self-play training.

## Current Status

**Random Battles**: Complete - training, self-play, browser extension all working.

**Gen 9 OU**: Complete - joint player/teambuilder training with curriculum learning.
See `src/showdown_bot/ou/README.md` for full OU documentation.

## Quick Start

**Prerequisite**: Pokemon Showdown installed at `~/pokemon-showdown`
```bash
# If not installed:
git clone https://github.com/smogon/pokemon-showdown.git ~/pokemon-showdown
cd ~/pokemon-showdown && npm install
```

### Training Commands

```bash
# Random Battles - single-process (default, ~290/s)
./scripts/run_training.sh

# Random Battles - multi-process (faster, ~500/s, uses 2 CPU cores)
./scripts/run_training.sh --multiproc

# OU Joint Training - player + teambuilder together
./scripts/run_training_ou.sh

# OU Player-only - faster, uses sample teams
./scripts/run_training_ou.sh --mode player
```

All scripts automatically:
- Start Pokemon Showdown servers
- Resume from checkpoint
- Save on Ctrl+C
- Use optimal settings (1 server, 6 envs per worker)

### Monitoring

```bash
# Live stats in another terminal
./scripts/monitor_training.sh

# Or watch logs directly
tail -f logs/worker_0.log        # Random battles
tail -f logs/ou_worker_0.log     # OU training

# TensorBoard for detailed metrics
tensorboard --logdir runs/
```

### Training Speeds

| Mode | Speed | What's Training |
|------|-------|-----------------|
| Random Battles (single-process) | ~290 it/s | Player network |
| Random Battles (multi-process) | ~500 it/s | Player network (2 CPU cores) |
| OU Player-only | ~150 it/s | Player network (with OU mechanics) |
| OU Joint | ~60 it/s | Player + Teambuilder networks together |

### Browser Play
```bash
# Terminal 1: Pokemon Showdown server (see above)
# Terminal 2: HTTP server
./scripts/start_local_play.sh
# Terminal 3: Bot
python scripts/play.py
# Browser: http://localhost:8080/play.pokemonshowdown.com/testclient.html?~~localhost:8000
# Console: app.socket.send('|/trn YourName,0,')
# Challenge: TrainedBot to gen9randombattle
```

### Browser Coach Extension
```bash
python scripts/coach_server.py
# Load extension/ in chrome://extensions (Developer mode)
# Play on https://play.pokemonshowdown.com - coach panel shows AI recommendations
```

## Key Files

| Area | Files |
|------|-------|
| Training | `scripts/train.py`, `scripts/train_multiproc.py`, `scripts/train_ou.py` |
| Scripts | `scripts/run_training.sh`, `scripts/run_training_ou.sh` |
| Monitoring | `scripts/monitor_training.sh` |
| Network | `src/showdown_bot/models/network.py` |
| PPO | `src/showdown_bot/training/ppo.py` |
| Self-play | `src/showdown_bot/training/self_play.py` |
| OU module | `src/showdown_bot/ou/` (see its README) |
| Config | `src/showdown_bot/config.py` |
| Coach | `scripts/coach_server.py`, `extension/` |

## Architecture

- **Algorithm**: PPO with GAE
- **Model**: Transformer encoder with attention over Pokemon/moves
- **Training**: Self-play against historical checkpoints
- **Action space**: 4 moves + 5 switches (+ tera variants for OU)

## Training Features

### Learning Rate Scheduling
Linear decay from `3e-4` → `3e-5` over training. Prevents late-stage oscillation.

### Curriculum Opponent Selection
Opponent mix adjusts based on Bench% (win rate vs MaxDamage):

| Stage | Self-Play | MaxDamage |
|-------|-----------|-----------|
| Early (Bench <30%) | 30% | 70% |
| Late (Bench >70%) | 95% | 5% |

- **MaxDamage**: Deterministic optimal-damage player, teaches damage calculations
- **Self-play**: Historical checkpoints with diverse sampling
- Model graduates to more self-play only after proving it can beat MaxDamage

### Diverse Opponent Sampling
Self-play uses mixed sampling to avoid echo chambers:
- 40% uniform (any opponent in pool)
- 30% challenge (prefer stronger opponents)
- 30% skill-matched (similar skill level)

### Opponent Pool Diversity
Pool pruning keeps: oldest (weak baseline), newest, best, worst, plus skill-diverse selection.

## Technical Notes

- Uses action masking for illegal moves
- Reward: win/loss + HP differential + KO bonus
- Browser extension uses `world: "MAIN"` to access PS variables
- Battle request at `app.curRoom.request` (not `.battle.request`)
- Optimal config: 1 PS server, 6 envs per process (~290/s single, ~500/s multi)
- Multi-process training uses 2 workers to bypass Python's GIL
- Entropy coefficient: 0.025 (encourages exploration)
