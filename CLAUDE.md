# Pokemon Showdown RL Bot

A reinforcement learning bot for Pokemon Showdown using PPO and self-play training.

## Current Status

**Random Battles**: Complete - training, self-play, browser extension all working.

**Gen 9 OU**: Complete - joint player/teambuilder training with curriculum learning.
See `src/showdown_bot/ou/README.md` for full OU documentation.

**Next up**: Extended training runs, ladder evaluation.

## Quick Start

**Prerequisite**: Pokemon Showdown installed at `~/pokemon-showdown`
```bash
# If not installed:
git clone https://github.com/smogon/pokemon-showdown.git ~/pokemon-showdown
cd ~/pokemon-showdown && npm install
```

### Random Battles Training

**Recommended: Automated training script**
```bash
./scripts/run_training.sh    # Does everything automatically
```
This script:
- Starts 3 Pokemon Showdown servers automatically
- Runs training with 12 parallel environments across all servers
- Auto-resumes from checkpoint
- Restarts on memory issues or run completion (5M steps per run)
- Ctrl+C gracefully saves checkpoint and exits

**Manual training:**
```bash
python scripts/train.py                    # Train with self-play
python scripts/train.py --num-envs 4       # Parallel envs (faster)
python scripts/train.py --resume           # Resume from checkpoint
tensorboard --logdir runs/                 # Monitor

# Multi-server (manual)
python scripts/train.py --num-envs 12 --server-ports 8000 8001 8002
```

**Multi-process training (bypasses GIL):**
```bash
# For better CPU utilization with multi-core systems
./scripts/run_training_multiproc.sh --workers 4 --envs-per-worker 4
# = 16 total envs across 4 Python processes, fully utilizing multiple CPU cores
```

### OU Training
```bash
# Joint training (recommended) - trains player + teambuilder together
python scripts/train_ou.py --mode joint --num-envs 4

# With curriculum learning
python scripts/train_ou.py --mode joint --curriculum adaptive  # default
python scripts/train_ou.py --mode joint --curriculum progressive

# Player only (with sample teams)
python scripts/train_ou.py --mode player
```

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
uv run python scripts/coach_server.py
# Load extension/ in chrome://extensions (Developer mode)
# Play on https://play.pokemonshowdown.com - coach panel shows AI recommendations
```

## Key Files

| Area | Files |
|------|-------|
| Training | `scripts/train.py`, `scripts/train_ou.py`, `scripts/train_multiproc.py` |
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

## Technical Notes

- Uses action masking for illegal moves
- Reward: win/loss + HP differential + KO bonus
- Browser extension uses `world: "MAIN"` to access PS variables
- Battle request at `app.curRoom.request` (not `.battle.request`)
- Standard training is single-threaded due to Python's GIL; use `train_multiproc.py` for multi-core CPUs
