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

- **Algorithm**: PPO with GAE, target_kl=0.015 for stability
- **Model**: Transformer encoder with attention over Pokemon/moves
- **Training**: Pure self-play (current model vs itself)
- **Evaluation**: Periodic benchmark against MaxDamage (every 50k steps)
- **Action space**: 4 moves + 5 switches (+ tera variants for OU)

## Training Features

### Learning Rate Scheduling
Linear decay from `3e-4` → `3e-5` over training. Prevents late-stage oscillation.

### Self-Play Training
- **Training**: 100% self-play (current model vs itself)
- **Evaluation**: 50 games vs MaxDamage every 50k steps (Bench% metric)
- MaxDamage is only used for benchmarking, not training
- This ensures the agent learns that opponents can switch from the start

### Why Not Train Against MaxDamage?
MaxDamage never switches and plays predictably. Training against it teaches:
- Never switch (opponent won't punish it)
- Pure aggression is optimal
These habits hurt against real opponents. MaxDamage is useful as a benchmark only.

## Technical Notes

- Uses action masking for illegal moves
- Reward: win/loss + HP differential + KO bonus
- Browser extension uses `world: "MAIN"` to access PS variables
- Battle request at `app.curRoom.request` (not `.battle.request`)
- Optimal config: 1 PS server, 6 envs per process (~290/s single, ~500/s multi)
- Multi-process training uses 2 workers to bypass Python's GIL
- Entropy coefficient: 0.01 (balanced exploration)

## Planned Improvements

### Best Checkpoint Gating (AlphaZero-style)
Add a "best" checkpoint that gates model promotion:

- **Training**: 100% self-play (current vs current) - unchanged
- **Evaluation every 100k steps**:
  - 50 games vs MaxDamage → Bench% (absolute skill metric)
  - 150 games vs Best → promotion gate
- **Promotion**: If current beats best ≥60%, current becomes new best

Benefits:
- Prevents regression (best only updates when convincingly beaten)
- MaxDamage catches "learned to beat itself but forgot fundamentals"
- Best checkpoint catches "got worse at the game overall"
- 60% threshold prevents noisy promotions
