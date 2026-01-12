# Gen 9 OU Module

A reinforcement learning system for playing and teambuilding in Pokemon Showdown's Gen 9 OverUsed (OU) tier.

## Overview

Unlike Random Battles where teams are assigned, OU requires:
1. **Teambuilding**: Constructing a team of 6 Pokemon with optimal movesets, items, abilities, EVs, and natures
2. **Playing**: Making optimal decisions during battle with that team
3. **Team Preview**: Strategic lead selection at the start of each battle

This module uses a **hybrid approach** where the player and teambuilder are trained separately but connected through shared representations and feedback loops.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Training Loop                             │
│  ┌──────────────┐    teams    ┌──────────────┐                  │
│  │  Teambuilder │ ──────────> │    Player    │                  │
│  │   (Policy)   │             │   (Policy)   │                  │
│  └──────────────┘             └──────────────┘                  │
│         ^                            │                          │
│         │      win/loss feedback     │                          │
│         └────────────────────────────┘                          │
│                                                                  │
│  Shared: Pokemon Embeddings, Move Embeddings, Type Charts       │
└─────────────────────────────────────────────────────────────────┘
```

### Why Separate Systems?

**Teambuilding** has:
- Massive discrete action space (~1000 Pokemon × ~800 moves × ~200 items × ...)
- Sparse rewards (only know if team is good after many battles)
- Long-horizon optimization (team is fixed for entire battle)

**Playing** has:
- Smaller action space (~9 actions per turn)
- Dense rewards (HP changes, knockouts)
- Short-horizon decisions (per-turn)

Training them jointly would be extremely difficult. Instead:
1. Train player with fixed/sampled teams
2. Use player's performance to evaluate teams
3. Train teambuilder to maximize expected player performance

## Directory Structure

```
ou/
├── README.md                 # This file
├── __init__.py
├── shared/                   # Shared components
│   ├── __init__.py
│   ├── embeddings.py         # Pokemon, move, item, ability embeddings
│   ├── encoders.py           # State encoders for OU battles
│   └── data_loader.py        # Load Pokemon/move data, usage stats
├── teambuilder/              # Team construction system
│   ├── __init__.py
│   ├── team_repr.py          # Team representation/encoding
│   ├── generator.py          # Team generation policy
│   ├── evaluator.py          # Team fitness evaluation
│   └── trainer.py            # Teambuilder training loop
├── player/                   # Battle playing system
│   ├── __init__.py
│   ├── state_encoder.py      # OU-specific state encoding
│   ├── network.py            # OU player network (with team preview)
│   ├── team_preview.py       # Lead selection logic
│   └── trainer.py            # Player training loop
├── training/                 # Combined training pipelines
│   ├── __init__.py
│   ├── config.py             # OUTrainingConfig hyperparameters
│   ├── buffer.py             # OUExperienceBuffer, TeamOutcomeBuffer
│   ├── player_trainer.py     # OUPlayerTrainer (PPO)
│   ├── teambuilder_trainer.py # TeambuilderTrainer (evaluator + generator)
│   ├── self_play.py          # OUSelfPlayManager, OUOpponentPool
│   ├── joint_trainer.py      # JointTrainer (feedback loop coordinator)
│   └── curriculum.py         # Curriculum learning strategies (future)
└── data/                     # Static data and caches
    ├── pokemon_data/         # Pokemon stats, movesets, etc.
    ├── usage_stats/          # Smogon usage statistics
    └── sample_teams/         # Known good teams for bootstrapping
```

## Components

### 1. Shared Embeddings (`shared/`)

Learned representations shared between player and teambuilder:

```python
class PokemonEmbedding:
    """Dense embedding for Pokemon species.

    Captures: typing, base stats, common roles, tier placement
    Initialized from: Smogon usage stats, type chart, stat distributions
    Updated during: Player training (which Pokemon are valuable in which contexts)
    """

class MoveEmbedding:
    """Dense embedding for moves.

    Captures: type, power, accuracy, priority, effects, common targets
    """

class SynergyEncoder:
    """Encodes relationships between Pokemon.

    Captures: Type coverage, hazard support, weather/terrain synergy,
              offensive/defensive cores, speed tiers
    """
```

### 2. Teambuilder (`teambuilder/`)

**Approach**: Autoregressive generation with learned value function

The teambuilder generates teams one Pokemon at a time:
1. Select Pokemon 1 (from ~100 OU-viable options)
2. Select moveset for Pokemon 1 (from legal moves)
3. Select item for Pokemon 1
4. Repeat for Pokemon 2-6, conditioning on previous selections

**Key insight**: Order matters for conditioning, but final team is unordered.
Generate in order of "team anchor" importance (e.g., start with win condition).

```python
class TeamGenerator:
    """Autoregressive team generation.

    State: Partially built team (0-5 Pokemon)
    Action: Add next Pokemon with moveset/item/ability

    Uses: Transformer over current team members to condition generation
    """

class TeamEvaluator:
    """Predicts expected win rate for a complete team.

    Trained on: (team, opponent_team, outcome) tuples from battles
    Used for: MCTS-style team search, filtering generated teams
    """
```

**Training signal**:
- Primary: Win rate when played by the trained player
- Auxiliary: Coverage metrics, archetype classification, similarity to high-usage teams

### 3. Player (`player/`)

Similar to randbats player but with key additions:

```python
class OUStateEncoder:
    """Encodes OU battle state.

    Additional features vs randbats:
    - Full knowledge of own team (moves, items, EVs)
    - Team preview information
    - Opponent's revealed team members
    - Usage-based opponent prediction
    """

class TeamPreviewSelector:
    """Selects lead Pokemon during team preview.

    Input: Own team (6 Pokemon) + Opponent team (6 Pokemon)
    Output: Ordering preference for own team (who leads, who stays back)

    Critical for: Weather wars, hazard control, matchup fishing
    """

class OUPlayerNetwork:
    """Policy/value network for OU battles.

    Architecture:
    - Pokemon encoder (shared with teambuilder)
    - Team encoder (attention over all team members)
    - Battle state encoder
    - Policy head (move/switch selection)
    - Value head (expected game outcome)
    """
```

### 4. Training Pipeline (`training/`)

**Phase 1: Bootstrap Player**
- Use sample teams from Smogon/high-ladder replays
- Train player via self-play and vs ladder

**Phase 2: Train Teambuilder**
- Generate candidate teams
- Evaluate with trained player (play N games each)
- Update teambuilder toward higher-performing teams

**Phase 3: Joint Refinement**
- Alternate: Generate teams → Train player on new teams → Evaluate teams → Update teambuilder
- Player improves at playing diverse teams
- Teambuilder learns what the player can execute well

```python
class JointTrainer:
    """Alternating training of player and teambuilder.

    1. Sample/generate batch of teams
    2. Play games with each team (collect player experiences)
    3. Update player policy
    4. Update teambuilder with team performance data
    5. Repeat
    """
```

## Key Challenges

### 1. Massive Teambuilding Space
- ~100 OU Pokemon × ~20 viable moves each × ~50 items × abilities × EVs
- Solution: Autoregressive generation + learned constraints from usage data

### 2. Team Evaluation is Noisy
- Need many games to estimate team quality
- Solution: Auxiliary losses (coverage metrics, archetype matching), ensemble evaluation

### 3. Opponent Modeling
- In OU, predicting opponent's unrevealed Pokemon is crucial
- Solution: Usage-based priors, Bayesian updating as Pokemon are revealed

### 4. Metagame Shift
- OU meta changes with bans, new discoveries
- Solution: Continual learning, usage stats integration, periodic retraining

### 5. Team Preview Strategy
- Lead selection is a mind game
- Solution: Learned lead selection policy, possibly with opponent modeling

## Data Sources

1. **Smogon Usage Stats**: https://www.smogon.com/stats/
   - Pokemon usage rates, common movesets, teammates, checks/counters

2. **Pokemon Showdown Replays**: Via replay URLs or local server
   - High-ELO games for training data

3. **Smogon Analyses**: https://www.smogon.com/dex/sv/pokemon/
   - Role descriptions, set recommendations

4. **poke-env**: Already integrated
   - Battle simulation, legal move checking

## Implementation Phases

### Phase 1: Foundation - COMPLETE
- [x] Shared embeddings (Pokemon, moves, items)
- [x] OU state encoder
- [x] Basic OU player (fixed team)
- [x] Sample team loader

### Phase 2: Player Training - IN PROGRESS
- [x] Training script (`scripts/train_ou.py`)
- [x] OURLPlayer / OUNeuralNetworkPlayer integration
- [x] action_to_battle_order() for 13 actions
- [x] Experience collection (`OUTrainablePlayer` + PPO updates)
- [x] Self-play training loop (`OUSelfPlayManager`, `OUOpponentPool`)
- [x] Team preview selection (heuristic lead selector integrated)
- [ ] Team preview learning (neural network training - future)
- [ ] Evaluation vs ladder

### Phase 3: Teambuilder MVP - COMPLETE
- [x] Team representation
- [x] Autoregressive generator (neural + usage-based)
- [x] UsageBasedGenerator (generates teams from Smogon usage stats)
- [x] Basic team evaluation (placeholder network)
- [x] Teambuilder training loop (infrastructure ready)
- [x] Full TeamGenerator implementation with:
  - Species selection weighted by usage rate + teammate correlations
  - Move selection from common movesets
  - Item/ability selection from usage data
  - Nature/EV spreads from common spreads
  - Proper name formatting (format_name helper)
- [x] Team → tensor encoding for evaluator (TeamTensorEncoder)

### Phase 4: Joint Training - COMPLETE
- [x] Player-teambuilder feedback loop (JointTrainer class)
- [x] Evaluator training with BCE loss
- [x] Team performance tracking (TeamOutcomeBuffer)
- [x] Team rotation based on win rates
- [x] JointTrainingManager (battle loop integration via poke-env)
- [x] Joint checkpoint save/load
- [x] Curriculum learning strategies (progressive, matchup, complexity, adaptive)
- [ ] Metagame adaptation (future)

**Note**: The TeamGenerator now generates fully functional teams using Smogon usage statistics.
The training script supports three modes:
- `--mode player`: Train battle decisions (default, fully functional)
- `--mode teambuilder`: Train team generator (uses usage-based generation)
- `--mode joint`: Alternating training (player + teambuilder)

## Usage

```bash
# Train OU player with sample teams (requires Pokemon Showdown server)
python scripts/train_ou.py                      # Default player mode
python scripts/train_ou.py -m player            # Explicit player mode

# Short training run
python scripts/train_ou.py -t 100000

# Resume from checkpoint
python scripts/train_ou.py --resume

# Use parallel environments
python scripts/train_ou.py --num-envs 4

# Disable self-play (random/maxdamage opponents only)
python scripts/train_ou.py --no-self-play

# Monitor with TensorBoard
tensorboard --logdir runs/ou/
```

### Generate Teams (Fully Functional)
```python
from showdown_bot.ou.teambuilder import UsageBasedGenerator, team_to_showdown_paste
from showdown_bot.ou.shared.data_loader import UsageStatsLoader

# Load Smogon usage stats
loader = UsageStatsLoader()
loader.load(tier="gen9ou", rating=1695)

# Generate a team
gen = UsageBasedGenerator(usage_loader=loader)
team = gen.generate_team(temperature=1.0)  # Higher temp = more random

# Convert to Showdown paste format
full_team = team.to_team()
paste = team_to_showdown_paste(full_team)
print(paste)
```

### Teambuilder Training
```bash
# Train teambuilder with frozen player
python scripts/train_ou.py --mode teambuilder --player-checkpoint data/checkpoints/ou/best_model.pt

# Teambuilder with custom settings
python scripts/train_ou.py -m teambuilder --total-teams 500 --games-per-team 5
```

### Joint Training (Fully Functional)
Joint training coordinates player and teambuilder with a feedback loop:
1. Teams are drawn from an active pool (initially sample teams + generated teams)
2. Player battles with these teams against random/maxdamage opponents
3. Battle outcomes train both:
   - Player: PPO updates on battle experiences
   - Teambuilder: Evaluator learns to predict win rates (BCE loss)
4. Worst-performing teams are rotated out and replaced with newly generated ones
5. TensorBoard logs track player win rate, skill rating, and best team performance

```bash
# Joint player + teambuilder training
python scripts/train_ou.py --mode joint

# With parallel environments
python scripts/train_ou.py --mode joint --num-envs 4

# Resume from checkpoint
python scripts/train_ou.py --mode joint --resume

# Monitor with TensorBoard
tensorboard --logdir runs/ou/joint/
```

Checkpoints are saved to `data/checkpoints/ou/joint/`:
- `player.pt`: Player network checkpoint
- `teambuilder.pt`: Generator + evaluator checkpoint
- `joint_state.pt`: Training statistics

### Curriculum Learning
Curriculum learning progressively increases training difficulty based on agent performance:

```bash
# Use adaptive curriculum (default, combines all strategies)
python scripts/train_ou.py --mode joint --curriculum adaptive

# Progressive difficulty only (random → maxdamage → self-play)
python scripts/train_ou.py --mode joint --curriculum progressive

# Matchup focus (targets weak opponent types)
python scripts/train_ou.py --mode joint --curriculum matchup

# Team complexity (starts with sample teams, adds generated)
python scripts/train_ou.py --mode joint --curriculum complexity

# Disable curriculum (random opponent selection)
python scripts/train_ou.py --mode joint --curriculum none
```

**Available strategies:**
- **`adaptive`**: Combines all strategies based on training phase (recommended)
- **`progressive`**: 5 difficulty levels from BEGINNER (random only) to EXPERT (strong self-play)
- **`matchup`**: Tracks win rates per opponent type, focuses on weaknesses
- **`complexity`**: Gradually introduces generated teams vs sample teams

**Difficulty progression:**
| Level | Opponent Mix |
|-------|-------------|
| BEGINNER | 100% Random |
| EASY | 70% Random, 30% MaxDamage |
| MEDIUM | 50% MaxDamage, 50% Weak Self-Play |
| HARD | Mixed Self-Play |
| EXPERT | Strong Self-Play |

Promotion requires 65%+ win rate over 50 battles. Demotion occurs at <35% win rate.
Curriculum state is saved in checkpoints and restored on resume.

### Planned Commands (Not Yet Implemented)
```bash
# Generate a team from CLI
python scripts/generate_team.py --style balanced

# Play on ladder with generated teams
python scripts/play_ou.py --player-checkpoint player.pt
```

## References

- Smogon OU Metagame: https://www.smogon.com/dex/sv/formats/ou/
- poke-env documentation: https://poke-env.readthedocs.io/
- Pokemon Showdown: https://pokemonshowdown.com/
