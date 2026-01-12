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
│   ├── joint_trainer.py      # Alternating player/teambuilder training
│   └── curriculum.py         # Curriculum learning strategies
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

### Phase 1: Foundation (Current)
- [ ] Shared embeddings (Pokemon, moves, items)
- [ ] OU state encoder
- [ ] Basic OU player (fixed team)
- [ ] Sample team loader

### Phase 2: Player Training
- [ ] Team preview selection
- [ ] Self-play training loop
- [ ] Evaluation vs ladder

### Phase 3: Teambuilder MVP
- [ ] Team representation
- [ ] Autoregressive generator
- [ ] Basic team evaluation

### Phase 4: Joint Training
- [ ] Player-teambuilder feedback loop
- [ ] Curriculum strategies
- [ ] Metagame adaptation

## Usage (Planned)

```bash
# Train OU player with sample teams
python scripts/train_ou.py --mode player --teams data/sample_teams/

# Train teambuilder with frozen player
python scripts/train_ou.py --mode teambuilder --player-checkpoint best_player.pt

# Joint training
python scripts/train_ou.py --mode joint

# Generate a team
python scripts/generate_team.py --checkpoint teambuilder.pt --style balanced

# Play on ladder
python scripts/play_ou.py --player-checkpoint player.pt --team-checkpoint teambuilder.pt
```

## References

- Smogon OU Metagame: https://www.smogon.com/dex/sv/formats/ou/
- poke-env documentation: https://poke-env.readthedocs.io/
- Pokemon Showdown: https://pokemonshowdown.com/
