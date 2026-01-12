"""Training module for OU player and teambuilder.

This module provides:
- PlayerTrainer: PPO-based trainer for the battle player
- TeambuilderTrainer: Trainer for the autoregressive team generator
- OUExperienceBuffer: Experience replay for OU battles
- OUTrainingConfig: Configuration for OU training

The training pipeline works as follows:
1. Generate teams using the teambuilder
2. Play battles using generated teams
3. Update player network based on battle outcomes
4. Update teambuilder based on team win rates
"""

from showdown_bot.ou.training.config import OUTrainingConfig
from showdown_bot.ou.training.buffer import (
    OUExperienceBuffer,
    OUTransition,
    TeamOutcomeBuffer,
    TeamOutcome,
)
from showdown_bot.ou.training.player_trainer import OUPlayerTrainer
from showdown_bot.ou.training.teambuilder_trainer import TeambuilderTrainer

__all__ = [
    "OUTrainingConfig",
    "OUExperienceBuffer",
    "OUTransition",
    "TeamOutcomeBuffer",
    "TeamOutcome",
    "OUPlayerTrainer",
    "TeambuilderTrainer",
]
