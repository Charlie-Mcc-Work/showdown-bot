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
from showdown_bot.ou.training.self_play import (
    OUOpponentInfo,
    OUHistoricalPlayer,
    OUOpponentPool,
    OUSelfPlayManager,
    calculate_elo_update,
)
from showdown_bot.ou.training.joint_trainer import (
    JointTrainer,
    JointTrainingStats,
    create_joint_trainer,
)
from showdown_bot.ou.training.curriculum import (
    CurriculumStrategy,
    CurriculumConfig,
    CurriculumState,
    OpponentType,
    DifficultyLevel,
    ProgressiveDifficultyCurriculum,
    MatchupFocusCurriculum,
    TeamComplexityCurriculum,
    AdaptiveCurriculum,
    create_curriculum,
)

__all__ = [
    "OUTrainingConfig",
    "OUExperienceBuffer",
    "OUTransition",
    "TeamOutcomeBuffer",
    "TeamOutcome",
    "OUPlayerTrainer",
    "TeambuilderTrainer",
    "OUOpponentInfo",
    "OUHistoricalPlayer",
    "OUOpponentPool",
    "OUSelfPlayManager",
    "calculate_elo_update",
    "JointTrainer",
    "JointTrainingStats",
    "create_joint_trainer",
    # Curriculum
    "CurriculumStrategy",
    "CurriculumConfig",
    "CurriculumState",
    "OpponentType",
    "DifficultyLevel",
    "ProgressiveDifficultyCurriculum",
    "MatchupFocusCurriculum",
    "TeamComplexityCurriculum",
    "AdaptiveCurriculum",
    "create_curriculum",
]
