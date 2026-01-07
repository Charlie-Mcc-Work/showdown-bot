"""Training module for PPO and self-play."""

from showdown_bot.training.buffer import RolloutBuffer
from showdown_bot.training.ppo import PPO, PPOStats
from showdown_bot.training.trainer import Trainer, TrainablePlayer, TrainingStats
from showdown_bot.training.self_play import (
    SelfPlayManager,
    OpponentPool,
    OpponentInfo,
    HistoricalPlayer,
    calculate_elo_update,
)

__all__ = [
    "RolloutBuffer",
    "PPO",
    "PPOStats",
    "Trainer",
    "TrainablePlayer",
    "TrainingStats",
    "SelfPlayManager",
    "OpponentPool",
    "OpponentInfo",
    "HistoricalPlayer",
    "calculate_elo_update",
]
