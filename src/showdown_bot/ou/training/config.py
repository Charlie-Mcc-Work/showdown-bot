"""Training configuration for OU module.

Contains hyperparameters for both the player and teambuilder training.
"""

from dataclasses import dataclass, field


@dataclass
class OUTrainingConfig:
    """Configuration for OU training.

    Attributes:
        # General
        device: Device to use for training
        seed: Random seed for reproducibility

        # Player training (PPO)
        player_lr: Learning rate for player network
        player_gamma: Discount factor
        player_gae_lambda: GAE lambda for advantage estimation
        player_clip_epsilon: PPO clipping parameter
        player_value_coef: Value loss coefficient
        player_entropy_coef: Entropy bonus coefficient
        player_max_grad_norm: Gradient clipping
        player_num_epochs: PPO epochs per update
        player_batch_size: Minibatch size for player updates
        player_rollout_steps: Steps per rollout

        # Teambuilder training
        teambuilder_lr: Learning rate for teambuilder
        teambuilder_batch_size: Batch size for teambuilder updates
        teambuilder_temperature: Sampling temperature
        teambuilder_min_games: Min games before evaluating a team

        # Self-play
        self_play_ratio: Fraction of games vs self-play opponents
        checkpoint_interval: Save checkpoint every N games
        opponent_pool_size: Max opponents in pool
        skill_k_factor: K-factor for skill rating updates

        # Logging
        log_interval: Log every N steps
        eval_interval: Evaluate every N steps
        save_interval: Save checkpoint every N steps
    """

    # General
    device: str = "cuda"
    seed: int = 42

    # Player training (PPO)
    player_lr: float = 3e-4
    player_gamma: float = 0.99
    player_gae_lambda: float = 0.95
    player_clip_epsilon: float = 0.2
    player_value_coef: float = 0.5
    player_entropy_coef: float = 0.01
    player_max_grad_norm: float = 0.5
    player_num_epochs: int = 4
    player_batch_size: int = 64
    player_rollout_steps: int = 2048

    # Teambuilder training
    teambuilder_lr: float = 1e-4
    teambuilder_batch_size: int = 32
    teambuilder_temperature: float = 1.0
    teambuilder_min_games: int = 10

    # Self-play
    self_play_ratio: float = 0.5
    checkpoint_interval: int = 1000
    opponent_pool_size: int = 20
    skill_k_factor: float = 32.0

    # Logging
    log_interval: int = 100
    eval_interval: int = 1000
    save_interval: int = 5000

    # Paths
    checkpoint_dir: str = "data/checkpoints/ou"
    log_dir: str = "runs/ou"

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "device": self.device,
            "seed": self.seed,
            "player_lr": self.player_lr,
            "player_gamma": self.player_gamma,
            "player_gae_lambda": self.player_gae_lambda,
            "player_clip_epsilon": self.player_clip_epsilon,
            "player_value_coef": self.player_value_coef,
            "player_entropy_coef": self.player_entropy_coef,
            "player_max_grad_norm": self.player_max_grad_norm,
            "player_num_epochs": self.player_num_epochs,
            "player_batch_size": self.player_batch_size,
            "player_rollout_steps": self.player_rollout_steps,
            "teambuilder_lr": self.teambuilder_lr,
            "teambuilder_batch_size": self.teambuilder_batch_size,
            "teambuilder_temperature": self.teambuilder_temperature,
            "teambuilder_min_games": self.teambuilder_min_games,
            "self_play_ratio": self.self_play_ratio,
            "checkpoint_interval": self.checkpoint_interval,
            "opponent_pool_size": self.opponent_pool_size,
            "skill_k_factor": self.skill_k_factor,
            "log_interval": self.log_interval,
            "eval_interval": self.eval_interval,
            "save_interval": self.save_interval,
            "checkpoint_dir": self.checkpoint_dir,
            "log_dir": self.log_dir,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "OUTrainingConfig":
        """Create from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
