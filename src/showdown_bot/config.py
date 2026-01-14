"""Configuration and hyperparameters for training."""

from pydantic import Field
from pydantic_settings import BaseSettings


class TrainingConfig(BaseSettings):
    """Training hyperparameters."""

    # PPO hyperparameters
    learning_rate: float = Field(default=3e-4, description="Initial learning rate for optimizer")
    final_learning_rate: float = Field(default=3e-5, description="Final learning rate (linear decay)")
    gamma: float = Field(default=0.99, description="Discount factor for rewards")
    gae_lambda: float = Field(default=0.95, description="GAE lambda for advantage estimation")
    clip_epsilon: float = Field(default=0.2, description="PPO clipping parameter")
    entropy_coef: float = Field(default=0.025, description="Entropy bonus coefficient (higher = more exploration)")
    value_coef: float = Field(default=0.15, description="Value loss coefficient (lower to prevent value loss dominating policy)")
    max_grad_norm: float = Field(default=0.5, description="Max gradient norm for clipping")

    # Training schedule
    num_epochs: int = Field(default=4, description="PPO epochs per update")
    batch_size: int = Field(default=64, description="Minibatch size")
    rollout_steps: int = Field(default=2048, description="Steps per rollout before update")
    total_timesteps: int = Field(default=10_000_000, description="Total training timesteps")

    # Self-play
    opponent_pool_size: int = Field(default=10, description="Number of past checkpoints to keep")
    checkpoint_interval: int = Field(default=50_000, description="Steps between checkpoints")
    self_play_ratio: float = Field(
        default=0.8, description="Fraction of games against self vs random (used if curriculum disabled)"
    )

    # Curriculum opponent selection - adjusts opponent mix based on skill
    curriculum_enabled: bool = Field(
        default=True, description="Enable curriculum-based opponent selection"
    )
    # Skill thresholds for curriculum stages
    curriculum_skill_min: float = Field(
        default=1000.0, description="Skill level where curriculum starts (early stage)"
    )
    curriculum_skill_max: float = Field(
        default=6000.0, description="Skill level where curriculum ends (late stage)"
    )
    # Early stage ratios (when skill <= curriculum_skill_min)
    curriculum_early_self_play: float = Field(
        default=0.2, description="Self-play ratio in early training"
    )
    curriculum_early_max_damage: float = Field(
        default=0.5, description="MaxDamage ratio in early training"
    )
    # Late stage ratios (when skill >= curriculum_skill_max)
    curriculum_late_self_play: float = Field(
        default=0.6, description="Self-play ratio in late training (reduced to avoid echo chamber)"
    )
    curriculum_late_max_damage: float = Field(
        default=0.4, description="MaxDamage ratio in late training (teaches optimal damage calc)"
    )
    # Random ratio = 1 - self_play - max_damage (0% at late stage - random provides no signal)

    # Environment
    num_envs: int = Field(default=8, description="Number of parallel environments")
    battle_format: str = Field(default="gen9randombattle", description="Pokemon Showdown format")

    # Model architecture
    hidden_dim: int = Field(default=256, description="Hidden dimension for MLP layers")
    num_attention_heads: int = Field(default=4, description="Number of attention heads")
    num_transformer_layers: int = Field(default=2, description="Number of transformer layers")
    embedding_dim: int = Field(default=64, description="Embedding dimension for entities")

    # Embeddings
    num_species: int = Field(default=1500, description="Number of Pokemon species to embed")
    num_moves: int = Field(default=1000, description="Number of moves to embed")
    num_abilities: int = Field(default=400, description="Number of abilities to embed")
    num_items: int = Field(default=500, description="Number of items to embed")

    # Logging
    log_interval: int = Field(default=1000, description="Steps between logging")
    eval_interval: int = Field(default=10_000, description="Steps between evaluation")
    save_dir: str = Field(default="data/checkpoints", description="Directory for saving models")
    log_dir: str = Field(default="runs", description="Directory for TensorBoard logs")

    class Config:
        env_prefix = "SHOWDOWN_"


class ServerConfig(BaseSettings):
    """Pokemon Showdown server configuration."""

    host: str = Field(default="localhost", description="Showdown server host")
    port: int = Field(default=8000, description="Showdown server port")
    username: str = Field(default="RLBot", description="Bot username")
    password: str | None = Field(default=None, description="Bot password (optional)")

    class Config:
        env_prefix = "SHOWDOWN_"

    @property
    def server_url(self) -> str:
        return f"ws://{self.host}:{self.port}/showdown/websocket"


# Global config instances
training_config = TrainingConfig()
server_config = ServerConfig()
