"""Joint trainer for player and teambuilder.

The JointTrainer coordinates the feedback loop between:
1. Teambuilder: Generates teams based on learned quality function
2. Player: Plays battles with generated teams
3. Feedback: Battle outcomes train both the evaluator and generator

Training alternates between:
- Player training: Improve battle decision-making with current teams
- Teambuilder training: Improve team generation based on battle outcomes

This creates a co-evolutionary system where better teams help the player
learn, and the player's performance informs which teams are good.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.tensorboard import SummaryWriter

from showdown_bot.ou.training.config import OUTrainingConfig
from showdown_bot.ou.training.player_trainer import OUPlayerTrainer
from showdown_bot.ou.training.teambuilder_trainer import TeambuilderTrainer
from showdown_bot.ou.training.curriculum import (
    CurriculumStrategy,
    CurriculumConfig,
    OpponentType,
    create_curriculum,
)
from showdown_bot.ou.player.network import OUPlayerNetwork
from showdown_bot.ou.player.state_encoder import OUStateEncoder
from showdown_bot.ou.teambuilder.generator import TeamGenerator
from showdown_bot.ou.teambuilder.evaluator import TeamEvaluator
from showdown_bot.ou.shared.data_loader import Team, PokemonDataLoader

logger = logging.getLogger(__name__)


@dataclass
class JointTrainingStats:
    """Statistics from joint training."""

    # Overall
    total_battles: int = 0
    total_player_updates: int = 0
    total_teambuilder_updates: int = 0

    # Player stats
    player_wins: int = 0
    player_losses: int = 0
    player_skill_rating: float = 1000.0

    # Teambuilder stats
    teams_generated: int = 0
    best_team_win_rate: float = 0.0
    evaluator_accuracy: float = 0.5

    @property
    def player_win_rate(self) -> float:
        """Current player win rate."""
        total = self.player_wins + self.player_losses
        if total == 0:
            return 0.5
        return self.player_wins / total


class JointTrainer:
    """Coordinates joint training of player and teambuilder.

    The training loop alternates between:
    1. Generate/sample teams from teambuilder
    2. Play battles with player using those teams
    3. Update player on battle experiences
    4. Update teambuilder on battle outcomes
    5. Repeat

    The feedback loop ensures:
    - Teams that win more get higher evaluator scores
    - Generator learns to produce high-scoring teams
    - Player learns to play well with diverse teams
    """

    def __init__(
        self,
        player_trainer: OUPlayerTrainer,
        teambuilder_trainer: TeambuilderTrainer,
        config: OUTrainingConfig,
        device: torch.device | None = None,
        curriculum_strategy: str = "adaptive",
        curriculum_config: CurriculumConfig | None = None,
    ):
        """Initialize joint trainer.

        Args:
            player_trainer: Pre-configured player trainer
            teambuilder_trainer: Pre-configured teambuilder trainer
            config: Training configuration
            device: Device for training
            curriculum_strategy: One of "progressive", "matchup", "complexity", "adaptive"
            curriculum_config: Optional curriculum configuration
        """
        self.player_trainer = player_trainer
        self.teambuilder_trainer = teambuilder_trainer
        self.config = config
        self.device = device or torch.device(config.device)

        # Training state
        self.stats = JointTrainingStats()
        self.current_phase = "player"  # "player" or "teambuilder"

        # Current teams being used for training
        self.active_teams: list[Team] = []
        self.sample_teams: list[Team] = []  # Original sample teams
        self.team_pool_size = 10  # Number of teams to maintain

        # Curriculum learning
        self.curriculum: CurriculumStrategy = create_curriculum(
            strategy=curriculum_strategy,
            config=curriculum_config,
        )
        self.curriculum_enabled = True

        # Logging
        self.writer: SummaryWriter | None = None
        self.global_step = 0

        # Phase scheduling
        self.player_steps_per_phase = config.player_rollout_steps
        self.teambuilder_updates_per_phase = 5
        self.battles_per_team_before_rotate = config.teambuilder_min_games

    def setup_logging(self, log_dir: str | Path) -> None:
        """Set up TensorBoard logging."""
        log_dir = Path(log_dir) / "joint"
        log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(log_dir))

        # Also set up logging for sub-trainers
        self.player_trainer.setup_logging(str(Path(log_dir) / "player"))
        self.teambuilder_trainer.setup_logging(str(Path(log_dir) / "teambuilder"))

    def initialize_teams(
        self,
        sample_teams: list[Team] | None = None,
        num_generated: int = 5,
    ) -> None:
        """Initialize the active team pool.

        Args:
            sample_teams: Pre-existing sample teams to use
            num_generated: Number of teams to generate
        """
        self.active_teams = []
        self.sample_teams = list(sample_teams) if sample_teams else []

        # Add sample teams if provided
        if sample_teams:
            self.active_teams.extend(sample_teams[:self.team_pool_size])
            logger.info(f"Added {len(sample_teams[:self.team_pool_size])} sample teams")

        # Generate additional teams if needed
        while len(self.active_teams) < self.team_pool_size:
            team = self.teambuilder_trainer.generate_team()
            if team:
                self.active_teams.append(team)
                self.stats.teams_generated += 1

        # Update curriculum with sample teams
        if hasattr(self.curriculum, 'sample_teams'):
            self.curriculum.sample_teams = self.sample_teams

        logger.info(f"Initialized team pool with {len(self.active_teams)} teams")

    def get_training_team(self) -> Team | None:
        """Get a team for the next training battle.

        Uses curriculum learning to select teams based on progress.

        Returns:
            Team to use for the battle
        """
        if not self.active_teams:
            # Try to generate a team
            team = self.teambuilder_trainer.generate_team()
            if team:
                self.active_teams.append(team)
                self.stats.teams_generated += 1
            return team

        # Use curriculum for team selection if enabled
        if self.curriculum_enabled:
            try:
                return self.curriculum.select_team(
                    available_teams=self.active_teams,
                    sample_teams=self.sample_teams,
                )
            except Exception as e:
                logger.warning(f"Curriculum team selection failed: {e}")

        # Fallback: Round-robin selection
        idx = self.stats.total_battles % len(self.active_teams)
        return self.active_teams[idx]

    def get_opponent_type(self) -> OpponentType:
        """Get the type of opponent to use for the next battle.

        Uses curriculum learning to select opponent difficulty.

        Returns:
            OpponentType for the next battle
        """
        if self.curriculum_enabled:
            try:
                return self.curriculum.select_opponent_type()
            except Exception as e:
                logger.warning(f"Curriculum opponent selection failed: {e}")

        # Fallback: random/maxdamage mix
        import random
        return random.choice([OpponentType.RANDOM, OpponentType.MAXDAMAGE])

    def record_battle_outcome(
        self,
        team: Team,
        won: bool,
        turns: int,
        opponent_revealed: list[str],
        opponent_rating: float = 1000.0,
        opponent_type: str = "unknown",
    ) -> None:
        """Record the outcome of a battle.

        Updates both player and teambuilder trainers, plus curriculum.

        Args:
            team: The team that was used
            won: Whether the battle was won
            turns: Number of turns in the battle
            opponent_revealed: List of opponent Pokemon species revealed
            opponent_rating: Opponent's skill rating
            opponent_type: Type of opponent (for curriculum tracking)
        """
        # Update statistics
        self.stats.total_battles += 1
        if won:
            self.stats.player_wins += 1
        else:
            self.stats.player_losses += 1

        # Update player skill rating
        self.player_trainer.update_skill_rating(won, opponent_rating)
        self.stats.player_skill_rating = self.player_trainer.skill_rating

        # Record outcome for teambuilder
        elo_delta = self.stats.player_skill_rating - opponent_rating
        self.teambuilder_trainer.add_battle_outcome(
            team=team,
            won=won,
            turns=turns,
            opponent_revealed=opponent_revealed,
            elo_delta=elo_delta,
        )

        # Update curriculum learning
        if self.curriculum_enabled:
            team_id = str(id(team))
            self.curriculum.update(
                won=won,
                opponent_type=opponent_type,
                team_id=team_id,
            )

        # Log to TensorBoard
        if self.writer:
            self.writer.add_scalar(
                "joint/win_rate", self.stats.player_win_rate, self.stats.total_battles
            )
            self.writer.add_scalar(
                "joint/skill_rating",
                self.stats.player_skill_rating,
                self.stats.total_battles,
            )
            # Log curriculum state
            if self.curriculum_enabled:
                self.writer.add_scalar(
                    "curriculum/level",
                    self.curriculum.state.current_level.value,
                    self.stats.total_battles,
                )
                self.writer.add_scalar(
                    "curriculum/recent_win_rate",
                    self.curriculum.state.recent_win_rate,
                    self.stats.total_battles,
                )

    def update_player(self) -> dict[str, float]:
        """Perform a player training update.

        Returns:
            Training metrics
        """
        metrics = self.player_trainer.update()
        self.stats.total_player_updates += 1

        if self.writer and metrics:
            for key, value in metrics.items():
                self.writer.add_scalar(
                    f"joint/player/{key}",
                    value,
                    self.stats.total_player_updates,
                )

        return metrics

    def update_teambuilder(self) -> dict[str, float]:
        """Perform a teambuilder training update.

        Returns:
            Training metrics
        """
        metrics = self.teambuilder_trainer.update()
        self.stats.total_teambuilder_updates += 1

        # Update stats
        if "evaluator_accuracy" in metrics.get("evaluator/", {}):
            self.stats.evaluator_accuracy = metrics["evaluator/evaluator_accuracy"]

        if self.writer and metrics:
            for key, value in metrics.items():
                self.writer.add_scalar(
                    f"joint/teambuilder/{key}",
                    value,
                    self.stats.total_teambuilder_updates,
                )

        return metrics

    def rotate_teams(self) -> None:
        """Rotate teams in the active pool.

        Replaces worst-performing teams with newly generated ones.
        """
        if len(self.active_teams) < 2:
            return

        # Get win rates for current teams
        team_win_rates = []
        for team in self.active_teams:
            win_rate = self.teambuilder_trainer.outcome_buffer.get_team_win_rate(team)
            team_win_rates.append((team, win_rate or 0.5))

        # Sort by win rate (worst first)
        team_win_rates.sort(key=lambda x: x[1])

        # Replace bottom 20% with new teams
        num_to_replace = max(1, len(self.active_teams) // 5)
        new_teams = []

        for _ in range(num_to_replace):
            new_team = self.teambuilder_trainer.generate_team()
            if new_team:
                new_teams.append(new_team)
                self.stats.teams_generated += 1

        # Replace worst teams
        for i, new_team in enumerate(new_teams):
            if i < len(team_win_rates):
                old_team = team_win_rates[i][0]
                idx = self.active_teams.index(old_team)
                self.active_teams[idx] = new_team
                logger.debug(
                    f"Replaced team with win rate {team_win_rates[i][1]:.2f} "
                    f"with new generated team"
                )

        # Update best team win rate stat
        if team_win_rates:
            self.stats.best_team_win_rate = team_win_rates[-1][1]

    def training_step(self) -> dict[str, Any]:
        """Perform one training step (player update + optional teambuilder update).

        This is called after each rollout is collected. The feedback loop:
        1. Player update happens every step
        2. Teambuilder update happens periodically
        3. Team rotation happens when enough games have been played

        Returns:
            Combined metrics from this step
        """
        metrics = {}

        # Always update player after rollout
        player_metrics = self.update_player()
        metrics["player"] = player_metrics

        # Update teambuilder periodically
        if self.stats.total_player_updates % self.teambuilder_updates_per_phase == 0:
            teambuilder_metrics = self.update_teambuilder()
            metrics["teambuilder"] = teambuilder_metrics

        # Rotate teams when enough games have been played with each
        games_per_team = self.stats.total_battles / max(len(self.active_teams), 1)
        if games_per_team >= self.battles_per_team_before_rotate:
            self.rotate_teams()

        self.global_step += 1

        # Log overall stats
        if self.writer and self.global_step % self.config.log_interval == 0:
            self.writer.add_scalar(
                "joint/total_battles", self.stats.total_battles, self.global_step
            )
            self.writer.add_scalar(
                "joint/teams_generated", self.stats.teams_generated, self.global_step
            )
            self.writer.add_scalar(
                "joint/best_team_win_rate",
                self.stats.best_team_win_rate,
                self.global_step,
            )
            self.writer.add_scalar(
                "joint/active_teams", len(self.active_teams), self.global_step
            )

        return metrics

    def save_checkpoint(self, path: str | Path) -> None:
        """Save joint training checkpoint.

        Args:
            path: Directory to save checkpoint
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save player checkpoint
        self.player_trainer.save_checkpoint(path / "player.pt")

        # Save teambuilder checkpoint
        self.teambuilder_trainer.save_checkpoint(path / "teambuilder.pt")

        # Save joint state
        joint_state = {
            "global_step": self.global_step,
            "stats": {
                "total_battles": self.stats.total_battles,
                "total_player_updates": self.stats.total_player_updates,
                "total_teambuilder_updates": self.stats.total_teambuilder_updates,
                "player_wins": self.stats.player_wins,
                "player_losses": self.stats.player_losses,
                "player_skill_rating": self.stats.player_skill_rating,
                "teams_generated": self.stats.teams_generated,
                "best_team_win_rate": self.stats.best_team_win_rate,
                "evaluator_accuracy": self.stats.evaluator_accuracy,
            },
            "config": self.config.to_dict(),
            "curriculum": self.curriculum.get_state_dict() if self.curriculum_enabled else None,
            "curriculum_enabled": self.curriculum_enabled,
        }
        torch.save(joint_state, path / "joint_state.pt")

        logger.info(f"Saved joint checkpoint to {path}")

    def load_checkpoint(self, path: str | Path) -> None:
        """Load joint training checkpoint.

        Args:
            path: Directory containing checkpoint
        """
        path = Path(path)

        # Load player checkpoint
        player_path = path / "player.pt"
        if player_path.exists():
            self.player_trainer.load_checkpoint(player_path)

        # Load teambuilder checkpoint
        teambuilder_path = path / "teambuilder.pt"
        if teambuilder_path.exists():
            self.teambuilder_trainer.load_checkpoint(teambuilder_path)

        # Load joint state
        joint_path = path / "joint_state.pt"
        if joint_path.exists():
            joint_state = torch.load(joint_path, map_location=self.device, weights_only=False)
            self.global_step = joint_state.get("global_step", 0)

            stats = joint_state.get("stats", {})
            self.stats.total_battles = stats.get("total_battles", 0)
            self.stats.total_player_updates = stats.get("total_player_updates", 0)
            self.stats.total_teambuilder_updates = stats.get(
                "total_teambuilder_updates", 0
            )
            self.stats.player_wins = stats.get("player_wins", 0)
            self.stats.player_losses = stats.get("player_losses", 0)
            self.stats.player_skill_rating = stats.get("player_skill_rating", 1000.0)
            self.stats.teams_generated = stats.get("teams_generated", 0)

            # Load curriculum state
            self.curriculum_enabled = joint_state.get("curriculum_enabled", True)
            curriculum_state = joint_state.get("curriculum")
            if curriculum_state and self.curriculum_enabled:
                self.curriculum.load_state_dict(curriculum_state)
                logger.info(
                    f"Loaded curriculum state: level={self.curriculum.state.current_level.name}"
                )
            self.stats.best_team_win_rate = stats.get("best_team_win_rate", 0.0)
            self.stats.evaluator_accuracy = stats.get("evaluator_accuracy", 0.5)

        logger.info(f"Loaded joint checkpoint from {path}")

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of current training state.

        Returns:
            Dictionary with training statistics
        """
        summary = {
            "global_step": self.global_step,
            "total_battles": self.stats.total_battles,
            "player_win_rate": self.stats.player_win_rate,
            "player_skill_rating": self.stats.player_skill_rating,
            "teams_generated": self.stats.teams_generated,
            "best_team_win_rate": self.stats.best_team_win_rate,
            "evaluator_accuracy": self.stats.evaluator_accuracy,
            "active_teams": len(self.active_teams),
            "player_updates": self.stats.total_player_updates,
            "teambuilder_updates": self.stats.total_teambuilder_updates,
        }

        # Add curriculum info
        if self.curriculum_enabled:
            summary["curriculum_level"] = self.curriculum.state.current_level.name
            summary["curriculum_recent_wr"] = self.curriculum.state.recent_win_rate
            if hasattr(self.curriculum, 'active_strategy'):
                summary["curriculum_strategy"] = self.curriculum.active_strategy

        return summary


def create_joint_trainer(
    config: OUTrainingConfig,
    player_network: OUPlayerNetwork,
    state_encoder: OUStateEncoder,
    team_generator: TeamGenerator,
    team_evaluator: TeamEvaluator,
    device: torch.device | None = None,
    curriculum_strategy: str = "adaptive",
    curriculum_config: CurriculumConfig | None = None,
) -> JointTrainer:
    """Factory function to create a JointTrainer with all components.

    Args:
        config: Training configuration
        player_network: OU player network
        state_encoder: State encoder for battles
        team_generator: Team generator
        team_evaluator: Team evaluator
        device: Device for training
        curriculum_strategy: One of "progressive", "matchup", "complexity", "adaptive"
        curriculum_config: Optional curriculum configuration

    Returns:
        Configured JointTrainer
    """
    device = device or torch.device(config.device)

    # Create player trainer
    player_trainer = OUPlayerTrainer(
        network=player_network,
        encoder=state_encoder,
        config=config,
        device=device,
    )

    # Create teambuilder trainer
    teambuilder_trainer = TeambuilderTrainer(
        generator=team_generator,
        evaluator=team_evaluator,
        config=config,
        device=device,
    )

    # Create joint trainer
    return JointTrainer(
        player_trainer=player_trainer,
        teambuilder_trainer=teambuilder_trainer,
        config=config,
        device=device,
        curriculum_strategy=curriculum_strategy,
        curriculum_config=curriculum_config,
    )
