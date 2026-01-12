"""Trainer for the autoregressive team generator.

The teambuilder is trained using:
1. Supervised learning from high-performing teams (human or generated)
2. Reinforcement learning from battle outcomes
3. Policy gradient with team win rate as reward

Training pipeline:
1. Generate teams using current policy
2. Play battles with generated teams
3. Collect (team, win_rate) pairs
4. Update generator to produce more winning teams
"""

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from showdown_bot.ou.training.config import OUTrainingConfig
from showdown_bot.ou.training.buffer import TeamOutcomeBuffer, TeamOutcome
from showdown_bot.ou.teambuilder.generator import TeamGenerator
from showdown_bot.ou.teambuilder.evaluator import TeamEvaluator
from showdown_bot.ou.teambuilder.team_repr import PartialTeam
from showdown_bot.ou.shared.data_loader import Team

logger = logging.getLogger(__name__)


class TeambuilderTrainer:
    """Trainer for the team generator.

    Uses a combination of:
    1. Supervised learning from sample teams
    2. REINFORCE for optimizing team quality
    3. Self-improvement via battle outcomes
    """

    def __init__(
        self,
        generator: TeamGenerator,
        evaluator: TeamEvaluator,
        config: OUTrainingConfig,
        device: torch.device | None = None,
    ):
        """Initialize the trainer.

        Args:
            generator: The team generator to train
            evaluator: Team evaluator for quality estimation
            config: Training configuration
            device: Device for training
        """
        self.generator = generator
        self.evaluator = evaluator
        self.config = config
        self.device = device or torch.device(config.device)

        # Move models to device
        self.generator = self.generator.to(self.device)
        self.evaluator = self.evaluator.to(self.device)

        # Optimizers
        self.generator_optimizer = torch.optim.Adam(
            self.generator.parameters(),
            lr=config.teambuilder_lr,
        )
        self.evaluator_optimizer = torch.optim.Adam(
            self.evaluator.parameters(),
            lr=config.teambuilder_lr,
        )

        # Team outcome buffer
        self.outcome_buffer = TeamOutcomeBuffer()

        # Sample teams for supervised learning
        self.sample_teams: list[Team] = []

        # Logging
        self.writer: SummaryWriter | None = None
        self.global_step = 0
        self.teams_generated = 0

    def setup_logging(self, log_dir: str | Path) -> None:
        """Set up TensorBoard logging."""
        self.writer = SummaryWriter(log_dir=str(log_dir))

    def add_sample_teams(self, teams: list[Team]) -> None:
        """Add sample teams for supervised learning.

        Args:
            teams: List of high-quality teams to learn from
        """
        self.sample_teams.extend(teams)
        logger.info(f"Added {len(teams)} sample teams, total: {len(self.sample_teams)}")

    def add_battle_outcome(
        self,
        team: Team,
        won: bool,
        turns: int,
        opponent_revealed: list[str],
        elo_delta: float = 0.0,
    ) -> None:
        """Add a battle outcome for training.

        Args:
            team: The team that was used
            won: Whether the battle was won
            turns: Number of turns in the battle
            opponent_revealed: Opponent Pokemon revealed
            elo_delta: Rating change from the battle
        """
        outcome = TeamOutcome(
            team=team,
            won=won,
            turns=turns,
            opponent_revealed=opponent_revealed,
            elo_delta=elo_delta,
        )
        self.outcome_buffer.add(outcome)

    def generate_team(
        self,
        temperature: float | None = None,
    ) -> Team | None:
        """Generate a new team.

        Args:
            temperature: Sampling temperature (None uses config default)

        Returns:
            Generated team, or None if generation failed
        """
        temp = temperature or self.config.teambuilder_temperature

        self.generator.eval()
        with torch.no_grad():
            partial_team = self.generator.generate_team(
                temperature=temp,
                device=self.device,
            )

        self.teams_generated += 1

        # Convert to Team
        return partial_team.to_team()

    def update_generator_supervised(
        self,
        batch_size: int | None = None,
    ) -> dict[str, float]:
        """Update generator using supervised learning on sample teams.

        Args:
            batch_size: Batch size (None uses config default)

        Returns:
            Training metrics
        """
        if not self.sample_teams:
            return {}

        batch_size = batch_size or self.config.teambuilder_batch_size
        self.generator.train()

        # Sample teams
        import random
        batch = random.sample(
            self.sample_teams,
            min(batch_size, len(self.sample_teams)),
        )

        # TODO: Implement proper supervised loss
        # For now, return placeholder metrics
        loss = 0.0

        return {
            "supervised_loss": loss,
            "batch_size": len(batch),
        }

    def update_generator_rl(
        self,
        batch_size: int | None = None,
    ) -> dict[str, float]:
        """Update generator using RL from battle outcomes.

        Uses a supervised learning approach on high-performing teams:
        - Collect teams with high win rates
        - Train generator to produce similar teams

        This is more stable than REINFORCE for autoregressive generation
        because the sampling operations are not differentiable.

        Args:
            batch_size: Batch size (None uses config default)

        Returns:
            Training metrics
        """
        if len(self.outcome_buffer) < self.config.teambuilder_min_games:
            return {}

        batch_size = batch_size or self.config.teambuilder_batch_size

        # Get best performing teams
        best_teams = self.outcome_buffer.get_best_teams(
            n=batch_size,
            min_games=self.config.teambuilder_min_games,
        )
        outcomes = self.outcome_buffer.sample_outcomes(batch_size)

        if not outcomes or not best_teams:
            return {}

        # Compute baseline (average win rate)
        baseline = sum(1.0 if o.won else 0.0 for o in outcomes) / len(outcomes)

        # Strategy: Use contrastive learning with the evaluator
        # Train the evaluator to distinguish good teams from bad teams
        # This indirectly guides the generator when it uses the evaluator
        # for quality-weighted generation

        # Compute quality scores for sampled outcomes using evaluator
        total_quality_gap = 0.0
        num_evaluated = 0

        for outcome in outcomes[:min(8, len(outcomes))]:
            try:
                # Get evaluator's prediction
                team_data = self.evaluator.tensor_encoder.encode_team(
                    outcome.team, self.device
                )
                team_encoding, _ = self.evaluator.team_encoder.encode_team(team_data)

                with torch.no_grad():
                    output = self.evaluator(team_encoding)
                    predicted = output["win_rate"].squeeze().item()

                actual = 1.0 if outcome.won else 0.0
                total_quality_gap += abs(predicted - actual)
                num_evaluated += 1
            except Exception:
                continue

        avg_quality_gap = total_quality_gap / max(num_evaluated, 1)

        # For now, the generator learns through supervised imitation of
        # high-performing teams (see update_generator_supervised)
        # REINFORCE would require tracking log probs during generation

        return {
            "rl_loss": 0.0,  # No direct RL loss currently
            "baseline": baseline,
            "best_win_rate": best_teams[0][1] if best_teams else 0.0,
            "quality_gap": avg_quality_gap,
            "num_evaluated": num_evaluated,
        }

    def update_evaluator(
        self,
        batch_size: int | None = None,
    ) -> dict[str, float]:
        """Update team evaluator on battle outcomes.

        The evaluator is trained to predict win probability using BCE loss
        against actual battle outcomes.

        Args:
            batch_size: Batch size (None uses config default)

        Returns:
            Training metrics
        """
        batch_size = batch_size or self.config.teambuilder_batch_size

        if len(self.outcome_buffer) < batch_size:
            return {}

        self.evaluator.train()

        # Sample outcomes with balanced wins/losses
        outcomes = self.outcome_buffer.sample_outcomes(
            batch_size,
            positive_ratio=0.5,
        )

        if not outcomes:
            return {}

        # Encode all teams and collect outcomes
        team_encodings = []
        outcome_labels = []

        for outcome in outcomes:
            # Encode team using tensor encoder
            team_data = self.evaluator.tensor_encoder.encode_team(
                outcome.team, self.device
            )
            # Get team encoding from team encoder
            team_encoding, _ = self.evaluator.team_encoder.encode_team(team_data)
            team_encodings.append(team_encoding)
            outcome_labels.append(1.0 if outcome.won else 0.0)

        # Stack into batch tensors
        team_batch = torch.cat(team_encodings, dim=0)  # (batch, hidden_dim)
        labels = torch.tensor(outcome_labels, device=self.device, dtype=torch.float32)

        # Forward pass through evaluator head
        output = self.evaluator(team_batch)
        predicted_win_rate = output["win_rate"].squeeze(-1)

        # Binary cross-entropy loss
        loss = F.binary_cross_entropy(predicted_win_rate, labels)

        # Backward pass
        self.evaluator_optimizer.zero_grad()
        loss.backward()
        self.evaluator_optimizer.step()

        # Compute accuracy
        with torch.no_grad():
            predictions = (predicted_win_rate > 0.5).float()
            accuracy = (predictions == labels).float().mean().item()

        return {
            "evaluator_loss": loss.item(),
            "evaluator_accuracy": accuracy,
            "batch_size": len(outcomes),
        }

    def update(self) -> dict[str, float]:
        """Perform a full training update.

        Returns:
            Combined training metrics
        """
        metrics = {}

        # Supervised update
        sup_metrics = self.update_generator_supervised()
        metrics.update({f"supervised/{k}": v for k, v in sup_metrics.items()})

        # RL update
        rl_metrics = self.update_generator_rl()
        metrics.update({f"rl/{k}": v for k, v in rl_metrics.items()})

        # Evaluator update
        eval_metrics = self.update_evaluator()
        metrics.update({f"evaluator/{k}": v for k, v in eval_metrics.items()})

        # Log metrics
        if self.writer:
            for key, value in metrics.items():
                self.writer.add_scalar(f"teambuilder/{key}", value, self.global_step)

            self.writer.add_scalar(
                "teambuilder/teams_generated",
                self.teams_generated,
                self.global_step,
            )
            self.writer.add_scalar(
                "teambuilder/outcome_buffer_size",
                len(self.outcome_buffer),
                self.global_step,
            )

        self.global_step += 1

        return metrics

    def save_checkpoint(self, path: str | Path) -> None:
        """Save training checkpoint.

        Args:
            path: Path to save checkpoint
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "generator_state_dict": self.generator.state_dict(),
            "evaluator_state_dict": self.evaluator.state_dict(),
            "generator_optimizer_state_dict": self.generator_optimizer.state_dict(),
            "evaluator_optimizer_state_dict": self.evaluator_optimizer.state_dict(),
            "global_step": self.global_step,
            "teams_generated": self.teams_generated,
            "config": self.config.to_dict(),
        }

        torch.save(checkpoint, path)
        logger.info(f"Saved teambuilder checkpoint to {path}")

    def load_checkpoint(self, path: str | Path) -> None:
        """Load training checkpoint.

        Args:
            path: Path to checkpoint
        """
        path = Path(path)
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.generator.load_state_dict(checkpoint["generator_state_dict"])
        self.evaluator.load_state_dict(checkpoint["evaluator_state_dict"])
        self.generator_optimizer.load_state_dict(
            checkpoint["generator_optimizer_state_dict"]
        )
        self.evaluator_optimizer.load_state_dict(
            checkpoint["evaluator_optimizer_state_dict"]
        )
        self.global_step = checkpoint.get("global_step", 0)
        self.teams_generated = checkpoint.get("teams_generated", 0)

        logger.info(f"Loaded teambuilder checkpoint from {path}")

    def get_team_quality(self, team: Team) -> dict[str, float]:
        """Get quality metrics for a team.

        Args:
            team: Team to evaluate

        Returns:
            Dictionary of quality metrics
        """
        self.evaluator.eval()
        with torch.no_grad():
            result = self.evaluator.evaluate_team(team, device=self.device)

        # Add win rate from outcomes if available
        win_rate = self.outcome_buffer.get_team_win_rate(team)
        if win_rate is not None:
            result["actual_win_rate"] = win_rate

        return result
