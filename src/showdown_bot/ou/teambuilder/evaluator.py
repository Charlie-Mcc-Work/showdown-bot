"""Team evaluator for predicting team quality.

The evaluator predicts expected win rate for a given team. This is used to:
1. Guide team generation (MCTS-style search)
2. Filter generated teams
3. Provide training signal for the generator

Training data comes from actual battles played by the trained player.
"""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from showdown_bot.ou.shared.embeddings import SharedEmbeddings
from showdown_bot.ou.shared.encoders import FullTeamEncoder
from showdown_bot.ou.shared.data_loader import Team


class TeamEvaluator(nn.Module):
    """Predicts expected performance of a team.

    Architecture:
    - Encode team using FullTeamEncoder
    - Pass through evaluation head
    - Output: Expected win rate (0-1) and uncertainty

    Can optionally condition on expected opponent distribution
    (e.g., current metagame usage stats).
    """

    def __init__(
        self,
        shared_embeddings: SharedEmbeddings,
        hidden_dim: int = 512,
        num_layers: int = 2,
    ):
        super().__init__()

        self.embeddings = shared_embeddings
        self.hidden_dim = hidden_dim

        # Team encoder
        self.team_encoder = FullTeamEncoder(
            shared_embeddings,
            pokemon_hidden=hidden_dim,
            team_hidden=hidden_dim,
        )

        # Evaluation head
        self.eval_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )

        # Output heads
        self.win_rate_head = nn.Linear(hidden_dim // 2, 1)  # Expected win rate
        self.uncertainty_head = nn.Linear(hidden_dim // 2, 1)  # Epistemic uncertainty

        # Auxiliary heads for interpretability
        self.archetype_head = nn.Linear(hidden_dim // 2, 10)  # Team archetype
        self.balance_head = nn.Linear(hidden_dim // 2, 3)  # Offense/Balance/Stall

    def forward(
        self,
        team_encoding: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Evaluate a team.

        Args:
            team_encoding: (batch, hidden_dim) Encoded team from FullTeamEncoder

        Returns:
            Dict with:
            - win_rate: (batch, 1) Expected win rate [0, 1]
            - uncertainty: (batch, 1) Prediction uncertainty
            - archetype_logits: (batch, 10) Team archetype probabilities
            - balance_logits: (batch, 3) Offense/Balance/Stall probs
        """
        hidden = self.eval_head(team_encoding)

        win_rate = torch.sigmoid(self.win_rate_head(hidden))
        uncertainty = F.softplus(self.uncertainty_head(hidden))
        archetype_logits = self.archetype_head(hidden)
        balance_logits = self.balance_head(hidden)

        return {
            "win_rate": win_rate,
            "uncertainty": uncertainty,
            "archetype_logits": archetype_logits,
            "balance_logits": balance_logits,
        }

    def evaluate_team(
        self,
        team: Team,
        device: torch.device | None = None,
    ) -> dict[str, float]:
        """Evaluate a complete team.

        Args:
            team: The team to evaluate
            device: Device for computation

        Returns:
            Dict with evaluation metrics
        """
        device = device or next(self.parameters()).device

        # Encode team
        # TODO: Convert Team to tensor representation
        team_encoding = torch.zeros(1, self.hidden_dim, device=device)

        # Get evaluation
        self.eval()
        with torch.no_grad():
            output = self.forward(team_encoding)

        return {
            "win_rate": output["win_rate"].item(),
            "uncertainty": output["uncertainty"].item(),
            "archetype": output["archetype_logits"].argmax(dim=-1).item(),
            "balance": output["balance_logits"].argmax(dim=-1).item(),
        }


class MatchupPredictor(nn.Module):
    """Predicts win probability for a specific matchup.

    Given our team and opponent's team (or partial information),
    predicts win probability. Useful for:
    1. Team preview decisions
    2. Understanding team weaknesses
    3. More accurate team evaluation
    """

    def __init__(
        self,
        hidden_dim: int = 512,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        # Combine two team encodings
        self.matchup_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        our_team_encoding: torch.Tensor,
        opp_team_encoding: torch.Tensor,
    ) -> torch.Tensor:
        """Predict win probability for matchup.

        Args:
            our_team_encoding: (batch, hidden_dim) Our team encoding
            opp_team_encoding: (batch, hidden_dim) Opponent team encoding

        Returns:
            (batch, 1) Win probability [0, 1]
        """
        combined = torch.cat([our_team_encoding, opp_team_encoding], dim=-1)
        logit = self.matchup_encoder(combined)
        return torch.sigmoid(logit)


class CoverageAnalyzer(nn.Module):
    """Analyzes type coverage and weaknesses of a team.

    Produces interpretable metrics:
    - Types team can hit super-effectively
    - Types team resists
    - Types team is weak to
    - Overall coverage score
    """

    def __init__(self, hidden_dim: int = 512):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_types = 18

        # Coverage heads
        self.offensive_coverage = nn.Linear(hidden_dim, self.num_types)
        self.defensive_coverage = nn.Linear(hidden_dim, self.num_types)
        self.weakness_detector = nn.Linear(hidden_dim, self.num_types)

    def forward(
        self,
        team_encoding: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Analyze team coverage.

        Args:
            team_encoding: (batch, hidden_dim) Encoded team

        Returns:
            Dict with:
            - offensive: (batch, 18) Offensive coverage per type
            - defensive: (batch, 18) Defensive coverage per type
            - weaknesses: (batch, 18) Weakness severity per type
        """
        return {
            "offensive": torch.sigmoid(self.offensive_coverage(team_encoding)),
            "defensive": torch.sigmoid(self.defensive_coverage(team_encoding)),
            "weaknesses": torch.sigmoid(self.weakness_detector(team_encoding)),
        }


class TeamEvaluatorTrainer:
    """Training loop for the team evaluator.

    Trains on (team, outcome) pairs collected from battles.
    """

    def __init__(
        self,
        evaluator: TeamEvaluator,
        learning_rate: float = 1e-4,
        device: torch.device | None = None,
    ):
        self.evaluator = evaluator
        self.device = device or torch.device("cpu")

        self.optimizer = torch.optim.Adam(
            evaluator.parameters(),
            lr=learning_rate,
        )

        # Training history
        self.train_losses: list[float] = []

    def train_step(
        self,
        team_encodings: torch.Tensor,
        outcomes: torch.Tensor,
    ) -> float:
        """Single training step.

        Args:
            team_encodings: (batch, hidden_dim) Encoded teams
            outcomes: (batch,) Win/loss outcomes (1.0 = win, 0.0 = loss)

        Returns:
            Loss value
        """
        self.evaluator.train()
        self.optimizer.zero_grad()

        # Forward pass
        output = self.evaluator(team_encodings)
        predicted_win_rate = output["win_rate"].squeeze(-1)

        # Binary cross-entropy loss
        loss = F.binary_cross_entropy(predicted_win_rate, outcomes)

        # Backward pass
        loss.backward()
        self.optimizer.step()

        loss_val = loss.item()
        self.train_losses.append(loss_val)

        return loss_val

    def add_game_result(
        self,
        team: Team,
        won: bool,
    ) -> None:
        """Add a game result to the training buffer.

        Args:
            team: The team that was played
            won: Whether we won the game
        """
        # TODO: Implement experience buffer and batch training
        pass
