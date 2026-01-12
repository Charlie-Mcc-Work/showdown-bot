"""Experience buffer for OU training.

Stores transitions from battles for training the player network,
and team-outcome pairs for training the teambuilder.
"""

from dataclasses import dataclass
from typing import Iterator
import random

import torch
import numpy as np

from showdown_bot.ou.player.state_encoder import OUEncodedState
from showdown_bot.ou.shared.data_loader import Team


@dataclass
class OUTransition:
    """A single transition in an OU battle.

    Attributes:
        state: Encoded battle state
        action: Action taken (move/switch index)
        reward: Immediate reward
        next_state: Next encoded state (or None if terminal)
        done: Whether episode ended
        log_prob: Log probability of action under policy
        value: Value estimate at state
        team: The team being used in this battle
    """

    state: OUEncodedState
    action: int
    reward: float
    next_state: OUEncodedState | None
    done: bool
    log_prob: float
    value: float
    team: Team | None = None


@dataclass
class TeamOutcome:
    """Outcome of a battle with a specific team.

    Used for training the teambuilder based on actual performance.
    """

    team: Team
    won: bool
    turns: int
    opponent_revealed: list[str]  # Opponent Pokemon species revealed
    elo_delta: float  # Rating change


class OUExperienceBuffer:
    """Buffer for storing OU battle transitions.

    Supports both online (PPO-style) and offline (replay buffer) training.
    """

    def __init__(
        self,
        max_size: int = 100000,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ):
        """Initialize the buffer.

        Args:
            max_size: Maximum number of transitions to store
            gamma: Discount factor for returns
            gae_lambda: GAE lambda for advantage estimation
        """
        self.max_size = max_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        # Transition storage
        self.transitions: list[OUTransition] = []
        self.episode_starts: list[int] = [0]  # Track episode boundaries

        # Computed values (set by compute_returns)
        self.returns: list[float] = []
        self.advantages: list[float] = []

    def add(self, transition: OUTransition) -> None:
        """Add a transition to the buffer."""
        if len(self.transitions) >= self.max_size:
            # Remove oldest transitions
            self.transitions.pop(0)
            # Adjust episode starts
            self.episode_starts = [max(0, s - 1) for s in self.episode_starts]
            if self.episode_starts[0] != 0:
                self.episode_starts.insert(0, 0)

        self.transitions.append(transition)

        if transition.done:
            self.episode_starts.append(len(self.transitions))

    def compute_returns_and_advantages(
        self,
        last_value: float = 0.0,
    ) -> None:
        """Compute returns and advantages using GAE.

        Args:
            last_value: Value estimate for the state after last transition
        """
        self.returns = []
        self.advantages = []

        # Process each episode separately
        for i in range(len(self.episode_starts) - 1):
            start = self.episode_starts[i]
            end = self.episode_starts[i + 1]
            episode = self.transitions[start:end]

            # Compute GAE
            gae = 0.0
            episode_advantages = []
            episode_returns = []

            for t in reversed(range(len(episode))):
                trans = episode[t]

                if t == len(episode) - 1:
                    # Last step in episode
                    if trans.done:
                        next_value = 0.0
                    else:
                        next_value = last_value
                else:
                    next_value = episode[t + 1].value

                delta = trans.reward + self.gamma * next_value - trans.value
                gae = delta + self.gamma * self.gae_lambda * gae * (1 - int(trans.done))

                episode_advantages.insert(0, gae)
                episode_returns.insert(0, gae + trans.value)

            self.advantages.extend(episode_advantages)
            self.returns.extend(episode_returns)

        # Normalize advantages
        if self.advantages:
            adv_tensor = torch.tensor(self.advantages)
            adv_mean = adv_tensor.mean()
            adv_std = adv_tensor.std() + 1e-8
            self.advantages = ((adv_tensor - adv_mean) / adv_std).tolist()

    def sample_batch(
        self,
        batch_size: int,
    ) -> tuple[list[OUTransition], list[float], list[float]]:
        """Sample a batch of transitions.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            (transitions, returns, advantages)
        """
        if len(self.transitions) < batch_size:
            indices = list(range(len(self.transitions)))
        else:
            indices = random.sample(range(len(self.transitions)), batch_size)

        transitions = [self.transitions[i] for i in indices]
        returns = [self.returns[i] for i in indices] if self.returns else [0.0] * len(indices)
        advantages = [self.advantages[i] for i in indices] if self.advantages else [0.0] * len(indices)

        return transitions, returns, advantages

    def iterate_batches(
        self,
        batch_size: int,
        shuffle: bool = True,
    ) -> Iterator[tuple[list[OUTransition], list[float], list[float]]]:
        """Iterate over all transitions in batches.

        Args:
            batch_size: Batch size
            shuffle: Whether to shuffle before iterating

        Yields:
            (transitions, returns, advantages) batches
        """
        indices = list(range(len(self.transitions)))
        if shuffle:
            random.shuffle(indices)

        for start in range(0, len(indices), batch_size):
            end = min(start + batch_size, len(indices))
            batch_indices = indices[start:end]

            transitions = [self.transitions[i] for i in batch_indices]
            returns = [self.returns[i] for i in batch_indices] if self.returns else [0.0] * len(batch_indices)
            advantages = [self.advantages[i] for i in batch_indices] if self.advantages else [0.0] * len(batch_indices)

            yield transitions, returns, advantages

    def clear(self) -> None:
        """Clear the buffer."""
        self.transitions = []
        self.episode_starts = [0]
        self.returns = []
        self.advantages = []

    def __len__(self) -> int:
        return len(self.transitions)


class TeamOutcomeBuffer:
    """Buffer for storing team outcomes.

    Used to track which teams perform well and update the teambuilder.
    """

    def __init__(self, max_size: int = 10000):
        """Initialize the buffer.

        Args:
            max_size: Maximum number of outcomes to store
        """
        self.max_size = max_size
        self.outcomes: list[TeamOutcome] = []

        # Team statistics
        self.team_stats: dict[str, dict] = {}  # team_hash -> stats

    def add(self, outcome: TeamOutcome) -> None:
        """Add a team outcome."""
        if len(self.outcomes) >= self.max_size:
            self.outcomes.pop(0)

        self.outcomes.append(outcome)

        # Update team statistics
        team_hash = self._hash_team(outcome.team)
        if team_hash not in self.team_stats:
            self.team_stats[team_hash] = {
                "wins": 0,
                "losses": 0,
                "total_turns": 0,
                "team": outcome.team,
            }

        stats = self.team_stats[team_hash]
        if outcome.won:
            stats["wins"] += 1
        else:
            stats["losses"] += 1
        stats["total_turns"] += outcome.turns

    def get_team_win_rate(self, team: Team) -> float | None:
        """Get win rate for a team.

        Returns None if team has not been played enough.
        """
        team_hash = self._hash_team(team)
        if team_hash not in self.team_stats:
            return None

        stats = self.team_stats[team_hash]
        total = stats["wins"] + stats["losses"]
        if total == 0:
            return None

        return stats["wins"] / total

    def get_best_teams(self, n: int = 10, min_games: int = 5) -> list[tuple[Team, float]]:
        """Get the best performing teams.

        Args:
            n: Number of teams to return
            min_games: Minimum games required for consideration

        Returns:
            List of (team, win_rate) tuples, sorted by win rate
        """
        teams_with_stats = []

        for team_hash, stats in self.team_stats.items():
            total = stats["wins"] + stats["losses"]
            if total >= min_games:
                win_rate = stats["wins"] / total
                teams_with_stats.append((stats["team"], win_rate))

        teams_with_stats.sort(key=lambda x: x[1], reverse=True)
        return teams_with_stats[:n]

    def sample_outcomes(
        self,
        batch_size: int,
        positive_ratio: float = 0.5,
    ) -> list[TeamOutcome]:
        """Sample outcomes for training.

        Args:
            batch_size: Number of outcomes to sample
            positive_ratio: Target ratio of wins in sample

        Returns:
            List of sampled outcomes
        """
        wins = [o for o in self.outcomes if o.won]
        losses = [o for o in self.outcomes if not o.won]

        n_wins = int(batch_size * positive_ratio)
        n_losses = batch_size - n_wins

        sampled = []
        if wins:
            sampled.extend(random.choices(wins, k=min(n_wins, len(wins))))
        if losses:
            sampled.extend(random.choices(losses, k=min(n_losses, len(losses))))

        random.shuffle(sampled)
        return sampled

    def _hash_team(self, team: Team) -> str:
        """Create a hash for a team based on Pokemon species."""
        species = sorted([p.species for p in team.pokemon])
        return "|".join(species)

    def __len__(self) -> int:
        return len(self.outcomes)
