"""Curriculum learning strategies for OU training.

Curriculum learning helps the agent learn more effectively by:
1. Starting with easier tasks and progressively increasing difficulty
2. Focusing on specific matchups or weaknesses
3. Adapting training based on current performance

Available strategies:
- ProgressiveDifficulty: Ramps up opponent strength based on win rate
- MatchupFocus: Tracks win rates per opponent type and focuses on weaknesses
- TeamComplexity: Starts with simpler teams and increases diversity
- Adaptive: Combines strategies based on training phase

Usage:
    curriculum = AdaptiveCurriculum(config)

    # During training loop
    opponent_type = curriculum.select_opponent(current_win_rate)
    team = curriculum.get_training_team(team_pool, current_skill)
    curriculum.update(battle_outcome)
"""

import logging
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from showdown_bot.ou.shared.data_loader import Team

logger = logging.getLogger(__name__)


class OpponentType(Enum):
    """Types of opponents available for training."""
    RANDOM = "random"          # Random move selection
    MAXDAMAGE = "maxdamage"    # Always picks highest damage move
    HEURISTIC = "heuristic"    # Rule-based intelligent play
    SELF_PLAY = "self_play"    # Historical self-play opponents
    LADDER = "ladder"          # Actual ladder opponents (future)


class DifficultyLevel(Enum):
    """Difficulty levels for curriculum progression."""
    BEGINNER = 1      # Random opponents only
    EASY = 2          # Mix of random and maxdamage
    MEDIUM = 3        # Maxdamage and weak self-play
    HARD = 4          # Mixed self-play opponents
    EXPERT = 5        # Strong self-play and heuristic


@dataclass
class CurriculumState:
    """Tracks the current state of curriculum learning."""

    # Overall progress
    total_battles: int = 0
    current_level: DifficultyLevel = DifficultyLevel.BEGINNER

    # Performance tracking
    recent_wins: int = 0
    recent_losses: int = 0
    window_size: int = 50

    # Per-opponent tracking
    opponent_wins: dict = field(default_factory=dict)
    opponent_losses: dict = field(default_factory=dict)

    # Team tracking
    team_usage: dict = field(default_factory=dict)
    team_scores: dict = field(default_factory=dict)

    @property
    def recent_win_rate(self) -> float:
        """Win rate over recent window."""
        total = self.recent_wins + self.recent_losses
        if total == 0:
            return 0.5
        return self.recent_wins / total

    def get_opponent_win_rate(self, opponent_type: str) -> float:
        """Get win rate against specific opponent type."""
        wins = self.opponent_wins.get(opponent_type, 0)
        losses = self.opponent_losses.get(opponent_type, 0)
        total = wins + losses
        if total == 0:
            return 0.5
        return wins / total

    def record_battle(
        self,
        won: bool,
        opponent_type: str,
        team_id: str | None = None,
    ) -> None:
        """Record a battle outcome."""
        self.total_battles += 1

        # Update recent window
        if won:
            self.recent_wins += 1
        else:
            self.recent_losses += 1

        # Decay window to keep it at window_size
        total_recent = self.recent_wins + self.recent_losses
        if total_recent > self.window_size:
            decay = 0.9
            self.recent_wins = int(self.recent_wins * decay)
            self.recent_losses = int(self.recent_losses * decay)

        # Update per-opponent stats
        if opponent_type not in self.opponent_wins:
            self.opponent_wins[opponent_type] = 0
            self.opponent_losses[opponent_type] = 0

        if won:
            self.opponent_wins[opponent_type] += 1
        else:
            self.opponent_losses[opponent_type] += 1

        # Update team tracking
        if team_id:
            if team_id not in self.team_usage:
                self.team_usage[team_id] = 0
                self.team_scores[team_id] = []
            self.team_usage[team_id] += 1
            self.team_scores[team_id].append(1.0 if won else 0.0)
            # Keep only recent scores
            if len(self.team_scores[team_id]) > 20:
                self.team_scores[team_id] = self.team_scores[team_id][-20:]

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "total_battles": self.total_battles,
            "current_level": self.current_level.value,
            "recent_wins": self.recent_wins,
            "recent_losses": self.recent_losses,
            "window_size": self.window_size,
            "opponent_wins": dict(self.opponent_wins),
            "opponent_losses": dict(self.opponent_losses),
            "team_usage": dict(self.team_usage),
            "team_scores": {k: list(v) for k, v in self.team_scores.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CurriculumState":
        """Deserialize from dictionary."""
        state = cls()
        state.total_battles = d.get("total_battles", 0)
        state.current_level = DifficultyLevel(d.get("current_level", 1))
        state.recent_wins = d.get("recent_wins", 0)
        state.recent_losses = d.get("recent_losses", 0)
        state.window_size = d.get("window_size", 50)
        state.opponent_wins = d.get("opponent_wins", {})
        state.opponent_losses = d.get("opponent_losses", {})
        state.team_usage = d.get("team_usage", {})
        state.team_scores = {k: list(v) for k, v in d.get("team_scores", {}).items()}
        return state


@dataclass
class CurriculumConfig:
    """Configuration for curriculum learning."""

    # Difficulty progression
    promotion_threshold: float = 0.65  # Win rate to advance
    demotion_threshold: float = 0.35   # Win rate to go back
    min_battles_per_level: int = 50    # Min battles before level change

    # Opponent selection
    random_opponent_prob: float = 0.1  # Always have some exploration
    focus_on_weakness: bool = True     # Extra training on weak matchups
    weakness_threshold: float = 0.4    # Win rate below this is "weak"

    # Team selection
    team_diversity_weight: float = 0.3  # Weight for underused teams
    team_performance_weight: float = 0.7  # Weight for high-performing teams
    min_team_games: int = 5            # Min games before team evaluation

    # Complexity progression
    start_with_sample_teams: bool = True  # Use sample teams initially
    generated_team_threshold: int = 100   # Battles before using generated

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "promotion_threshold": self.promotion_threshold,
            "demotion_threshold": self.demotion_threshold,
            "min_battles_per_level": self.min_battles_per_level,
            "random_opponent_prob": self.random_opponent_prob,
            "focus_on_weakness": self.focus_on_weakness,
            "weakness_threshold": self.weakness_threshold,
            "team_diversity_weight": self.team_diversity_weight,
            "team_performance_weight": self.team_performance_weight,
            "min_team_games": self.min_team_games,
            "start_with_sample_teams": self.start_with_sample_teams,
            "generated_team_threshold": self.generated_team_threshold,
        }


class CurriculumStrategy(ABC):
    """Base class for curriculum learning strategies."""

    def __init__(self, config: CurriculumConfig | None = None):
        self.config = config or CurriculumConfig()
        self.state = CurriculumState()

    @abstractmethod
    def select_opponent_type(self) -> OpponentType:
        """Select the type of opponent for the next battle."""
        pass

    @abstractmethod
    def select_team(
        self,
        available_teams: list[Team],
        sample_teams: list[Team] | None = None,
    ) -> Team:
        """Select a team for the next battle."""
        pass

    def update(
        self,
        won: bool,
        opponent_type: str,
        team_id: str | None = None,
    ) -> None:
        """Update curriculum state after a battle."""
        self.state.record_battle(won, opponent_type, team_id)
        self._check_level_progression()

    def _check_level_progression(self) -> None:
        """Check if difficulty level should change."""
        # Only check after minimum battles
        if self.state.total_battles < self.config.min_battles_per_level:
            return

        win_rate = self.state.recent_win_rate
        current = self.state.current_level.value

        # Promotion
        if win_rate >= self.config.promotion_threshold and current < 5:
            self.state.current_level = DifficultyLevel(current + 1)
            self.state.recent_wins = 0
            self.state.recent_losses = 0
            logger.info(f"Curriculum: Promoted to {self.state.current_level.name}")

        # Demotion
        elif win_rate <= self.config.demotion_threshold and current > 1:
            self.state.current_level = DifficultyLevel(current - 1)
            self.state.recent_wins = 0
            self.state.recent_losses = 0
            logger.info(f"Curriculum: Demoted to {self.state.current_level.name}")

    def get_state_dict(self) -> dict:
        """Get serializable state."""
        return {
            "state": self.state.to_dict(),
            "config": self.config.to_dict(),
        }

    def load_state_dict(self, d: dict) -> None:
        """Load state from dictionary."""
        if "state" in d:
            self.state = CurriculumState.from_dict(d["state"])


class ProgressiveDifficultyCurriculum(CurriculumStrategy):
    """Curriculum that progressively increases opponent difficulty.

    Levels:
    1. BEGINNER: 100% random opponents
    2. EASY: 70% random, 30% maxdamage
    3. MEDIUM: 50% maxdamage, 50% weak self-play
    4. HARD: 30% maxdamage, 70% self-play
    5. EXPERT: 100% self-play (skill-matched)
    """

    # Opponent distributions per level
    LEVEL_DISTRIBUTIONS = {
        DifficultyLevel.BEGINNER: {
            OpponentType.RANDOM: 1.0,
        },
        DifficultyLevel.EASY: {
            OpponentType.RANDOM: 0.7,
            OpponentType.MAXDAMAGE: 0.3,
        },
        DifficultyLevel.MEDIUM: {
            OpponentType.RANDOM: 0.2,
            OpponentType.MAXDAMAGE: 0.5,
            OpponentType.SELF_PLAY: 0.3,
        },
        DifficultyLevel.HARD: {
            OpponentType.MAXDAMAGE: 0.3,
            OpponentType.SELF_PLAY: 0.7,
        },
        DifficultyLevel.EXPERT: {
            OpponentType.SELF_PLAY: 1.0,
        },
    }

    def select_opponent_type(self) -> OpponentType:
        """Select opponent based on current difficulty level."""
        distribution = self.LEVEL_DISTRIBUTIONS[self.state.current_level]

        # Sample from distribution
        r = random.random()
        cumulative = 0.0
        for opponent_type, prob in distribution.items():
            cumulative += prob
            if r <= cumulative:
                return opponent_type

        # Fallback
        return OpponentType.RANDOM

    def select_team(
        self,
        available_teams: list[Team],
        sample_teams: list[Team] | None = None,
    ) -> Team:
        """Select team with slight preference for underused teams."""
        if not available_teams:
            if sample_teams:
                return random.choice(sample_teams)
            raise ValueError("No teams available")

        # Early training: prefer sample teams
        if (
            self.config.start_with_sample_teams
            and sample_teams
            and self.state.total_battles < self.config.generated_team_threshold
        ):
            return random.choice(sample_teams)

        # Weight by inverse usage (prefer underused)
        weights = []
        for team in available_teams:
            team_id = str(id(team))
            usage = self.state.team_usage.get(team_id, 0)
            # Inverse usage weight (less used = higher weight)
            weight = 1.0 / (1.0 + usage * 0.1)
            weights.append(weight)

        # Normalize
        total = sum(weights)
        weights = [w / total for w in weights]

        # Weighted random selection
        return random.choices(available_teams, weights=weights, k=1)[0]


class MatchupFocusCurriculum(CurriculumStrategy):
    """Curriculum that focuses training on weak matchups.

    Tracks win rate against each opponent type and allocates more
    training time to matchups where the agent struggles.
    """

    def __init__(self, config: CurriculumConfig | None = None):
        super().__init__(config)
        self.opponent_types = [
            OpponentType.RANDOM,
            OpponentType.MAXDAMAGE,
            OpponentType.SELF_PLAY,
        ]

    def select_opponent_type(self) -> OpponentType:
        """Select opponent, focusing on weak matchups."""
        # Always some exploration
        if random.random() < self.config.random_opponent_prob:
            return random.choice(self.opponent_types)

        if not self.config.focus_on_weakness:
            return random.choice(self.opponent_types)

        # Find weakest matchup
        weakest_type = None
        lowest_win_rate = 1.0

        for opp_type in self.opponent_types:
            win_rate = self.state.get_opponent_win_rate(opp_type.value)
            total_games = (
                self.state.opponent_wins.get(opp_type.value, 0) +
                self.state.opponent_losses.get(opp_type.value, 0)
            )

            # Need minimum games to evaluate
            if total_games < 10:
                # Prioritize unexplored matchups
                return opp_type

            if win_rate < lowest_win_rate:
                lowest_win_rate = win_rate
                weakest_type = opp_type

        # Focus on weakness if below threshold
        if weakest_type and lowest_win_rate < self.config.weakness_threshold:
            # 70% chance to train weakness, 30% other
            if random.random() < 0.7:
                return weakest_type

        return random.choice(self.opponent_types)

    def select_team(
        self,
        available_teams: list[Team],
        sample_teams: list[Team] | None = None,
    ) -> Team:
        """Select team based on performance and diversity."""
        if not available_teams:
            if sample_teams:
                return random.choice(sample_teams)
            raise ValueError("No teams available")

        weights = []
        for team in available_teams:
            team_id = str(id(team))

            # Diversity component (inverse usage)
            usage = self.state.team_usage.get(team_id, 0)
            diversity_score = 1.0 / (1.0 + usage * 0.1)

            # Performance component
            scores = self.state.team_scores.get(team_id, [])
            if len(scores) >= self.config.min_team_games:
                performance_score = sum(scores) / len(scores)
            else:
                performance_score = 0.5  # Neutral for unexplored

            # Combined weight
            weight = (
                self.config.team_diversity_weight * diversity_score +
                self.config.team_performance_weight * performance_score
            )
            weights.append(max(weight, 0.01))  # Ensure positive

        # Normalize
        total = sum(weights)
        weights = [w / total for w in weights]

        return random.choices(available_teams, weights=weights, k=1)[0]


class TeamComplexityCurriculum(CurriculumStrategy):
    """Curriculum that progresses team complexity.

    Starts with known-good sample teams and gradually introduces
    generated teams as the player improves.
    """

    def __init__(
        self,
        config: CurriculumConfig | None = None,
        sample_teams: list[Team] | None = None,
    ):
        super().__init__(config)
        self.sample_teams = sample_teams or []
        self.generated_team_ratio = 0.0  # Starts at 0%

    def select_opponent_type(self) -> OpponentType:
        """Simple progressive difficulty for opponents."""
        level = self.state.current_level

        if level == DifficultyLevel.BEGINNER:
            return OpponentType.RANDOM
        elif level == DifficultyLevel.EASY:
            return random.choice([OpponentType.RANDOM, OpponentType.MAXDAMAGE])
        else:
            return random.choice([OpponentType.MAXDAMAGE, OpponentType.SELF_PLAY])

    def select_team(
        self,
        available_teams: list[Team],
        sample_teams: list[Team] | None = None,
    ) -> Team:
        """Select team based on complexity progression."""
        sample_teams = sample_teams or self.sample_teams

        # Update generated team ratio based on progress
        self._update_generated_ratio()

        # Decide sample vs generated
        use_generated = (
            available_teams and
            random.random() < self.generated_team_ratio
        )

        if use_generated:
            return random.choice(available_teams)
        elif sample_teams:
            return random.choice(sample_teams)
        elif available_teams:
            return random.choice(available_teams)
        else:
            raise ValueError("No teams available")

    def _update_generated_ratio(self) -> None:
        """Update the ratio of generated teams based on progress."""
        # Linear progression based on battles
        battles = self.state.total_battles
        threshold = self.config.generated_team_threshold

        if battles < threshold:
            self.generated_team_ratio = 0.0
        else:
            # Ramp up from 0% to 80% over 5x threshold battles
            progress = (battles - threshold) / (threshold * 5)
            self.generated_team_ratio = min(0.8, progress * 0.8)


class AdaptiveCurriculum(CurriculumStrategy):
    """Adaptive curriculum that combines multiple strategies.

    Monitors training progress and adjusts strategy:
    - Early training: Progressive difficulty
    - Plateaued: Focus on weaknesses
    - Advanced: Team complexity progression
    """

    def __init__(
        self,
        config: CurriculumConfig | None = None,
        sample_teams: list[Team] | None = None,
    ):
        super().__init__(config)
        self.sample_teams = sample_teams or []

        # Sub-strategies
        self.progressive = ProgressiveDifficultyCurriculum(config)
        self.matchup = MatchupFocusCurriculum(config)
        self.complexity = TeamComplexityCurriculum(config, sample_teams)

        # Track which strategy is active
        self.active_strategy = "progressive"
        self.plateau_counter = 0
        self.last_win_rate = 0.5

    def select_opponent_type(self) -> OpponentType:
        """Select opponent using active strategy."""
        self._update_active_strategy()

        if self.active_strategy == "progressive":
            return self.progressive.select_opponent_type()
        elif self.active_strategy == "matchup":
            return self.matchup.select_opponent_type()
        else:
            return self.complexity.select_opponent_type()

    def select_team(
        self,
        available_teams: list[Team],
        sample_teams: list[Team] | None = None,
    ) -> Team:
        """Select team using active strategy."""
        sample_teams = sample_teams or self.sample_teams

        if self.active_strategy == "progressive":
            return self.progressive.select_team(available_teams, sample_teams)
        elif self.active_strategy == "matchup":
            return self.matchup.select_team(available_teams, sample_teams)
        else:
            return self.complexity.select_team(available_teams, sample_teams)

    def update(
        self,
        won: bool,
        opponent_type: str,
        team_id: str | None = None,
    ) -> None:
        """Update all sub-strategies."""
        super().update(won, opponent_type, team_id)

        # Also update sub-strategies
        self.progressive.update(won, opponent_type, team_id)
        self.matchup.update(won, opponent_type, team_id)
        self.complexity.update(won, opponent_type, team_id)

    def _update_active_strategy(self) -> None:
        """Decide which strategy to use based on progress."""
        win_rate = self.state.recent_win_rate
        level = self.state.current_level
        battles = self.state.total_battles

        # Check for plateau (win rate not improving)
        if abs(win_rate - self.last_win_rate) < 0.02:
            self.plateau_counter += 1
        else:
            self.plateau_counter = 0
        self.last_win_rate = win_rate

        # Strategy selection logic
        if battles < 100:
            # Early training: progressive difficulty
            self.active_strategy = "progressive"
        elif self.plateau_counter > 20:
            # Plateaued: focus on weaknesses
            self.active_strategy = "matchup"
        elif level.value >= 4:
            # Advanced: complexity progression
            self.active_strategy = "complexity"
        else:
            # Default: progressive
            self.active_strategy = "progressive"

    def get_state_dict(self) -> dict:
        """Get serializable state."""
        return {
            "state": self.state.to_dict(),
            "config": self.config.to_dict(),
            "active_strategy": self.active_strategy,
            "plateau_counter": self.plateau_counter,
            "last_win_rate": self.last_win_rate,
            "progressive": self.progressive.get_state_dict(),
            "matchup": self.matchup.get_state_dict(),
            "complexity": self.complexity.get_state_dict(),
        }

    def load_state_dict(self, d: dict) -> None:
        """Load state from dictionary."""
        super().load_state_dict(d)
        self.active_strategy = d.get("active_strategy", "progressive")
        self.plateau_counter = d.get("plateau_counter", 0)
        self.last_win_rate = d.get("last_win_rate", 0.5)

        if "progressive" in d:
            self.progressive.load_state_dict(d["progressive"])
        if "matchup" in d:
            self.matchup.load_state_dict(d["matchup"])
        if "complexity" in d:
            self.complexity.load_state_dict(d["complexity"])


def create_curriculum(
    strategy: str = "adaptive",
    config: CurriculumConfig | None = None,
    sample_teams: list[Team] | None = None,
) -> CurriculumStrategy:
    """Factory function to create curriculum strategies.

    Args:
        strategy: One of "progressive", "matchup", "complexity", "adaptive"
        config: Curriculum configuration
        sample_teams: Sample teams for complexity curriculum

    Returns:
        CurriculumStrategy instance
    """
    config = config or CurriculumConfig()

    if strategy == "progressive":
        return ProgressiveDifficultyCurriculum(config)
    elif strategy == "matchup":
        return MatchupFocusCurriculum(config)
    elif strategy == "complexity":
        return TeamComplexityCurriculum(config, sample_teams)
    elif strategy == "adaptive":
        return AdaptiveCurriculum(config, sample_teams)
    else:
        raise ValueError(f"Unknown curriculum strategy: {strategy}")
