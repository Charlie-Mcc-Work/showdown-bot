"""Elo rating system for tracking agent performance."""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json


@dataclass
class EloRating:
    """Elo rating with history tracking."""

    rating: float = 1000.0
    games_played: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    peak_rating: float = 1000.0
    history: list[tuple[int, float]] = field(default_factory=list)  # (timestep, rating)

    @property
    def win_rate(self) -> float:
        """Calculate overall win rate."""
        if self.games_played == 0:
            return 0.0
        return self.wins / self.games_played

    def update(
        self,
        opponent_rating: float,
        result: float,  # 1.0 = win, 0.5 = draw, 0.0 = loss
        k_factor: float = 32.0,
        timestep: int | None = None,
    ) -> float:
        """Update rating after a game.

        Args:
            opponent_rating: Opponent's Elo rating
            result: Game result (1.0 win, 0.5 draw, 0.0 loss)
            k_factor: Elo K-factor
            timestep: Optional timestep for history tracking

        Returns:
            Rating change
        """
        # Expected score
        expected = 1.0 / (1.0 + 10 ** ((opponent_rating - self.rating) / 400.0))

        # Rating change
        change = k_factor * (result - expected)
        self.rating += change

        # Update statistics
        self.games_played += 1
        if result == 1.0:
            self.wins += 1
        elif result == 0.0:
            self.losses += 1
        else:
            self.draws += 1

        # Track peak
        if self.rating > self.peak_rating:
            self.peak_rating = self.rating

        # Record history
        if timestep is not None:
            self.history.append((timestep, self.rating))

        return change

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "rating": self.rating,
            "games_played": self.games_played,
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
            "peak_rating": self.peak_rating,
            "history": self.history,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EloRating":
        """Create from dictionary."""
        return cls(
            rating=data.get("rating", 1000.0),
            games_played=data.get("games_played", 0),
            wins=data.get("wins", 0),
            losses=data.get("losses", 0),
            draws=data.get("draws", 0),
            peak_rating=data.get("peak_rating", 1000.0),
            history=data.get("history", []),
        )


@dataclass
class MatchResult:
    """Result of a match for record keeping."""

    timestamp: str
    opponent_type: str  # "random", "self_play", "max_damage", etc.
    opponent_rating: float
    won: bool
    rating_before: float
    rating_after: float
    timestep: int


class EloTracker:
    """Tracks Elo ratings across training and evaluation."""

    def __init__(self, save_path: Path | str | None = None):
        """Initialize Elo tracker.

        Args:
            save_path: Path to save/load ratings
        """
        self.save_path = Path(save_path) if save_path else None
        self.agent_rating = EloRating()
        self.opponent_ratings: dict[str, EloRating] = {}
        self.match_history: list[MatchResult] = []

        if self.save_path and self.save_path.exists():
            self.load()

    def record_match(
        self,
        opponent_type: str,
        opponent_rating: float,
        won: bool,
        timestep: int,
        k_factor: float = 32.0,
    ) -> float:
        """Record a match result and update ratings.

        Args:
            opponent_type: Type of opponent
            opponent_rating: Opponent's rating
            won: Whether the agent won
            timestep: Current training timestep
            k_factor: Elo K-factor

        Returns:
            Rating change for the agent
        """
        rating_before = self.agent_rating.rating
        result = 1.0 if won else 0.0

        change = self.agent_rating.update(
            opponent_rating=opponent_rating,
            result=result,
            k_factor=k_factor,
            timestep=timestep,
        )

        # Record match
        self.match_history.append(
            MatchResult(
                timestamp=datetime.now().isoformat(),
                opponent_type=opponent_type,
                opponent_rating=opponent_rating,
                won=won,
                rating_before=rating_before,
                rating_after=self.agent_rating.rating,
                timestep=timestep,
            )
        )

        # Update opponent type rating
        if opponent_type not in self.opponent_ratings:
            self.opponent_ratings[opponent_type] = EloRating(rating=opponent_rating)
        self.opponent_ratings[opponent_type].update(
            opponent_rating=rating_before,
            result=0.0 if won else 1.0,
            k_factor=k_factor,
        )

        return change

    def get_stats(self) -> dict:
        """Get current statistics."""
        # Recent performance (last 100 games)
        recent = self.match_history[-100:] if self.match_history else []
        recent_wins = sum(1 for m in recent if m.won)
        recent_win_rate = recent_wins / len(recent) if recent else 0.0

        # Performance by opponent type
        by_opponent = {}
        for opponent_type in self.opponent_ratings:
            matches = [m for m in self.match_history if m.opponent_type == opponent_type]
            wins = sum(1 for m in matches if m.won)
            by_opponent[opponent_type] = {
                "games": len(matches),
                "wins": wins,
                "win_rate": wins / len(matches) if matches else 0.0,
            }

        return {
            "current_rating": self.agent_rating.rating,
            "peak_rating": self.agent_rating.peak_rating,
            "total_games": self.agent_rating.games_played,
            "overall_win_rate": self.agent_rating.win_rate,
            "recent_win_rate": recent_win_rate,
            "by_opponent": by_opponent,
        }

    def save(self) -> None:
        """Save ratings to file."""
        if self.save_path is None:
            return

        self.save_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "agent_rating": self.agent_rating.to_dict(),
            "opponent_ratings": {
                k: v.to_dict() for k, v in self.opponent_ratings.items()
            },
            "match_history": [
                {
                    "timestamp": m.timestamp,
                    "opponent_type": m.opponent_type,
                    "opponent_rating": m.opponent_rating,
                    "won": m.won,
                    "rating_before": m.rating_before,
                    "rating_after": m.rating_after,
                    "timestep": m.timestep,
                }
                for m in self.match_history
            ],
        }

        with open(self.save_path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self) -> None:
        """Load ratings from file."""
        if self.save_path is None or not self.save_path.exists():
            return

        with open(self.save_path) as f:
            data = json.load(f)

        self.agent_rating = EloRating.from_dict(data.get("agent_rating", {}))
        self.opponent_ratings = {
            k: EloRating.from_dict(v)
            for k, v in data.get("opponent_ratings", {}).items()
        }
        self.match_history = [
            MatchResult(**m) for m in data.get("match_history", [])
        ]
