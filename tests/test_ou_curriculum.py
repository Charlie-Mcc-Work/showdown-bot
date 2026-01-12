"""Tests for OU curriculum learning strategies."""

import pytest
from unittest.mock import MagicMock

from showdown_bot.ou.training.curriculum import (
    CurriculumState,
    CurriculumConfig,
    CurriculumStrategy,
    OpponentType,
    DifficultyLevel,
    ProgressiveDifficultyCurriculum,
    MatchupFocusCurriculum,
    TeamComplexityCurriculum,
    AdaptiveCurriculum,
    create_curriculum,
)


class TestCurriculumState:
    """Tests for CurriculumState dataclass."""

    def test_initial_state(self):
        """Test default state values."""
        state = CurriculumState()
        assert state.total_battles == 0
        assert state.current_level == DifficultyLevel.BEGINNER
        assert state.recent_wins == 0
        assert state.recent_losses == 0
        assert state.window_size == 50

    def test_recent_win_rate_empty(self):
        """Test win rate with no battles."""
        state = CurriculumState()
        assert state.recent_win_rate == 0.5  # Default

    def test_recent_win_rate_calculation(self):
        """Test win rate calculation."""
        state = CurriculumState()
        state.recent_wins = 7
        state.recent_losses = 3
        assert state.recent_win_rate == 0.7

    def test_record_battle_win(self):
        """Test recording a win."""
        state = CurriculumState()
        state.record_battle(won=True, opponent_type="random")
        assert state.total_battles == 1
        assert state.recent_wins == 1
        assert state.recent_losses == 0
        assert state.opponent_wins["random"] == 1

    def test_record_battle_loss(self):
        """Test recording a loss."""
        state = CurriculumState()
        state.record_battle(won=False, opponent_type="maxdamage")
        assert state.total_battles == 1
        assert state.recent_wins == 0
        assert state.recent_losses == 1
        assert state.opponent_losses["maxdamage"] == 1

    def test_record_battle_with_team(self):
        """Test recording battle with team tracking."""
        state = CurriculumState()
        state.record_battle(won=True, opponent_type="random", team_id="team1")
        assert state.team_usage["team1"] == 1
        assert state.team_scores["team1"] == [1.0]

        state.record_battle(won=False, opponent_type="random", team_id="team1")
        assert state.team_usage["team1"] == 2
        assert state.team_scores["team1"] == [1.0, 0.0]

    def test_window_decay(self):
        """Test that window decays when exceeds size."""
        state = CurriculumState(window_size=10)
        # Fill window
        for _ in range(15):
            state.record_battle(won=True, opponent_type="random")
        # Should have decayed
        assert state.recent_wins + state.recent_losses <= 15

    def test_get_opponent_win_rate(self):
        """Test per-opponent win rate calculation."""
        state = CurriculumState()
        state.opponent_wins["random"] = 8
        state.opponent_losses["random"] = 2
        assert state.get_opponent_win_rate("random") == 0.8
        # Unknown opponent
        assert state.get_opponent_win_rate("unknown") == 0.5

    def test_team_scores_limit(self):
        """Test that team scores are limited to 20."""
        state = CurriculumState()
        for i in range(25):
            state.record_battle(won=True, opponent_type="random", team_id="team1")
        assert len(state.team_scores["team1"]) == 20

    def test_serialization(self):
        """Test state serialization and deserialization."""
        state = CurriculumState()
        state.total_battles = 100
        state.current_level = DifficultyLevel.MEDIUM
        state.recent_wins = 30
        state.recent_losses = 20
        state.opponent_wins["random"] = 25
        state.opponent_losses["maxdamage"] = 15
        state.team_usage["team1"] = 10
        state.team_scores["team1"] = [1.0, 0.0, 1.0]

        # Serialize
        d = state.to_dict()
        assert d["total_battles"] == 100
        assert d["current_level"] == 3  # MEDIUM

        # Deserialize
        loaded = CurriculumState.from_dict(d)
        assert loaded.total_battles == 100
        assert loaded.current_level == DifficultyLevel.MEDIUM
        assert loaded.recent_wins == 30
        assert loaded.opponent_wins["random"] == 25
        assert loaded.team_scores["team1"] == [1.0, 0.0, 1.0]


class TestCurriculumConfig:
    """Tests for CurriculumConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = CurriculumConfig()
        assert config.promotion_threshold == 0.65
        assert config.demotion_threshold == 0.35
        assert config.min_battles_per_level == 50
        assert config.random_opponent_prob == 0.1

    def test_serialization(self):
        """Test config serialization."""
        config = CurriculumConfig(promotion_threshold=0.7)
        d = config.to_dict()
        assert d["promotion_threshold"] == 0.7


class TestOpponentType:
    """Tests for OpponentType enum."""

    def test_all_types_exist(self):
        """Test all opponent types are defined."""
        assert OpponentType.RANDOM.value == "random"
        assert OpponentType.MAXDAMAGE.value == "maxdamage"
        assert OpponentType.HEURISTIC.value == "heuristic"
        assert OpponentType.SELF_PLAY.value == "self_play"
        assert OpponentType.LADDER.value == "ladder"


class TestDifficultyLevel:
    """Tests for DifficultyLevel enum."""

    def test_level_ordering(self):
        """Test difficulty levels are ordered correctly."""
        assert DifficultyLevel.BEGINNER.value == 1
        assert DifficultyLevel.EASY.value == 2
        assert DifficultyLevel.MEDIUM.value == 3
        assert DifficultyLevel.HARD.value == 4
        assert DifficultyLevel.EXPERT.value == 5


class TestProgressiveDifficultyCurriculum:
    """Tests for ProgressiveDifficultyCurriculum."""

    def test_initial_level(self):
        """Test starts at BEGINNER level."""
        curriculum = ProgressiveDifficultyCurriculum()
        assert curriculum.state.current_level == DifficultyLevel.BEGINNER

    def test_beginner_opponent_selection(self):
        """Test BEGINNER level only selects RANDOM opponents."""
        curriculum = ProgressiveDifficultyCurriculum()
        # Sample many times to ensure consistency
        for _ in range(20):
            opp = curriculum.select_opponent_type()
            assert opp == OpponentType.RANDOM

    def test_level_distributions_exist(self):
        """Test all levels have opponent distributions."""
        for level in DifficultyLevel:
            assert level in ProgressiveDifficultyCurriculum.LEVEL_DISTRIBUTIONS

    def test_promotion_after_wins(self):
        """Test level promotion after sufficient wins."""
        config = CurriculumConfig(
            promotion_threshold=0.65,
            min_battles_per_level=10,
        )
        curriculum = ProgressiveDifficultyCurriculum(config)

        # Win 8 out of 10 (80%)
        for i in range(10):
            curriculum.update(won=(i < 8), opponent_type="random")

        assert curriculum.state.current_level == DifficultyLevel.EASY

    def test_demotion_after_losses(self):
        """Test level demotion after sufficient losses."""
        config = CurriculumConfig(
            demotion_threshold=0.35,
            min_battles_per_level=10,
        )
        curriculum = ProgressiveDifficultyCurriculum(config)
        # Start at EASY level
        curriculum.state.current_level = DifficultyLevel.EASY

        # Lose 8 out of 10 (20% win rate)
        for i in range(10):
            curriculum.update(won=(i < 2), opponent_type="random")

        assert curriculum.state.current_level == DifficultyLevel.BEGINNER

    def test_no_demotion_below_beginner(self):
        """Test cannot demote below BEGINNER."""
        config = CurriculumConfig(min_battles_per_level=5)
        curriculum = ProgressiveDifficultyCurriculum(config)

        # Lose many battles at BEGINNER
        for _ in range(10):
            curriculum.update(won=False, opponent_type="random")

        assert curriculum.state.current_level == DifficultyLevel.BEGINNER

    def test_no_promotion_above_expert(self):
        """Test cannot promote above EXPERT."""
        config = CurriculumConfig(min_battles_per_level=5)
        curriculum = ProgressiveDifficultyCurriculum(config)
        curriculum.state.current_level = DifficultyLevel.EXPERT

        # Win many battles at EXPERT
        for _ in range(10):
            curriculum.update(won=True, opponent_type="self_play")

        assert curriculum.state.current_level == DifficultyLevel.EXPERT

    def test_select_team_with_sample_teams(self):
        """Test team selection prefers sample teams early."""
        config = CurriculumConfig(
            start_with_sample_teams=True,
            generated_team_threshold=100,
        )
        curriculum = ProgressiveDifficultyCurriculum(config)

        sample_teams = [MagicMock(), MagicMock()]
        generated_teams = [MagicMock(), MagicMock()]

        # Early training should prefer sample teams
        team = curriculum.select_team(generated_teams, sample_teams)
        assert team in sample_teams

    def test_select_team_no_teams_raises(self):
        """Test selecting from empty team list raises."""
        curriculum = ProgressiveDifficultyCurriculum()
        with pytest.raises(ValueError, match="No teams available"):
            curriculum.select_team([], None)


class TestMatchupFocusCurriculum:
    """Tests for MatchupFocusCurriculum."""

    def test_explores_unexplored_matchups(self):
        """Test prioritizes unexplored opponent types."""
        config = CurriculumConfig(random_opponent_prob=0.0)
        curriculum = MatchupFocusCurriculum(config)

        # No battles recorded yet - should explore
        opp = curriculum.select_opponent_type()
        assert opp in curriculum.opponent_types

    def test_focuses_on_weakness(self):
        """Test focuses on weak matchups."""
        config = CurriculumConfig(
            random_opponent_prob=0.0,
            focus_on_weakness=True,
            weakness_threshold=0.4,
        )
        curriculum = MatchupFocusCurriculum(config)

        # Make RANDOM a weakness (low win rate)
        for _ in range(20):
            curriculum.state.record_battle(won=False, opponent_type="random")
        # Make MAXDAMAGE strong
        for _ in range(20):
            curriculum.state.record_battle(won=True, opponent_type="maxdamage")
        # Make SELF_PLAY moderate
        for _ in range(20):
            curriculum.state.record_battle(won=True, opponent_type="self_play")
            curriculum.state.record_battle(won=False, opponent_type="self_play")

        # Should often select RANDOM (the weakness)
        random_count = sum(
            1 for _ in range(50)
            if curriculum.select_opponent_type() == OpponentType.RANDOM
        )
        # Should be focused on weakness (>50% of selections)
        assert random_count > 25

    def test_team_selection_weights_performance(self):
        """Test team selection considers performance."""
        config = CurriculumConfig(
            team_diversity_weight=0.0,
            team_performance_weight=1.0,
            min_team_games=3,
        )
        curriculum = MatchupFocusCurriculum(config)

        team1, team2 = MagicMock(), MagicMock()
        team1_id, team2_id = str(id(team1)), str(id(team2))

        # Team1 has high win rate
        curriculum.state.team_usage[team1_id] = 5
        curriculum.state.team_scores[team1_id] = [1.0, 1.0, 1.0, 1.0, 1.0]
        # Team2 has low win rate
        curriculum.state.team_usage[team2_id] = 5
        curriculum.state.team_scores[team2_id] = [0.0, 0.0, 0.0, 0.0, 0.0]

        # Should prefer team1
        selections = [curriculum.select_team([team1, team2]) for _ in range(20)]
        team1_count = sum(1 for t in selections if t == team1)
        assert team1_count > 10  # Should be biased toward team1


class TestTeamComplexityCurriculum:
    """Tests for TeamComplexityCurriculum."""

    def test_starts_with_sample_teams(self):
        """Test initially uses sample teams only."""
        config = CurriculumConfig(generated_team_threshold=100)
        sample_teams = [MagicMock(), MagicMock()]
        curriculum = TeamComplexityCurriculum(config, sample_teams)

        # Early battles should use sample teams
        assert curriculum.generated_team_ratio == 0.0
        team = curriculum.select_team([], sample_teams)
        assert team in sample_teams

    def test_generated_ratio_increases(self):
        """Test generated team ratio increases with battles."""
        config = CurriculumConfig(generated_team_threshold=100)
        curriculum = TeamComplexityCurriculum(config)

        # Before threshold
        curriculum.state.total_battles = 50
        curriculum._update_generated_ratio()
        assert curriculum.generated_team_ratio == 0.0

        # After threshold
        curriculum.state.total_battles = 200
        curriculum._update_generated_ratio()
        assert curriculum.generated_team_ratio > 0.0

    def test_opponent_selection_by_level(self):
        """Test opponent selection varies by level."""
        curriculum = TeamComplexityCurriculum()

        curriculum.state.current_level = DifficultyLevel.BEGINNER
        assert curriculum.select_opponent_type() == OpponentType.RANDOM

        curriculum.state.current_level = DifficultyLevel.HARD
        opp = curriculum.select_opponent_type()
        assert opp in [OpponentType.MAXDAMAGE, OpponentType.SELF_PLAY]


class TestAdaptiveCurriculum:
    """Tests for AdaptiveCurriculum."""

    def test_starts_with_progressive(self):
        """Test starts with progressive strategy."""
        curriculum = AdaptiveCurriculum()
        assert curriculum.active_strategy == "progressive"

    def test_updates_all_sub_strategies(self):
        """Test update() updates all sub-strategies."""
        curriculum = AdaptiveCurriculum()
        curriculum.update(won=True, opponent_type="random", team_id="team1")

        assert curriculum.state.total_battles == 1
        assert curriculum.progressive.state.total_battles == 1
        assert curriculum.matchup.state.total_battles == 1
        assert curriculum.complexity.state.total_battles == 1

    def test_switches_to_matchup_on_plateau(self):
        """Test switches to matchup strategy on plateau."""
        curriculum = AdaptiveCurriculum()
        curriculum.state.total_battles = 200
        curriculum.plateau_counter = 25  # Simulated plateau

        curriculum._update_active_strategy()
        assert curriculum.active_strategy == "matchup"

    def test_switches_to_complexity_at_advanced_level(self):
        """Test switches to complexity at advanced level."""
        curriculum = AdaptiveCurriculum()
        curriculum.state.total_battles = 200
        curriculum.state.current_level = DifficultyLevel.HARD

        curriculum._update_active_strategy()
        assert curriculum.active_strategy == "complexity"

    def test_serialization(self):
        """Test full state serialization."""
        curriculum = AdaptiveCurriculum()
        curriculum.state.total_battles = 100
        curriculum.active_strategy = "matchup"
        curriculum.plateau_counter = 5

        d = curriculum.get_state_dict()
        assert d["active_strategy"] == "matchup"
        assert d["plateau_counter"] == 5

        # Load into new curriculum
        new_curriculum = AdaptiveCurriculum()
        new_curriculum.load_state_dict(d)
        assert new_curriculum.active_strategy == "matchup"
        assert new_curriculum.plateau_counter == 5


class TestCreateCurriculum:
    """Tests for create_curriculum factory function."""

    def test_create_progressive(self):
        """Test creating progressive curriculum."""
        curriculum = create_curriculum("progressive")
        assert isinstance(curriculum, ProgressiveDifficultyCurriculum)

    def test_create_matchup(self):
        """Test creating matchup curriculum."""
        curriculum = create_curriculum("matchup")
        assert isinstance(curriculum, MatchupFocusCurriculum)

    def test_create_complexity(self):
        """Test creating complexity curriculum."""
        curriculum = create_curriculum("complexity")
        assert isinstance(curriculum, TeamComplexityCurriculum)

    def test_create_adaptive(self):
        """Test creating adaptive curriculum."""
        curriculum = create_curriculum("adaptive")
        assert isinstance(curriculum, AdaptiveCurriculum)

    def test_create_with_config(self):
        """Test creating curriculum with custom config."""
        config = CurriculumConfig(promotion_threshold=0.8)
        curriculum = create_curriculum("progressive", config=config)
        assert curriculum.config.promotion_threshold == 0.8

    def test_create_unknown_raises(self):
        """Test creating unknown strategy raises."""
        with pytest.raises(ValueError, match="Unknown curriculum strategy"):
            create_curriculum("unknown")


class TestLevelProgression:
    """Integration tests for level progression mechanics."""

    def test_full_progression_beginner_to_expert(self):
        """Test progressing through all levels."""
        config = CurriculumConfig(
            promotion_threshold=0.7,
            min_battles_per_level=10,
        )
        curriculum = ProgressiveDifficultyCurriculum(config)

        levels_reached = [DifficultyLevel.BEGINNER]

        # Progress through levels by winning
        for level_target in range(2, 6):
            # Win 8 out of 10 to promote
            for i in range(10):
                curriculum.update(won=(i < 8), opponent_type="random")

            levels_reached.append(curriculum.state.current_level)

        # Should have reached EXPERT
        assert DifficultyLevel.EXPERT in levels_reached

    def test_oscillation_prevention(self):
        """Test that level changes reset recent counters."""
        config = CurriculumConfig(min_battles_per_level=5)
        curriculum = ProgressiveDifficultyCurriculum(config)
        curriculum.state.current_level = DifficultyLevel.EASY

        # Win to promote
        for _ in range(6):
            curriculum.update(won=True, opponent_type="random")

        # After promotion, recent counters should be reset
        assert curriculum.state.recent_wins == 0
        assert curriculum.state.recent_losses == 0
