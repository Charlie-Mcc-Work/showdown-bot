"""Tests for OU joint trainer."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from showdown_bot.ou.training.joint_trainer import (
    JointTrainer,
    JointTrainingStats,
    create_joint_trainer,
)
from showdown_bot.ou.training.config import OUTrainingConfig
from showdown_bot.ou.training.curriculum import (
    OpponentType,
    DifficultyLevel,
    CurriculumConfig,
)


class TestJointTrainingStats:
    """Tests for JointTrainingStats dataclass."""

    def test_initial_values(self):
        """Test default statistics values."""
        stats = JointTrainingStats()
        assert stats.total_battles == 0
        assert stats.player_wins == 0
        assert stats.player_losses == 0
        assert stats.player_skill_rating == 1000.0
        assert stats.teams_generated == 0

    def test_player_win_rate_empty(self):
        """Test win rate with no battles."""
        stats = JointTrainingStats()
        assert stats.player_win_rate == 0.5

    def test_player_win_rate_calculation(self):
        """Test win rate calculation."""
        stats = JointTrainingStats()
        stats.player_wins = 60
        stats.player_losses = 40
        assert stats.player_win_rate == 0.6


class TestJointTrainer:
    """Tests for JointTrainer class."""

    @pytest.fixture
    def mock_player_trainer(self):
        """Create mock player trainer."""
        trainer = MagicMock()
        trainer.skill_rating = 1000.0
        trainer.update.return_value = {"policy_loss": 0.1}
        return trainer

    @pytest.fixture
    def mock_teambuilder_trainer(self):
        """Create mock teambuilder trainer."""
        trainer = MagicMock()
        trainer.update.return_value = {"evaluator/loss": 0.2}
        trainer.outcome_buffer.get_team_win_rate.return_value = 0.5
        return trainer

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return OUTrainingConfig(
            player_rollout_steps=128,
            teambuilder_min_games=10,
            log_interval=10,
        )

    @pytest.fixture
    def joint_trainer(self, mock_player_trainer, mock_teambuilder_trainer, config):
        """Create JointTrainer with mocks."""
        return JointTrainer(
            player_trainer=mock_player_trainer,
            teambuilder_trainer=mock_teambuilder_trainer,
            config=config,
            curriculum_strategy="progressive",
        )

    def test_initialization(self, joint_trainer):
        """Test JointTrainer initialization."""
        assert joint_trainer.stats.total_battles == 0
        assert joint_trainer.current_phase == "player"
        assert joint_trainer.active_teams == []
        assert joint_trainer.curriculum_enabled

    def test_initialize_teams_with_samples(self, joint_trainer, mock_teambuilder_trainer):
        """Test team pool initialization with sample teams."""
        sample_teams = [MagicMock() for _ in range(5)]
        mock_teambuilder_trainer.generate_team.return_value = MagicMock()

        joint_trainer.initialize_teams(sample_teams=sample_teams)

        assert len(joint_trainer.active_teams) == joint_trainer.team_pool_size
        assert len(joint_trainer.sample_teams) == 5

    def test_initialize_teams_generates_when_needed(self, joint_trainer, mock_teambuilder_trainer):
        """Test that teams are generated when not enough samples."""
        mock_teambuilder_trainer.generate_team.return_value = MagicMock()

        joint_trainer.initialize_teams(sample_teams=[])

        # Should have generated teams
        assert joint_trainer.stats.teams_generated > 0

    def test_get_training_team_empty_generates(self, joint_trainer, mock_teambuilder_trainer):
        """Test get_training_team generates when pool is empty."""
        mock_team = MagicMock()
        mock_teambuilder_trainer.generate_team.return_value = mock_team

        team = joint_trainer.get_training_team()

        assert team == mock_team
        assert len(joint_trainer.active_teams) == 1

    def test_get_training_team_uses_curriculum(self, joint_trainer):
        """Test get_training_team uses curriculum for selection."""
        teams = [MagicMock(), MagicMock()]
        joint_trainer.active_teams = teams
        joint_trainer.sample_teams = teams

        team = joint_trainer.get_training_team()

        assert team in teams

    def test_get_opponent_type_uses_curriculum(self, joint_trainer):
        """Test get_opponent_type uses curriculum."""
        # At BEGINNER level, should return RANDOM
        opp_type = joint_trainer.get_opponent_type()
        assert opp_type == OpponentType.RANDOM

    def test_get_opponent_type_fallback(self, joint_trainer):
        """Test get_opponent_type falls back when curriculum disabled."""
        joint_trainer.curriculum_enabled = False

        opp_type = joint_trainer.get_opponent_type()

        assert opp_type in [OpponentType.RANDOM, OpponentType.MAXDAMAGE]

    def test_record_battle_outcome_updates_stats(self, joint_trainer, mock_player_trainer):
        """Test record_battle_outcome updates statistics."""
        team = MagicMock()
        joint_trainer.active_teams = [team]

        joint_trainer.record_battle_outcome(
            team=team,
            won=True,
            turns=20,
            opponent_revealed=["Pikachu", "Charizard"],
            opponent_type="random",
        )

        assert joint_trainer.stats.total_battles == 1
        assert joint_trainer.stats.player_wins == 1
        mock_player_trainer.update_skill_rating.assert_called_once()

    def test_record_battle_outcome_loss(self, joint_trainer, mock_player_trainer):
        """Test record_battle_outcome with loss."""
        team = MagicMock()

        joint_trainer.record_battle_outcome(
            team=team,
            won=False,
            turns=15,
            opponent_revealed=["Mewtwo"],
            opponent_type="maxdamage",
        )

        assert joint_trainer.stats.player_losses == 1

    def test_record_battle_outcome_updates_curriculum(self, joint_trainer):
        """Test record_battle_outcome updates curriculum state."""
        team = MagicMock()

        for _ in range(5):
            joint_trainer.record_battle_outcome(
                team=team,
                won=True,
                turns=10,
                opponent_revealed=[],
                opponent_type="random",
            )

        assert joint_trainer.curriculum.state.total_battles == 5

    def test_update_player(self, joint_trainer, mock_player_trainer):
        """Test update_player calls trainer."""
        metrics = joint_trainer.update_player()

        mock_player_trainer.update.assert_called_once()
        assert joint_trainer.stats.total_player_updates == 1
        assert "policy_loss" in metrics

    def test_update_teambuilder(self, joint_trainer, mock_teambuilder_trainer):
        """Test update_teambuilder calls trainer."""
        metrics = joint_trainer.update_teambuilder()

        mock_teambuilder_trainer.update.assert_called_once()
        assert joint_trainer.stats.total_teambuilder_updates == 1

    def test_rotate_teams_replaces_worst(self, joint_trainer, mock_teambuilder_trainer):
        """Test rotate_teams replaces worst-performing teams."""
        # Create teams with different win rates
        teams = [MagicMock() for _ in range(5)]
        joint_trainer.active_teams = teams

        # Mock win rates
        mock_teambuilder_trainer.outcome_buffer.get_team_win_rate.side_effect = [
            0.2, 0.4, 0.6, 0.8, 0.9
        ]
        mock_teambuilder_trainer.generate_team.return_value = MagicMock()

        joint_trainer.rotate_teams()

        # Should have generated new team(s)
        assert joint_trainer.stats.teams_generated > 0

    def test_training_step(self, joint_trainer, mock_player_trainer):
        """Test training_step updates player."""
        metrics = joint_trainer.training_step()

        assert "player" in metrics
        mock_player_trainer.update.assert_called_once()
        assert joint_trainer.global_step == 1

    def test_get_summary(self, joint_trainer):
        """Test get_summary returns correct info."""
        joint_trainer.stats.total_battles = 100
        joint_trainer.stats.player_wins = 60
        joint_trainer.stats.player_losses = 40

        summary = joint_trainer.get_summary()

        assert summary["total_battles"] == 100
        assert summary["player_win_rate"] == 0.6
        assert "curriculum_level" in summary


class TestJointTrainerCheckpoints:
    """Tests for JointTrainer checkpoint save/load."""

    @pytest.fixture
    def mock_trainer_pair(self):
        """Create mock trainer pair."""
        player_trainer = MagicMock()
        player_trainer.skill_rating = 1000.0
        player_trainer.update.return_value = {}

        teambuilder_trainer = MagicMock()
        teambuilder_trainer.update.return_value = {}
        teambuilder_trainer.outcome_buffer.get_team_win_rate.return_value = 0.5

        return player_trainer, teambuilder_trainer

    def test_save_checkpoint(self, mock_trainer_pair):
        """Test saving checkpoint."""
        player_trainer, teambuilder_trainer = mock_trainer_pair
        config = OUTrainingConfig()
        trainer = JointTrainer(
            player_trainer=player_trainer,
            teambuilder_trainer=teambuilder_trainer,
            config=config,
        )
        trainer.stats.total_battles = 50
        trainer.stats.player_wins = 30

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer.save_checkpoint(tmpdir)

            # Check files exist
            assert (Path(tmpdir) / "joint_state.pt").exists()
            player_trainer.save_checkpoint.assert_called_once()
            teambuilder_trainer.save_checkpoint.assert_called_once()

    def test_load_checkpoint(self, mock_trainer_pair):
        """Test loading checkpoint."""
        player_trainer, teambuilder_trainer = mock_trainer_pair
        config = OUTrainingConfig()
        trainer = JointTrainer(
            player_trainer=player_trainer,
            teambuilder_trainer=teambuilder_trainer,
            config=config,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save a checkpoint first
            trainer.stats.total_battles = 100
            trainer.stats.player_wins = 60
            trainer.stats.player_skill_rating = 1200.0
            trainer.save_checkpoint(tmpdir)

            # Create new trainer and load
            new_trainer = JointTrainer(
                player_trainer=player_trainer,
                teambuilder_trainer=teambuilder_trainer,
                config=config,
            )
            new_trainer.load_checkpoint(tmpdir)

            assert new_trainer.stats.total_battles == 100
            assert new_trainer.stats.player_wins == 60
            assert new_trainer.stats.player_skill_rating == 1200.0

    def test_checkpoint_preserves_curriculum(self, mock_trainer_pair):
        """Test checkpoint preserves curriculum state."""
        player_trainer, teambuilder_trainer = mock_trainer_pair
        config = OUTrainingConfig()
        trainer = JointTrainer(
            player_trainer=player_trainer,
            teambuilder_trainer=teambuilder_trainer,
            config=config,
            curriculum_strategy="progressive",
        )

        # Progress curriculum
        trainer.curriculum.state.current_level = DifficultyLevel.MEDIUM
        trainer.curriculum.state.total_battles = 200

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer.save_checkpoint(tmpdir)

            # Load into new trainer
            new_trainer = JointTrainer(
                player_trainer=player_trainer,
                teambuilder_trainer=teambuilder_trainer,
                config=config,
                curriculum_strategy="progressive",
            )
            new_trainer.load_checkpoint(tmpdir)

            assert new_trainer.curriculum.state.current_level == DifficultyLevel.MEDIUM
            assert new_trainer.curriculum.state.total_battles == 200


class TestJointTrainerIntegration:
    """Integration tests for JointTrainer."""

    @pytest.fixture
    def mock_components(self):
        """Create all mock components."""
        player_trainer = MagicMock()
        player_trainer.skill_rating = 1000.0
        player_trainer.update.return_value = {"policy_loss": 0.1, "value_loss": 0.05}

        teambuilder_trainer = MagicMock()
        teambuilder_trainer.update.return_value = {"evaluator/loss": 0.2}
        teambuilder_trainer.generate_team.return_value = MagicMock()
        teambuilder_trainer.outcome_buffer.get_team_win_rate.return_value = 0.5

        return player_trainer, teambuilder_trainer

    def test_full_training_cycle(self, mock_components):
        """Test a full training cycle."""
        player_trainer, teambuilder_trainer = mock_components
        config = OUTrainingConfig(
            teambuilder_min_games=5,
        )
        trainer = JointTrainer(
            player_trainer=player_trainer,
            teambuilder_trainer=teambuilder_trainer,
            config=config,
            curriculum_strategy="progressive",
        )

        # Initialize
        sample_teams = [MagicMock() for _ in range(3)]
        trainer.initialize_teams(sample_teams=sample_teams)

        # Simulate some battles
        for i in range(20):
            team = trainer.get_training_team()
            won = i % 2 == 0  # Alternating wins/losses

            trainer.record_battle_outcome(
                team=team,
                won=won,
                turns=15,
                opponent_revealed=["Pokemon1"],
                opponent_type="random",
            )

        # Do training steps
        for _ in range(5):
            trainer.training_step()

        # Check state
        assert trainer.stats.total_battles == 20
        assert trainer.stats.player_wins == 10
        assert trainer.stats.player_losses == 10
        assert trainer.stats.total_player_updates == 5

    def test_curriculum_tracks_battles_during_training(self, mock_components):
        """Test curriculum tracks battles during training."""
        player_trainer, teambuilder_trainer = mock_components
        config = OUTrainingConfig()
        curriculum_config = CurriculumConfig(
            promotion_threshold=0.7,
            min_battles_per_level=10,
        )
        trainer = JointTrainer(
            player_trainer=player_trainer,
            teambuilder_trainer=teambuilder_trainer,
            config=config,
            curriculum_strategy="progressive",
            curriculum_config=curriculum_config,
        )

        sample_teams = [MagicMock() for _ in range(3)]
        trainer.initialize_teams(sample_teams=sample_teams)

        # Record battles
        team = trainer.get_training_team()
        for i in range(15):
            trainer.record_battle_outcome(
                team=team,
                won=(i < 12),  # 80% win rate
                turns=10,
                opponent_revealed=[],
                opponent_type="random",
            )

        # Verify battles were tracked
        assert trainer.curriculum.state.total_battles == 15
        # Verify opponent-specific tracking works
        total_opponent = (
            trainer.curriculum.state.opponent_wins.get("random", 0) +
            trainer.curriculum.state.opponent_losses.get("random", 0)
        )
        assert total_opponent == 15

    def test_team_rotation_integration(self, mock_components):
        """Test team rotation during training."""
        player_trainer, teambuilder_trainer = mock_components
        config = OUTrainingConfig(
            teambuilder_min_games=5,
        )
        trainer = JointTrainer(
            player_trainer=player_trainer,
            teambuilder_trainer=teambuilder_trainer,
            config=config,
        )
        trainer.battles_per_team_before_rotate = 5

        sample_teams = [MagicMock() for _ in range(5)]
        trainer.initialize_teams(sample_teams=sample_teams)
        initial_teams = list(trainer.active_teams)

        # Simulate enough battles for rotation
        for i in range(30):
            team = trainer.get_training_team()
            trainer.record_battle_outcome(
                team=team,
                won=True,
                turns=10,
                opponent_revealed=[],
            )
            trainer.training_step()

        # Teams should have been generated for rotation
        assert trainer.stats.teams_generated > 0


class TestCreateJointTrainer:
    """Tests for create_joint_trainer factory function."""

    def test_create_joint_trainer_creates_components(self):
        """Test factory creates all components."""
        config = OUTrainingConfig()

        # Create mock components
        player_network = MagicMock()
        state_encoder = MagicMock()
        team_generator = MagicMock()
        team_evaluator = MagicMock()

        with patch("showdown_bot.ou.training.joint_trainer.OUPlayerTrainer") as mock_player_cls, \
             patch("showdown_bot.ou.training.joint_trainer.TeambuilderTrainer") as mock_tb_cls:

            mock_player_cls.return_value = MagicMock()
            mock_tb_cls.return_value = MagicMock()
            mock_tb_cls.return_value.outcome_buffer.get_team_win_rate.return_value = 0.5

            trainer = create_joint_trainer(
                config=config,
                player_network=player_network,
                state_encoder=state_encoder,
                team_generator=team_generator,
                team_evaluator=team_evaluator,
                curriculum_strategy="adaptive",
            )

            assert isinstance(trainer, JointTrainer)
            mock_player_cls.assert_called_once()
            mock_tb_cls.assert_called_once()
