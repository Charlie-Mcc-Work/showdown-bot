"""Tests for OU self-play system."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from showdown_bot.ou.training.self_play import (
    OUOpponentInfo,
    OUOpponentPool,
    OUSelfPlayManager,
    calculate_elo_update,
)


class TestOUOpponentInfo:
    """Tests for OUOpponentInfo dataclass."""

    def test_initial_values(self):
        """Test default values."""
        info = OUOpponentInfo(checkpoint_path=Path("/tmp/test.pt"))
        assert info.skill_rating == 1000.0
        assert info.games_played == 0
        assert info.wins == 0
        assert info.losses == 0
        assert info.timestep_added == 0

    def test_win_rate_no_games(self):
        """Test win rate with no games played."""
        info = OUOpponentInfo(checkpoint_path=Path("/tmp/test.pt"))
        assert info.win_rate == 0.0

    def test_win_rate_calculation(self):
        """Test win rate calculation."""
        info = OUOpponentInfo(
            checkpoint_path=Path("/tmp/test.pt"),
            games_played=10,
            wins=7,
            losses=3,
        )
        assert info.win_rate == 0.7


class TestCalculateEloUpdate:
    """Tests for Elo calculation function."""

    def test_equal_rating_win(self):
        """Test Elo update when equal-rated player wins."""
        new_player, new_opponent = calculate_elo_update(1000.0, 1000.0, True)
        # Winner should gain, loser should lose equal amounts
        assert new_player > 1000.0
        assert new_opponent < 1000.0
        assert abs((new_player - 1000.0) - (1000.0 - new_opponent)) < 0.01

    def test_equal_rating_loss(self):
        """Test Elo update when equal-rated player loses."""
        new_player, new_opponent = calculate_elo_update(1000.0, 1000.0, False)
        assert new_player < 1000.0
        assert new_opponent > 1000.0

    def test_underdog_win(self):
        """Test Elo update when underdog wins (gains more)."""
        # Lower rated player beats higher rated
        new_player, new_opponent = calculate_elo_update(900.0, 1100.0, True)
        # Underdog should gain a lot
        assert new_player > 900.0
        assert new_player - 900.0 > 16.0  # More than default K/2

    def test_favorite_win(self):
        """Test Elo update when favorite wins (gains less)."""
        # Higher rated player beats lower rated
        new_player, new_opponent = calculate_elo_update(1100.0, 900.0, True)
        # Favorite should gain less
        assert new_player > 1100.0
        assert new_player - 1100.0 < 16.0  # Less than default K/2

    def test_k_factor_affects_magnitude(self):
        """Test that K-factor affects the magnitude of changes."""
        new_p1, _ = calculate_elo_update(1000.0, 1000.0, True, k_factor=32.0)
        new_p2, _ = calculate_elo_update(1000.0, 1000.0, True, k_factor=16.0)

        change1 = new_p1 - 1000.0
        change2 = new_p2 - 1000.0
        assert abs(change1 / change2 - 2.0) < 0.01  # 2x K = 2x change


class TestOUOpponentPool:
    """Tests for OUOpponentPool class."""

    @pytest.fixture
    def temp_pool_dir(self):
        """Create temporary directory for opponent pool."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def mock_network(self):
        """Create a mock network."""
        network = MagicMock()
        network.state_dict.return_value = {"layer": torch.zeros(10)}
        return network

    def test_initialization_empty(self, temp_pool_dir):
        """Test pool initialization with empty directory."""
        pool = OUOpponentPool(
            pool_dir=temp_pool_dir,
            max_size=5,
        )
        assert pool.size == 0
        assert pool.average_skill == 1000.0

    def test_add_opponent(self, temp_pool_dir, mock_network):
        """Test adding an opponent to the pool."""
        pool = OUOpponentPool(pool_dir=temp_pool_dir, max_size=5)

        info = pool.add_opponent(
            network=mock_network,
            timestep=1000,
            skill_rating=1050.0,
        )

        assert pool.size == 1
        assert info.skill_rating == 1050.0
        assert info.timestep_added == 1000
        assert info.checkpoint_path.exists()

    def test_add_multiple_opponents(self, temp_pool_dir, mock_network):
        """Test adding multiple opponents."""
        pool = OUOpponentPool(pool_dir=temp_pool_dir, max_size=10)

        for i in range(5):
            pool.add_opponent(
                network=mock_network,
                timestep=i * 1000,
                skill_rating=1000.0 + i * 50,
            )

        assert pool.size == 5
        assert pool.average_skill == 1100.0  # (1000 + 1050 + 1100 + 1150 + 1200) / 5

    def test_prune_pool_when_over_max(self, temp_pool_dir, mock_network):
        """Test pool pruning when over max size."""
        pool = OUOpponentPool(pool_dir=temp_pool_dir, max_size=3)

        for i in range(5):
            pool.add_opponent(
                network=mock_network,
                timestep=i * 1000,
                skill_rating=1000.0 + i * 50,
            )

        # Should have pruned to max_size
        assert pool.size <= 3

    def test_sample_opponent_uniform(self, temp_pool_dir, mock_network):
        """Test uniform opponent sampling."""
        pool = OUOpponentPool(pool_dir=temp_pool_dir, max_size=10)

        for i in range(5):
            pool.add_opponent(
                network=mock_network,
                timestep=i * 1000,
                skill_rating=1000.0 + i * 100,
            )

        sampled = pool.sample_opponent(strategy="uniform")
        assert sampled is not None
        assert sampled in pool.opponents

    def test_sample_opponent_skill_matched(self, temp_pool_dir, mock_network):
        """Test skill-matched opponent sampling."""
        pool = OUOpponentPool(pool_dir=temp_pool_dir, max_size=10)

        # Add opponents with varying skills
        for skill in [800, 900, 1000, 1100, 1200]:
            pool.add_opponent(
                network=mock_network,
                timestep=skill,
                skill_rating=float(skill),
            )

        # Sample many times and check distribution favors close skills
        close_count = 0
        for _ in range(200):  # More samples for statistical stability
            sampled = pool.sample_opponent(
                strategy="skill_matched", current_skill=1000.0
            )
            if sampled and 950 <= sampled.skill_rating <= 1050:
                close_count += 1

        # Should favor skill=1000 opponent (uniform would be ~40/200=20%)
        # With skill matching, should be higher than uniform
        assert close_count > 50  # At least 25% (more lenient due to variance)

    def test_sample_opponent_prioritized(self, temp_pool_dir, mock_network):
        """Test prioritized (less played) opponent sampling."""
        pool = OUOpponentPool(pool_dir=temp_pool_dir, max_size=10)

        for i in range(3):
            pool.add_opponent(
                network=mock_network,
                timestep=i * 1000,
                skill_rating=1000.0,
            )

        # Mark first two as played many times
        pool.opponents[0].games_played = 100
        pool.opponents[1].games_played = 50
        pool.opponents[2].games_played = 0

        # Sample many times - should prefer less played
        last_count = 0
        for _ in range(100):
            sampled = pool.sample_opponent(strategy="prioritized")
            if sampled == pool.opponents[2]:
                last_count += 1

        assert last_count > 50  # Should heavily favor unplayed opponent

    def test_sample_opponent_empty_pool(self, temp_pool_dir):
        """Test sampling from empty pool returns None."""
        pool = OUOpponentPool(pool_dir=temp_pool_dir, max_size=5)
        assert pool.sample_opponent() is None

    def test_update_stats_win(self, temp_pool_dir, mock_network):
        """Test updating stats after a win."""
        pool = OUOpponentPool(pool_dir=temp_pool_dir, max_size=5)

        info = pool.add_opponent(
            network=mock_network,
            timestep=1000,
            skill_rating=1000.0,
        )

        new_agent_skill, new_opp_skill = pool.update_stats(
            opponent_info=info,
            won=True,
            agent_skill=1000.0,
        )

        assert info.games_played == 1
        assert info.losses == 1  # Opponent lost
        assert info.wins == 0
        assert new_agent_skill > 1000.0
        assert new_opp_skill < 1000.0

    def test_update_stats_loss(self, temp_pool_dir, mock_network):
        """Test updating stats after a loss."""
        pool = OUOpponentPool(pool_dir=temp_pool_dir, max_size=5)

        info = pool.add_opponent(
            network=mock_network,
            timestep=1000,
            skill_rating=1000.0,
        )

        new_agent_skill, new_opp_skill = pool.update_stats(
            opponent_info=info,
            won=False,
            agent_skill=1000.0,
        )

        assert info.games_played == 1
        assert info.wins == 1  # Opponent won
        assert info.losses == 0
        assert new_agent_skill < 1000.0
        assert new_opp_skill > 1000.0

    def test_load_existing_checkpoints(self, temp_pool_dir, mock_network):
        """Test loading existing checkpoints on init."""
        # First, create a pool and add opponents
        pool1 = OUOpponentPool(pool_dir=temp_pool_dir, max_size=5)
        for i in range(3):
            pool1.add_opponent(
                network=mock_network,
                timestep=i * 1000,
                skill_rating=1000.0,
            )
        assert pool1.size == 3

        # Create new pool from same directory - should load existing
        pool2 = OUOpponentPool(pool_dir=temp_pool_dir, max_size=5)
        assert pool2.size == 3


class TestOUSelfPlayManager:
    """Tests for OUSelfPlayManager class."""

    @pytest.fixture
    def temp_pool_dir(self):
        """Create temporary directory for self-play manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def mock_teams(self):
        """Create mock teams list."""
        return [MagicMock() for _ in range(3)]

    @pytest.fixture
    def mock_network(self):
        """Create a mock network."""
        network = MagicMock()
        network.state_dict.return_value = {"layer": torch.zeros(10)}
        return network

    def test_initialization(self, temp_pool_dir, mock_teams):
        """Test manager initialization."""
        manager = OUSelfPlayManager(
            pool_dir=temp_pool_dir,
            teams=mock_teams,
            max_pool_size=10,
            self_play_ratio=0.7,
            checkpoint_interval=50000,
        )

        assert manager.agent_skill == 1000.0
        assert manager.opponent_pool.size == 0
        assert manager.games_vs_self_play == 0
        assert manager.games_vs_random == 0
        assert manager.games_vs_maxdamage == 0

    def test_should_use_self_play_empty_pool(self, temp_pool_dir, mock_teams):
        """Test self-play decision with empty pool."""
        manager = OUSelfPlayManager(
            pool_dir=temp_pool_dir,
            teams=mock_teams,
            self_play_ratio=0.7,
        )

        # Should never use self-play when pool is empty
        for _ in range(10):
            assert manager.should_use_self_play() is False

    def test_should_use_self_play_with_opponents(
        self, temp_pool_dir, mock_teams, mock_network
    ):
        """Test self-play decision with opponents in pool."""
        manager = OUSelfPlayManager(
            pool_dir=temp_pool_dir,
            teams=mock_teams,
            self_play_ratio=0.7,
        )

        # Add an opponent
        manager.add_checkpoint(mock_network, 1000)

        # Should use self-play approximately 70% of the time
        self_play_count = sum(manager.should_use_self_play() for _ in range(100))
        assert 50 < self_play_count < 90  # Allow variance

    def test_should_add_checkpoint(self, temp_pool_dir, mock_teams):
        """Test checkpoint timing logic."""
        manager = OUSelfPlayManager(
            pool_dir=temp_pool_dir,
            teams=mock_teams,
            checkpoint_interval=50000,
        )

        # Should add at start (last_checkpoint is negative)
        assert manager.should_add_checkpoint(0) is True
        assert manager.should_add_checkpoint(50000) is True

        # After adding, shouldn't add again until interval passes
        manager.last_checkpoint_timestep = 0
        assert manager.should_add_checkpoint(10000) is False
        assert manager.should_add_checkpoint(49999) is False
        assert manager.should_add_checkpoint(50000) is True

    def test_add_checkpoint(self, temp_pool_dir, mock_teams, mock_network):
        """Test adding checkpoint to pool."""
        manager = OUSelfPlayManager(
            pool_dir=temp_pool_dir,
            teams=mock_teams,
        )

        info = manager.add_checkpoint(mock_network, timestep=10000)

        assert manager.opponent_pool.size == 1
        assert manager.last_checkpoint_timestep == 10000
        assert info.timestep_added == 10000
        assert info.skill_rating == manager.agent_skill

    def test_get_opponent_random(self, temp_pool_dir, mock_teams):
        """Test getting random opponent (empty pool)."""
        manager = OUSelfPlayManager(
            pool_dir=temp_pool_dir,
            teams=mock_teams,
        )

        # With empty pool, should get random or maxdamage
        opponent, info, opp_type = manager.get_opponent()

        assert info is None
        assert opp_type in ["random", "maxdamage"]

    def test_get_opponent_self_play(self, temp_pool_dir, mock_teams, mock_network):
        """Test getting self-play opponent."""
        manager = OUSelfPlayManager(
            pool_dir=temp_pool_dir,
            teams=mock_teams,
            self_play_ratio=1.0,  # Always use self-play
        )

        manager.add_checkpoint(mock_network, 1000)

        # Mock the player creation to avoid network loading
        with patch.object(manager.opponent_pool, "create_player") as mock_create:
            mock_create.return_value = MagicMock()
            opponent, info, opp_type = manager.get_opponent()

            assert info is not None
            assert opp_type == "self_play"
            assert manager.games_vs_self_play == 1

    def test_update_after_game_self_play(
        self, temp_pool_dir, mock_teams, mock_network
    ):
        """Test updating stats after self-play game."""
        manager = OUSelfPlayManager(
            pool_dir=temp_pool_dir,
            teams=mock_teams,
        )

        info = manager.add_checkpoint(mock_network, 1000)
        initial_skill = manager.agent_skill

        # Win against self-play opponent
        manager.update_after_game(info, won=True)

        assert manager.agent_skill > initial_skill
        assert info.games_played == 1

    def test_update_after_game_baseline(self, temp_pool_dir, mock_teams):
        """Test updating stats after baseline game."""
        manager = OUSelfPlayManager(
            pool_dir=temp_pool_dir,
            teams=mock_teams,
        )

        initial_skill = manager.agent_skill

        # Win against random opponent
        manager.update_after_game(None, won=True, opponent_type="random")

        # Should gain skill (but less than against equal opponent)
        assert manager.agent_skill > initial_skill

    def test_get_stats(self, temp_pool_dir, mock_teams, mock_network):
        """Test stats retrieval."""
        manager = OUSelfPlayManager(
            pool_dir=temp_pool_dir,
            teams=mock_teams,
        )

        manager.add_checkpoint(mock_network, 1000)
        manager.games_vs_self_play = 10
        manager.games_vs_random = 3
        manager.games_vs_maxdamage = 2

        stats = manager.get_stats()

        assert stats["agent_skill"] == 1000.0
        assert stats["pool_size"] == 1
        assert stats["games_vs_self_play"] == 10
        assert stats["games_vs_random"] == 3
        assert stats["games_vs_maxdamage"] == 2
        assert abs(stats["self_play_ratio_actual"] - 0.667) < 0.01


class TestOUSelfPlayIntegration:
    """Integration tests for the self-play system."""

    @pytest.fixture
    def temp_pool_dir(self):
        """Create temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def mock_network(self):
        """Create a mock network."""
        network = MagicMock()
        network.state_dict.return_value = {"layer": torch.zeros(10)}
        return network

    def test_full_self_play_cycle(self, temp_pool_dir, mock_network):
        """Test a full cycle of self-play training."""
        manager = OUSelfPlayManager(
            pool_dir=temp_pool_dir,
            teams=[],
            max_pool_size=5,
            self_play_ratio=0.5,
            checkpoint_interval=100,
        )

        # Simulate training with checkpoints
        for timestep in range(0, 500, 100):
            if manager.should_add_checkpoint(timestep):
                manager.add_checkpoint(mock_network, timestep)

        assert manager.opponent_pool.size == 5

        # Simulate games
        for _ in range(20):
            with patch.object(manager.opponent_pool, "create_player") as mock_create:
                mock_create.return_value = MagicMock()
                _, info, opp_type = manager.get_opponent()

            won = True  # Always win
            manager.update_after_game(info, won, opp_type)

        # Agent skill should have increased from wins
        assert manager.agent_skill > 1000.0

        stats = manager.get_stats()
        assert stats["games_vs_self_play"] + stats["games_vs_random"] + stats["games_vs_maxdamage"] == 20

    def test_skill_progression(self, temp_pool_dir, mock_network):
        """Test skill rating changes over many games."""
        manager = OUSelfPlayManager(
            pool_dir=temp_pool_dir,
            teams=[],
            self_play_ratio=1.0,
        )

        manager.add_checkpoint(mock_network, 1000)

        # Win streak should increase skill
        for _ in range(10):
            info = manager.opponent_pool.opponents[0]
            manager.update_after_game(info, won=True)

        skill_after_wins = manager.agent_skill

        # Lose streak should decrease skill
        for _ in range(10):
            info = manager.opponent_pool.opponents[0]
            manager.update_after_game(info, won=False)

        skill_after_losses = manager.agent_skill

        assert skill_after_wins > 1000.0
        assert skill_after_losses < skill_after_wins
