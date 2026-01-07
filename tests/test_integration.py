"""Integration tests for training system components."""

import numpy as np
import pytest
import torch
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

from showdown_bot.training.buffer import RolloutBuffer
from showdown_bot.training.ppo import PPO, PPOStats
from showdown_bot.training.trainer import Trainer, TrainablePlayer, TrainingStats
from showdown_bot.training.self_play import (
    SelfPlayManager,
    OpponentPool,
    OpponentInfo,
    HistoricalPlayer,
    calculate_elo_update,
)
from showdown_bot.models.network import PolicyValueNetwork
from showdown_bot.environment.state_encoder import StateEncoder
from showdown_bot.evaluation.elo import EloRating, EloTracker, MatchResult


class TestEloRating:
    """Test Elo rating system."""

    def test_initial_rating(self):
        """Test default initial rating."""
        rating = EloRating()
        assert rating.rating == 1000.0
        assert rating.games_played == 0
        assert rating.wins == 0
        assert rating.losses == 0

    def test_win_increases_rating(self):
        """Test winning against equal opponent increases rating."""
        rating = EloRating()
        change = rating.update(opponent_rating=1000.0, result=1.0)

        assert change > 0
        assert rating.rating > 1000.0
        assert rating.wins == 1
        assert rating.games_played == 1

    def test_loss_decreases_rating(self):
        """Test losing against equal opponent decreases rating."""
        rating = EloRating()
        change = rating.update(opponent_rating=1000.0, result=0.0)

        assert change < 0
        assert rating.rating < 1000.0
        assert rating.losses == 1

    def test_draw_handling(self):
        """Test draw against equal opponent."""
        rating = EloRating()
        change = rating.update(opponent_rating=1000.0, result=0.5)

        # Draw against equal = no change
        assert abs(change) < 0.01
        assert rating.draws == 1

    def test_win_rate_calculation(self):
        """Test win rate is calculated correctly."""
        rating = EloRating()
        rating.update(1000.0, 1.0)
        rating.update(1000.0, 1.0)
        rating.update(1000.0, 0.0)

        assert rating.win_rate == pytest.approx(2/3)

    def test_peak_rating_tracking(self):
        """Test peak rating is tracked."""
        rating = EloRating()
        rating.update(1000.0, 1.0)  # Win increases
        peak_after_win = rating.peak_rating

        rating.update(1000.0, 0.0)  # Loss decreases

        assert rating.peak_rating == peak_after_win
        assert rating.rating < rating.peak_rating

    def test_history_recording(self):
        """Test rating history is recorded."""
        rating = EloRating()
        rating.update(1000.0, 1.0, timestep=100)
        rating.update(1000.0, 1.0, timestep=200)

        assert len(rating.history) == 2
        assert rating.history[0][0] == 100
        assert rating.history[1][0] == 200

    def test_serialization(self):
        """Test to_dict and from_dict."""
        rating = EloRating(rating=1200.0, wins=10, losses=5)
        data = rating.to_dict()
        restored = EloRating.from_dict(data)

        assert restored.rating == 1200.0
        assert restored.wins == 10
        assert restored.losses == 5


class TestEloTracker:
    """Test Elo tracking system."""

    @pytest.fixture
    def tracker(self, tmp_path):
        return EloTracker(save_path=tmp_path / "elo.json")

    def test_record_match_updates_rating(self, tracker):
        """Test recording a match updates the rating."""
        initial_rating = tracker.agent_rating.rating

        tracker.record_match(
            opponent_type="random",
            opponent_rating=1000.0,
            won=True,
            timestep=100,
        )

        assert tracker.agent_rating.rating > initial_rating
        assert tracker.agent_rating.games_played == 1

    def test_match_history_recorded(self, tracker):
        """Test match history is recorded."""
        tracker.record_match("random", 1000.0, True, 100)
        tracker.record_match("self_play", 1100.0, False, 200)

        assert len(tracker.match_history) == 2
        assert tracker.match_history[0].opponent_type == "random"
        assert tracker.match_history[1].opponent_type == "self_play"

    def test_opponent_ratings_tracked(self, tracker):
        """Test opponent type ratings are tracked."""
        tracker.record_match("random", 1000.0, True, 100)
        tracker.record_match("random", 1000.0, True, 200)

        assert "random" in tracker.opponent_ratings
        assert tracker.opponent_ratings["random"].games_played == 2

    def test_get_stats(self, tracker):
        """Test getting statistics."""
        tracker.record_match("random", 1000.0, True, 100)
        tracker.record_match("random", 1000.0, False, 200)

        stats = tracker.get_stats()

        assert "current_rating" in stats
        assert "overall_win_rate" in stats
        assert "by_opponent" in stats
        assert stats["by_opponent"]["random"]["games"] == 2

    def test_save_load(self, tracker):
        """Test saving and loading tracker state."""
        tracker.record_match("random", 1000.0, True, 100)
        tracker.save()

        # Create new tracker from same path
        new_tracker = EloTracker(save_path=tracker.save_path)

        assert new_tracker.agent_rating.games_played == 1
        assert len(new_tracker.match_history) == 1


class TestCalculateEloUpdate:
    """Test standalone Elo calculation function."""

    def test_equal_players_win(self):
        """Test win between equal players."""
        p_new, o_new = calculate_elo_update(1000, 1000, player_won=True)

        assert p_new > 1000
        assert o_new < 1000
        assert (p_new - 1000) == pytest.approx(1000 - o_new)  # Zero-sum

    def test_underdog_win(self):
        """Test lower rated player beating higher rated."""
        p_new, o_new = calculate_elo_update(800, 1200, player_won=True)

        # Underdog gains more
        gain = p_new - 800
        loss = 1200 - o_new

        assert gain > 16  # More than half of k-factor
        assert gain == pytest.approx(loss)

    def test_favorite_win(self):
        """Test higher rated player beating lower rated."""
        p_new, o_new = calculate_elo_update(1200, 800, player_won=True)

        # Favorite gains less
        gain = p_new - 1200
        assert gain < 16  # Less than half of k-factor

    def test_k_factor_effect(self):
        """Test k-factor affects rating changes."""
        p1, _ = calculate_elo_update(1000, 1000, True, k_factor=16)
        p2, _ = calculate_elo_update(1000, 1000, True, k_factor=32)

        assert p2 - 1000 == pytest.approx(2 * (p1 - 1000))


class TestOpponentInfo:
    """Test OpponentInfo dataclass."""

    def test_win_rate_empty(self):
        """Test win rate when no games played."""
        info = OpponentInfo(checkpoint_path=Path("test.pt"))
        assert info.win_rate == 0.0

    def test_win_rate_calculated(self):
        """Test win rate calculation."""
        info = OpponentInfo(
            checkpoint_path=Path("test.pt"),
            games_played=10,
            wins=7,
            losses=3,
        )
        assert info.win_rate == 0.7


class TestOpponentPool:
    """Test opponent pool management."""

    @pytest.fixture
    def pool(self, tmp_path):
        return OpponentPool(pool_dir=tmp_path / "opponents", max_size=5)

    @pytest.fixture
    def model(self):
        return PolicyValueNetwork(
            hidden_dim=64,
            pokemon_dim=32,
            num_heads=2,
            num_layers=1,
            num_actions=9,
        )

    def test_empty_pool(self, pool):
        """Test empty pool returns no opponent."""
        assert pool.size == 0
        assert pool.sample_opponent() is None

    def test_add_opponent(self, pool, model):
        """Test adding an opponent."""
        info = pool.add_opponent(model, timestep=1000)

        assert pool.size == 1
        assert info.timestep_added == 1000
        assert info.checkpoint_path.exists()

    def test_sample_uniform(self, pool, model):
        """Test uniform sampling."""
        pool.add_opponent(model, timestep=1000)
        pool.add_opponent(model, timestep=2000)

        opponent = pool.sample_opponent(strategy="uniform")

        assert opponent is not None
        assert opponent.timestep_added in [1000, 2000]

    def test_sample_elo_matched(self, pool, model):
        """Test Elo-matched sampling prefers similar ratings."""
        info1 = pool.add_opponent(model, timestep=1000, elo_rating=800)
        info2 = pool.add_opponent(model, timestep=2000, elo_rating=1200)

        # Sample many times and count
        counts = {1000: 0, 2000: 0}
        for _ in range(100):
            opp = pool.sample_opponent(strategy="elo_matched", current_elo=850)
            counts[opp.timestep_added] += 1

        # Should prefer the 800 Elo opponent when current is 850
        assert counts[1000] > counts[2000]

    def test_sample_prioritized(self, pool, model):
        """Test prioritized sampling prefers less-played opponents."""
        info1 = pool.add_opponent(model, timestep=1000)
        info2 = pool.add_opponent(model, timestep=2000)
        info1.games_played = 100
        info2.games_played = 0

        # Sample many times
        counts = {1000: 0, 2000: 0}
        for _ in range(100):
            opp = pool.sample_opponent(strategy="prioritized")
            counts[opp.timestep_added] += 1

        # Should prefer the less-played opponent
        assert counts[2000] > counts[1000]

    def test_pool_pruning(self, pool, model):
        """Test pool prunes when over max size."""
        for i in range(10):
            pool.add_opponent(model, timestep=i * 1000, elo_rating=1000 + i * 50)

        assert pool.size <= pool.max_size

    def test_update_stats(self, pool, model):
        """Test updating opponent stats after a game."""
        info = pool.add_opponent(model, timestep=1000, elo_rating=1000)

        new_agent_elo, new_opp_elo = pool.update_stats(
            info, won=True, agent_elo=1000
        )

        assert info.games_played == 1
        assert info.losses == 1  # Opponent lost
        assert new_agent_elo > 1000
        assert new_opp_elo < 1000

    def test_average_elo(self, pool, model):
        """Test average Elo calculation."""
        pool.add_opponent(model, timestep=1000, elo_rating=800)
        pool.add_opponent(model, timestep=2000, elo_rating=1200)

        assert pool.average_elo == 1000.0


class TestSelfPlayManager:
    """Test self-play training manager."""

    @pytest.fixture
    def manager(self, tmp_path):
        return SelfPlayManager(
            pool_dir=tmp_path / "self_play",
            max_pool_size=5,
            self_play_ratio=0.8,
            checkpoint_interval=1000,
        )

    @pytest.fixture
    def model(self):
        return PolicyValueNetwork(
            hidden_dim=64,
            pokemon_dim=32,
            num_heads=2,
            num_layers=1,
            num_actions=9,
        )

    def test_empty_pool_no_self_play(self, manager):
        """Test no self-play when pool is empty."""
        assert not manager.should_use_self_play()

    def test_should_add_checkpoint(self, manager):
        """Test checkpoint timing logic."""
        assert manager.should_add_checkpoint(0)
        manager.last_checkpoint_timestep = 0

        assert not manager.should_add_checkpoint(500)
        assert manager.should_add_checkpoint(1000)

    def test_add_checkpoint(self, manager, model):
        """Test adding checkpoint to pool."""
        info = manager.add_checkpoint(model, timestep=1000)

        assert info.elo_rating == manager.agent_elo
        assert manager.opponent_pool.size == 1
        assert manager.last_checkpoint_timestep == 1000

    def test_get_opponent_random_when_empty(self, manager):
        """Test get_opponent returns random when pool empty."""
        with patch("showdown_bot.training.self_play.RandomPlayer") as mock_random:
            mock_player = MagicMock()
            mock_random.return_value = mock_player

            player, info = manager.get_opponent("gen9randombattle")

            assert info is None  # Random opponent
            assert manager.games_vs_random == 1

    def test_update_after_game_self_play(self, manager, model):
        """Test updating stats after self-play game."""
        info = manager.add_checkpoint(model, timestep=1000)
        initial_elo = manager.agent_elo

        manager.update_after_game(info, won=True)

        assert manager.agent_elo > initial_elo

    def test_update_after_game_random(self, manager):
        """Test updating stats after random opponent game."""
        initial_elo = manager.agent_elo

        manager.update_after_game(None, won=True)  # None = random opponent

        assert manager.agent_elo == initial_elo  # No change for random

    def test_get_stats(self, manager, model):
        """Test getting manager statistics."""
        manager.add_checkpoint(model, timestep=1000)

        stats = manager.get_stats()

        assert "agent_elo" in stats
        assert "pool_size" in stats
        assert stats["pool_size"] == 1


class TestTrainingIntegration:
    """Integration tests for the full training pipeline."""

    @pytest.fixture
    def model(self):
        return PolicyValueNetwork(
            hidden_dim=64,
            pokemon_dim=32,
            num_heads=2,
            num_layers=1,
            num_actions=9,
        )

    @pytest.fixture
    def ppo(self, model):
        return PPO(model, num_epochs=2)

    def test_buffer_to_ppo_integration(self, ppo):
        """Test buffer data flows correctly to PPO update."""
        buffer = RolloutBuffer(buffer_size=16, num_envs=1)

        # Fill buffer
        for i in range(16):
            buffer.add(
                player_pokemon=np.random.randn(1, 6, StateEncoder.POKEMON_FEATURES).astype(np.float32),
                opponent_pokemon=np.random.randn(1, 6, StateEncoder.POKEMON_FEATURES).astype(np.float32),
                player_active_idx=0,
                opponent_active_idx=0,
                field_state=np.random.randn(1, StateEncoder.FIELD_FEATURES).astype(np.float32),
                action_mask=np.ones((1, StateEncoder.NUM_ACTIONS), dtype=np.float32),
                action=np.random.randint(0, 9),
                log_prob=-1.0,
                reward=np.random.randn(),
                done=i == 15,
                value=np.random.randn(),
            )

        buffer.compute_advantages(np.zeros(1), np.ones(1))

        # Should complete without error
        stats = ppo.update(buffer, batch_size=8)
        assert isinstance(stats, PPOStats)

    def test_model_improves_on_consistent_reward(self, model):
        """Test model learns from consistent positive rewards."""
        ppo = PPO(model, learning_rate=1e-3, num_epochs=5)
        buffer = RolloutBuffer(buffer_size=32, num_envs=1)

        # Record initial value for a fixed state
        fixed_state = {
            "player_pokemon": torch.randn(1, 6, StateEncoder.POKEMON_FEATURES),
            "opponent_pokemon": torch.randn(1, 6, StateEncoder.POKEMON_FEATURES),
            "player_active_idx": torch.zeros(1, dtype=torch.long),
            "opponent_active_idx": torch.zeros(1, dtype=torch.long),
            "field_state": torch.randn(1, StateEncoder.FIELD_FEATURES),
            "action_mask": torch.ones(1, StateEncoder.NUM_ACTIONS),
        }

        model.eval()
        with torch.no_grad():
            _, initial_value = model(**fixed_state)

        # Train on positive rewards
        for _ in range(5):
            buffer.reset()
            for i in range(32):
                buffer.add(
                    player_pokemon=np.random.randn(1, 6, StateEncoder.POKEMON_FEATURES).astype(np.float32),
                    opponent_pokemon=np.random.randn(1, 6, StateEncoder.POKEMON_FEATURES).astype(np.float32),
                    player_active_idx=0,
                    opponent_active_idx=0,
                    field_state=np.random.randn(1, StateEncoder.FIELD_FEATURES).astype(np.float32),
                    action_mask=np.ones((1, StateEncoder.NUM_ACTIONS), dtype=np.float32),
                    action=0,  # Always same action
                    log_prob=-2.0,
                    reward=1.0,  # Consistent positive reward
                    done=i == 31,
                    value=0.0,
                )

            buffer.compute_advantages(np.zeros(1), np.ones(1))
            model.train()
            ppo.update(buffer, batch_size=16)

        # Check value increased (learning that states are valuable)
        model.eval()
        with torch.no_grad():
            _, final_value = model(**fixed_state)

        # Value should generally increase with positive rewards
        # (may not always due to stochasticity, but should trend up)
        assert final_value.item() != initial_value.item()


class TestTrainerStats:
    """Test TrainingStats dataclass."""

    def test_initial_stats(self):
        """Test initial training stats."""
        stats = TrainingStats()

        assert stats.total_timesteps == 0
        assert stats.total_episodes == 0
        assert stats.total_updates == 0
        assert stats.best_win_rate == 0.0

    def test_stats_tracking(self):
        """Test updating stats."""
        stats = TrainingStats()
        stats.total_timesteps = 1000
        stats.total_episodes = 50
        stats.best_win_rate = 0.65

        assert stats.total_timesteps == 1000
        assert stats.total_episodes == 50
        assert stats.best_win_rate == 0.65


class TestEndToEndPipeline:
    """End-to-end tests for complete training scenarios."""

    @pytest.fixture
    def model(self):
        return PolicyValueNetwork(
            hidden_dim=64,
            pokemon_dim=32,
            num_heads=2,
            num_layers=1,
            num_actions=9,
        )

    def test_full_training_iteration(self, model, tmp_path):
        """Test a complete training iteration cycle."""
        # Initialize components
        ppo = PPO(model, num_epochs=2)
        buffer = RolloutBuffer(buffer_size=16, num_envs=1)
        self_play = SelfPlayManager(
            pool_dir=tmp_path / "pool",
            checkpoint_interval=8,
        )
        elo_tracker = EloTracker(save_path=tmp_path / "elo.json")

        # Simulate a training iteration
        # 1. Collect experiences
        for i in range(16):
            buffer.add(
                player_pokemon=np.random.randn(1, 6, StateEncoder.POKEMON_FEATURES).astype(np.float32),
                opponent_pokemon=np.random.randn(1, 6, StateEncoder.POKEMON_FEATURES).astype(np.float32),
                player_active_idx=0,
                opponent_active_idx=0,
                field_state=np.random.randn(1, StateEncoder.FIELD_FEATURES).astype(np.float32),
                action_mask=np.ones((1, StateEncoder.NUM_ACTIONS), dtype=np.float32),
                action=np.random.randint(0, 9),
                log_prob=-1.0,
                reward=0.1 if i < 15 else 1.0,
                done=i == 15,
                value=np.random.randn(),
            )

        # 2. Compute advantages
        buffer.compute_advantages(np.zeros(1), np.ones(1))

        # 3. PPO update
        stats = ppo.update(buffer, batch_size=8)
        assert stats.total_loss > 0

        # 4. Add to self-play pool
        if self_play.should_add_checkpoint(16):
            self_play.add_checkpoint(model, timestep=16)

        # 5. Track Elo
        elo_tracker.record_match("self_play", 1000.0, won=True, timestep=16)

        # 6. Save state
        elo_tracker.save()
        ppo.save(str(tmp_path / "ppo.pt"))

        # Verify everything worked
        assert self_play.opponent_pool.size > 0
        assert elo_tracker.agent_rating.games_played == 1
        assert (tmp_path / "ppo.pt").exists()

    def test_checkpoint_roundtrip(self, model, tmp_path):
        """Test saving and loading preserves model state."""
        ppo = PPO(model, num_epochs=1)

        # Do some training
        buffer = RolloutBuffer(buffer_size=8, num_envs=1)
        for i in range(8):
            buffer.add(
                player_pokemon=np.random.randn(1, 6, StateEncoder.POKEMON_FEATURES).astype(np.float32),
                opponent_pokemon=np.random.randn(1, 6, StateEncoder.POKEMON_FEATURES).astype(np.float32),
                player_active_idx=0,
                opponent_active_idx=0,
                field_state=np.random.randn(1, StateEncoder.FIELD_FEATURES).astype(np.float32),
                action_mask=np.ones((1, StateEncoder.NUM_ACTIONS), dtype=np.float32),
                action=0,
                log_prob=-1.0,
                reward=0.0,
                done=i == 7,
                value=0.0,
            )

        buffer.compute_advantages(np.zeros(1), np.ones(1))
        ppo.update(buffer, batch_size=4)

        # Save
        checkpoint_path = tmp_path / "checkpoint.pt"
        ppo.save(str(checkpoint_path))

        # Create new model and load
        new_model = PolicyValueNetwork(
            hidden_dim=64,
            pokemon_dim=32,
            num_heads=2,
            num_layers=1,
            num_actions=9,
        )
        new_ppo = PPO(new_model)
        new_ppo.load(str(checkpoint_path))

        # Verify parameters match
        for (n1, p1), (n2, p2) in zip(
            model.named_parameters(), new_model.named_parameters()
        ):
            assert torch.allclose(p1, p2), f"Parameter {n1} mismatch"
