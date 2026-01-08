"""Tests for the coaching server and browser state encoder."""

import pytest
import numpy as np
import torch
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from showdown_bot.environment.state_encoder import StateEncoder


class TestBrowserStateEncoder:
    """Test the BrowserStateEncoder that converts browser JSON to model tensors."""

    @pytest.fixture
    def encoder(self):
        """Create encoder with CPU device."""
        # Import here to avoid issues if flask isn't installed
        from coach_server import BrowserStateEncoder
        return BrowserStateEncoder(torch.device("cpu"))

    @pytest.fixture
    def sample_browser_state(self):
        """Sample battle state as sent by the browser extension."""
        return {
            "myTeam": [
                {
                    "species": "Pikachu",
                    "hp": 80,
                    "maxHp": 100,
                    "status": None,
                    "active": True,
                    "fainted": False,
                    "moves": ["thunderbolt", "quick attack", "iron tail", "volt tackle"],
                    "ability": "Static",
                    "item": "Light Ball"
                },
                {
                    "species": "Charizard",
                    "hp": 150,
                    "maxHp": 150,
                    "status": None,
                    "active": False,
                    "fainted": False,
                    "moves": ["flamethrower", "air slash", "dragon pulse", "roost"],
                    "ability": "Blaze",
                    "item": None
                },
            ],
            "myActive": {
                "species": "Pikachu",
                "hp": 80,
                "maxHp": 100,
                "status": None,
                "boosts": {"atk": 1, "spa": 0},
                "volatiles": []
            },
            "opponentTeam": [
                {
                    "species": "Blastoise",
                    "hp": 90,
                    "maxHp": 100,
                    "status": "brn",
                    "fainted": False,
                    "revealed": True
                }
            ],
            "opponentActive": {
                "species": "Blastoise",
                "hp": 90,
                "maxHp": 100,
                "status": "brn",
                "boosts": {},
                "volatiles": []
            },
            "weather": "sunnyday",
            "terrain": None,
            "availableMoves": [
                {"id": "thunderbolt", "name": "Thunderbolt", "pp": 15, "maxpp": 15, "disabled": False, "index": 0},
                {"id": "quickattack", "name": "Quick Attack", "pp": 30, "maxpp": 30, "disabled": False, "index": 1},
                {"id": "irontail", "name": "Iron Tail", "pp": 15, "maxpp": 15, "disabled": False, "index": 2},
                {"id": "volttackle", "name": "Volt Tackle", "pp": 15, "maxpp": 15, "disabled": False, "index": 3},
            ],
            "availableSwitches": [
                {"species": "Charizard", "hp": 150, "maxHp": 150, "index": 1}
            ],
            "turn": 5,
            "waitingForChoice": True
        }

    def test_encode_output_shapes(self, encoder, sample_browser_state):
        """Test that encoded state has correct tensor shapes."""
        result = encoder.encode(sample_browser_state)

        assert result["player_pokemon"].shape == (1, 6, StateEncoder.POKEMON_FEATURES)
        assert result["opponent_pokemon"].shape == (1, 6, StateEncoder.POKEMON_FEATURES)
        assert result["player_active_idx"].shape == (1,)
        assert result["opponent_active_idx"].shape == (1,)
        assert result["field_state"].shape == (1, StateEncoder.FIELD_FEATURES)
        assert result["action_mask"].shape == (1, StateEncoder.NUM_ACTIONS)

    def test_encode_tensors_are_finite(self, encoder, sample_browser_state):
        """Test that all encoded values are finite."""
        result = encoder.encode(sample_browser_state)

        for key, tensor in result.items():
            if tensor.is_floating_point():
                assert torch.isfinite(tensor).all(), f"{key} contains non-finite values"

    def test_action_mask_reflects_moves(self, encoder, sample_browser_state):
        """Test action mask correctly enables available moves."""
        result = encoder.encode(sample_browser_state)
        mask = result["action_mask"].numpy()[0]

        # First 4 moves should be enabled (indices 0-3)
        assert mask[0] == 1.0
        assert mask[1] == 1.0
        assert mask[2] == 1.0
        assert mask[3] == 1.0

        # First switch should be enabled (index 4)
        assert mask[4] == 1.0

        # Remaining switches should be disabled
        assert mask[5] == 0.0
        assert mask[6] == 0.0

    def test_action_mask_respects_disabled(self, encoder, sample_browser_state):
        """Test disabled moves are masked out."""
        sample_browser_state["availableMoves"][1]["disabled"] = True
        result = encoder.encode(sample_browser_state)
        mask = result["action_mask"].numpy()[0]

        assert mask[0] == 1.0
        assert mask[1] == 0.0  # Disabled
        assert mask[2] == 1.0

    def test_hp_fraction_encoding(self, encoder, sample_browser_state):
        """Test HP is correctly encoded as fraction."""
        result = encoder.encode(sample_browser_state)
        player = result["player_pokemon"].numpy()[0]

        # First Pokemon (Pikachu) has 80/100 HP
        # HP fraction is the first feature
        assert abs(player[0, 0] - 0.8) < 0.01

    def test_empty_team_handling(self, encoder):
        """Test encoder handles empty team gracefully."""
        state = {
            "myTeam": [],
            "opponentTeam": [],
            "availableMoves": [],
            "availableSwitches": [],
            "turn": 1,
            "waitingForChoice": True
        }

        result = encoder.encode(state)

        # Should produce valid (zero-filled) tensors
        assert result["player_pokemon"].shape == (1, 6, StateEncoder.POKEMON_FEATURES)
        assert torch.isfinite(result["player_pokemon"]).all()

    def test_find_active_idx(self, encoder, sample_browser_state):
        """Test active Pokemon index is correctly identified."""
        result = encoder.encode(sample_browser_state)

        # First Pokemon is active
        assert result["player_active_idx"].item() == 0

    def test_weather_encoding(self, encoder, sample_browser_state):
        """Test weather is encoded in field state."""
        result = encoder.encode(sample_browser_state)
        field = result["field_state"].numpy()[0]

        # Weather is encoded at the start of field features
        # sunnyday should have a 1.0 somewhere in the weather section
        weather_section = field[:6]  # 5 weathers + 1 none
        assert weather_section.sum() == 1.0  # One-hot

    def test_status_encoding(self, encoder):
        """Test status conditions are encoded."""
        state = {
            "myTeam": [{
                "species": "Pikachu",
                "hp": 50,
                "maxHp": 100,
                "status": "par",
                "active": True,
                "fainted": False,
                "moves": []
            }],
            "opponentTeam": [],
            "availableMoves": [],
            "availableSwitches": [],
            "turn": 1,
            "waitingForChoice": True
        }

        from coach_server import BrowserStateEncoder
        encoder = BrowserStateEncoder(torch.device("cpu"))
        result = encoder.encode(state)

        # The encoded tensor should have status info
        assert result["player_pokemon"].shape == (1, 6, StateEncoder.POKEMON_FEATURES)


class TestHeuristicRecommendations:
    """Test the fallback heuristic recommendation system."""

    @pytest.fixture
    def sample_state(self):
        return {
            "myActive": {"hp": 100, "maxHp": 100},
            "opponentActive": {"hp": 80, "maxHp": 100, "status": None},
            "availableMoves": [
                {"name": "Thunderbolt", "id": "thunderbolt", "pp": 15, "index": 0},
                {"name": "Quick Attack", "id": "quickattack", "pp": 30, "index": 1},
                {"name": "Swords Dance", "id": "swordsdance", "pp": 20, "index": 2},
                {"name": "Thunder Wave", "id": "thunderwave", "pp": 20, "index": 3},
            ],
            "availableSwitches": [
                {"species": "Charizard", "hp": 150, "maxHp": 150, "index": 1}
            ],
            "turn": 1,
            "waitingForChoice": True
        }

    def test_returns_list(self, sample_state):
        from coach_server import _get_heuristic_recommendations
        result = _get_heuristic_recommendations(sample_state)

        assert isinstance(result, list)
        assert len(result) <= 4

    def test_each_recommendation_has_required_fields(self, sample_state):
        from coach_server import _get_heuristic_recommendations
        result = _get_heuristic_recommendations(sample_state)

        for rec in result:
            assert "name" in rec
            assert "score" in rec
            assert "index" in rec

    def test_scores_are_normalized(self, sample_state):
        from coach_server import _get_heuristic_recommendations
        result = _get_heuristic_recommendations(sample_state)

        for rec in result:
            assert 0.0 <= rec["score"] <= 1.0

    def test_sorted_by_score(self, sample_state):
        from coach_server import _get_heuristic_recommendations
        result = _get_heuristic_recommendations(sample_state)

        scores = [r["score"] for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_setup_moves_boosted_when_healthy(self, sample_state):
        """When HP is high, setup moves should score higher."""
        from coach_server import _get_heuristic_recommendations

        # High HP
        sample_state["myActive"]["hp"] = 100
        sample_state["myActive"]["maxHp"] = 100

        result = _get_heuristic_recommendations(sample_state)

        # Find Swords Dance
        sd_rec = next((r for r in result if "sword" in r["name"].lower()), None)
        assert sd_rec is not None
        assert sd_rec["score"] >= 0.6  # Should be boosted

    def test_priority_moves_boosted_at_low_hp(self):
        """When HP is low, priority moves should score higher."""
        from coach_server import _get_heuristic_recommendations

        state = {
            "myActive": {"hp": 20, "maxHp": 100},  # Low HP
            "opponentActive": {"hp": 80, "maxHp": 100},
            "availableMoves": [
                {"name": "Hyper Beam", "id": "hyperbeam", "pp": 5, "index": 0},
                {"name": "Quick Attack", "id": "quickattack", "pp": 30, "index": 1},
            ],
            "availableSwitches": [],
            "turn": 1,
            "waitingForChoice": True
        }

        result = _get_heuristic_recommendations(state)

        # Quick Attack should have a boosted score due to low HP
        qa_rec = next((r for r in result if "quick" in r["name"].lower()), None)
        hb_rec = next((r for r in result if "hyper" in r["name"].lower()), None)

        assert qa_rec is not None
        # Priority move should be recommended higher at low HP
        assert qa_rec["score"] >= 0.7

    def test_status_moves_boosted_if_no_status(self, sample_state):
        """Status moves should be boosted if opponent has no status."""
        from coach_server import _get_heuristic_recommendations

        sample_state["opponentActive"]["status"] = None
        result = _get_heuristic_recommendations(sample_state)

        tw_rec = next((r for r in result if "thunder wave" in r["name"].lower()), None)
        assert tw_rec is not None
        assert tw_rec["score"] >= 0.6

    def test_disabled_moves_excluded(self, sample_state):
        """Disabled moves should not appear in recommendations."""
        from coach_server import _get_heuristic_recommendations

        sample_state["availableMoves"][0]["disabled"] = True
        result = _get_heuristic_recommendations(sample_state)

        # Thunderbolt should not be in results
        tb_rec = next((r for r in result if r["id"] == "thunderbolt"), None)
        assert tb_rec is None


class TestCoachServerIntegration:
    """Integration tests for the coach server Flask app."""

    @pytest.fixture
    def client(self):
        """Create test client for Flask app."""
        from coach_server import app
        app.config['TESTING'] = True
        with app.test_client() as client:
            yield client

    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get('/health')
        assert response.status_code == 200

        data = response.get_json()
        assert "status" in data
        assert data["status"] == "ok"

    def test_recommend_empty_state(self, client):
        """Test recommend endpoint handles empty state."""
        # Empty state triggers 400 error (no state provided)
        response = client.post('/recommend', json={})
        # Server checks for empty/falsy state and returns 400
        assert response.status_code == 400

    def test_recommend_not_waiting(self, client):
        """Test recommend returns empty when not waiting for choice."""
        response = client.post('/recommend', json={
            "waitingForChoice": False,
            "turn": 1
        })
        assert response.status_code == 200

        data = response.get_json()
        assert data["moves"] == []

    def test_recommend_returns_moves(self, client):
        """Test recommend endpoint returns move recommendations."""
        state = {
            "myTeam": [{
                "species": "Pikachu",
                "hp": 100,
                "maxHp": 100,
                "active": True,
                "fainted": False,
                "moves": ["thunderbolt"]
            }],
            "myActive": {"hp": 100, "maxHp": 100},
            "opponentTeam": [],
            "opponentActive": {"hp": 100, "maxHp": 100},
            "availableMoves": [
                {"name": "Thunderbolt", "id": "thunderbolt", "pp": 15, "index": 0, "disabled": False}
            ],
            "availableSwitches": [],
            "turn": 1,
            "waitingForChoice": True
        }

        response = client.post('/recommend', json=state)
        assert response.status_code == 200

        data = response.get_json()
        assert "moves" in data
        assert len(data["moves"]) >= 1

    def test_recommend_includes_turn(self, client):
        """Test recommend response includes turn number."""
        state = {
            "myTeam": [],
            "opponentTeam": [],
            "availableMoves": [
                {"name": "Tackle", "id": "tackle", "pp": 35, "index": 0, "disabled": False}
            ],
            "availableSwitches": [],
            "turn": 42,
            "waitingForChoice": True
        }

        response = client.post('/recommend', json=state)
        data = response.get_json()

        assert data["turn"] == 42


class TestModelIntegration:
    """Test integration with the trained model."""

    @pytest.fixture
    def network(self):
        """Create a fresh network for testing."""
        from showdown_bot.models.network import PolicyValueNetwork
        return PolicyValueNetwork.from_config()

    def test_browser_encoder_output_compatible_with_model(self, network):
        """Test that BrowserStateEncoder output works with the model."""
        from coach_server import BrowserStateEncoder

        encoder = BrowserStateEncoder(torch.device("cpu"))
        state = {
            "myTeam": [{
                "species": "Pikachu",
                "hp": 100,
                "maxHp": 100,
                "active": True,
                "fainted": False,
                "moves": []
            }],
            "opponentTeam": [{
                "species": "Charizard",
                "hp": 100,
                "maxHp": 100,
                "fainted": False
            }],
            "availableMoves": [
                {"name": "Thunderbolt", "id": "thunderbolt", "pp": 15, "disabled": False, "index": 0}
            ],
            "availableSwitches": [],
            "turn": 1,
            "waitingForChoice": True
        }

        tensors = encoder.encode(state)

        # Run through model
        network.eval()
        with torch.no_grad():
            logits, value = network.forward(
                tensors["player_pokemon"],
                tensors["opponent_pokemon"],
                tensors["player_active_idx"],
                tensors["opponent_active_idx"],
                tensors["field_state"],
                tensors["action_mask"],
            )

        assert logits.shape == (1, StateEncoder.NUM_ACTIONS)
        assert value.shape == (1, 1)
        assert torch.isfinite(logits[tensors["action_mask"] > 0]).all()
