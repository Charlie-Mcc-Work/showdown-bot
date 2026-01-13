"""Tests for the Trainer module - TrainingDisplay and related utilities."""

import io
import sys
import time
from unittest.mock import MagicMock, patch

import pytest
import torch

from showdown_bot.training.trainer import TrainingDisplay, MemoryMonitor


class TestTrainingDisplayInit:
    """Tests for TrainingDisplay initialization."""

    def test_default_initialization(self):
        """Test display initializes with correct defaults."""
        display = TrainingDisplay(total_timesteps=1000000)
        assert display.total_timesteps == 1000000
        assert display.start_timesteps == 0
        assert display.current_timesteps == 0
        assert display.iteration == 0
        assert display._speed_history == []

    def test_initialization_with_start_timesteps(self):
        """Test display initializes with custom start timesteps."""
        display = TrainingDisplay(total_timesteps=1000000, start_timesteps=500000)
        assert display.total_timesteps == 1000000
        assert display.start_timesteps == 500000
        assert display.current_timesteps == 500000

    def test_tty_detection_attribute_exists(self):
        """Test that TTY detection attribute exists."""
        display = TrainingDisplay(total_timesteps=1000)
        assert hasattr(display, "_is_tty")
        # Value depends on test runner, just check it's a bool
        assert isinstance(display._is_tty, bool)


class TestTrainingDisplayFormatting:
    """Tests for TrainingDisplay formatting methods."""

    def test_format_number_millions(self):
        """Test formatting numbers in millions."""
        display = TrainingDisplay(total_timesteps=1000)
        assert display._format_number(5000000) == "5.00M"
        assert display._format_number(12500000) == "12.50M"
        assert display._format_number(1234567) == "1.23M"

    def test_format_number_thousands(self):
        """Test formatting numbers in thousands."""
        display = TrainingDisplay(total_timesteps=1000)
        assert display._format_number(5000) == "5.0K"
        assert display._format_number(12500) == "12.5K"
        assert display._format_number(999000) == "999.0K"

    def test_format_number_small(self):
        """Test formatting small numbers."""
        display = TrainingDisplay(total_timesteps=1000)
        assert display._format_number(500) == "500"
        assert display._format_number(0) == "0"
        assert display._format_number(999) == "999"

    def test_format_time_seconds(self):
        """Test formatting time in seconds."""
        display = TrainingDisplay(total_timesteps=1000)
        assert display._format_time(30) == "30s"
        assert display._format_time(59) == "59s"
        assert display._format_time(5.5) == "6s"

    def test_format_time_minutes(self):
        """Test formatting time in minutes."""
        display = TrainingDisplay(total_timesteps=1000)
        assert display._format_time(60) == "1m"
        assert display._format_time(120) == "2m"
        assert display._format_time(3540) == "59m"  # 59 * 60 = 3540

    def test_format_time_hours(self):
        """Test formatting time in hours and minutes."""
        display = TrainingDisplay(total_timesteps=1000)
        assert display._format_time(3600) == "1h0m"
        assert display._format_time(7200) == "2h0m"
        assert display._format_time(5400) == "1h30m"
        assert display._format_time(86400) == "24h0m"


class TestTrainingDisplaySpeed:
    """Tests for TrainingDisplay speed calculation."""

    def test_speed_calculation_empty(self):
        """Test speed returns 0 with no history."""
        display = TrainingDisplay(total_timesteps=1000)
        assert display._get_speed() == 0.0

    def test_speed_history_tracks(self):
        """Test speed history is tracked correctly."""
        display = TrainingDisplay(total_timesteps=1000000)
        display._speed_history = [100.0, 200.0, 150.0]
        assert display._get_speed() == 150.0  # Average

    def test_speed_history_window_limit(self):
        """Test speed history respects window limit."""
        display = TrainingDisplay(total_timesteps=1000000)
        display._speed_window = 5
        # Add more than window size
        for i in range(10):
            display._speed_history.append(100.0)
            if len(display._speed_history) > display._speed_window:
                display._speed_history.pop(0)
        assert len(display._speed_history) == 5


class TestTrainingDisplayUpdate:
    """Tests for TrainingDisplay update method."""

    def test_update_increments_iteration(self):
        """Test update increments iteration counter."""
        display = TrainingDisplay(total_timesteps=1000)
        display._is_tty = False  # Force non-TTY to avoid terminal output
        display._last_log_time = 0  # Force logging

        with patch("sys.stdout", new_callable=io.StringIO):
            display.update(100)
        assert display.iteration == 1

        with patch("sys.stdout", new_callable=io.StringIO):
            display.update(200)
        assert display.iteration == 2

    def test_update_tracks_timesteps(self):
        """Test update tracks current timesteps."""
        display = TrainingDisplay(total_timesteps=1000)
        display._is_tty = False
        display._last_log_time = 0

        with patch("sys.stdout", new_callable=io.StringIO):
            display.update(500)
        assert display.current_timesteps == 500

    def test_update_tty_mode_uses_carriage_return(self):
        """Test TTY mode uses carriage return for in-place updates."""
        display = TrainingDisplay(total_timesteps=1000)
        display._is_tty = True

        output = io.StringIO()
        with patch("sys.stdout", output):
            display.update(500)

        result = output.getvalue()
        assert result.startswith("\r")
        assert "500" in result or "0.50K" in result

    def test_update_non_tty_mode_uses_newlines(self):
        """Test non-TTY mode uses newlines instead of carriage returns."""
        display = TrainingDisplay(total_timesteps=1000)
        display._is_tty = False
        display._last_log_time = 0  # Force immediate logging
        display._log_interval = 0  # No delay

        output = io.StringIO()
        with patch("sys.stdout", output):
            display.update(500)

        result = output.getvalue()
        # Should NOT start with carriage return when not TTY
        assert not result.startswith("\r")

    def test_update_non_tty_respects_log_interval(self):
        """Test non-TTY mode respects log interval."""
        display = TrainingDisplay(total_timesteps=1000)
        display._is_tty = False
        display._log_interval = 100  # Long interval
        display._last_log_time = time.time()  # Recent log

        output = io.StringIO()
        with patch("sys.stdout", output):
            display.update(500)

        # Should not log because interval not passed
        assert output.getvalue() == ""

    def test_update_with_rollout_stats(self):
        """Test update with rollout statistics."""
        display = TrainingDisplay(total_timesteps=1000)
        display._is_tty = True

        rollout_stats = {"win_rate": 0.75, "episode_rewards": [1.0, 2.0]}

        output = io.StringIO()
        with patch("sys.stdout", output):
            display.update(500, rollout_stats=rollout_stats)

        result = output.getvalue()
        assert "Win:75%" in result or "Win: 75%" in result

    def test_update_with_skill(self):
        """Test update with skill rating."""
        display = TrainingDisplay(total_timesteps=1000)
        display._is_tty = True

        output = io.StringIO()
        with patch("sys.stdout", output):
            display.update(500, skill=25000.0)

        result = output.getvalue()
        assert "25000" in result

    def test_close_prints_newline_when_active(self):
        """Test close prints newline when in-place update was active."""
        display = TrainingDisplay(total_timesteps=1000)
        display._in_place_active = True

        output = io.StringIO()
        with patch("sys.stdout", output):
            display.close()

        assert output.getvalue() == "\n"
        assert not display._in_place_active

    def test_close_no_newline_when_inactive(self):
        """Test close doesn't print newline when not active."""
        display = TrainingDisplay(total_timesteps=1000)
        display._in_place_active = False

        output = io.StringIO()
        with patch("sys.stdout", output):
            display.close()

        assert output.getvalue() == ""


class TestTrainingDisplayProgressBar:
    """Tests for progress bar rendering."""

    def test_progress_bar_empty_at_start(self):
        """Test progress bar is empty at start."""
        display = TrainingDisplay(total_timesteps=1000)
        display._is_tty = True

        output = io.StringIO()
        with patch("sys.stdout", output):
            display.update(0)

        result = output.getvalue()
        # Should have empty progress bar (all dashes)
        assert "--------------------" in result

    def test_progress_bar_half_filled(self):
        """Test progress bar is half filled at 50%."""
        display = TrainingDisplay(total_timesteps=1000)
        display._is_tty = True

        output = io.StringIO()
        with patch("sys.stdout", output):
            display.update(500)

        result = output.getvalue()
        # Should have ~50% filled bar
        assert "==========" in result

    def test_progress_bar_full_at_end(self):
        """Test progress bar is full at completion."""
        display = TrainingDisplay(total_timesteps=1000)
        display._is_tty = True

        output = io.StringIO()
        with patch("sys.stdout", output):
            display.update(1000)

        result = output.getvalue()
        # Should have full progress bar
        assert "====================" in result


class TestMemoryMonitorBasic:
    """Additional tests for MemoryMonitor."""

    def test_format_memory_various_sizes(self):
        """Test memory formatting for various sizes."""
        monitor = MemoryMonitor()
        assert "B" in monitor.format_memory(100)
        assert "KB" in monitor.format_memory(2048)
        assert "MB" in monitor.format_memory(5 * 1024 * 1024)
        assert "GB" in monitor.format_memory(2 * 1024 * 1024 * 1024)

    def test_check_memory_status_types(self):
        """Test check_memory returns valid status types."""
        monitor = MemoryMonitor()
        status, percent = monitor.check_memory()
        assert status in ("ok", "soft_limit", "hard_limit")
        assert isinstance(percent, float)
        assert 0.0 <= percent <= 100.0

    def test_memory_status_string_format(self):
        """Test memory status string format."""
        monitor = MemoryMonitor()
        status_str = monitor.get_memory_status_string()
        assert "/" in status_str
        assert "%" in status_str
