#!/usr/bin/env python3
"""Hyperparameter tuning script using Optuna."""

import argparse
import asyncio
import sys
import warnings
from pathlib import Path

# Suppress ROCm/PyTorch experimental feature warnings
warnings.filterwarnings("ignore", message=".*hipBLASLt.*")
warnings.filterwarnings("ignore", message=".*Flash attention.*experimental.*")
warnings.filterwarnings("ignore", message=".*Memory Efficient attention.*experimental.*")

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import optuna
from optuna.trial import Trial
import torch
import numpy as np

from showdown_bot.config import TrainingConfig
from showdown_bot.models.network import PolicyValueNetwork
from showdown_bot.training.trainer import Trainer


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def create_model_with_config(config: TrainingConfig, device: torch.device) -> PolicyValueNetwork:
    """Create a model with the given config."""
    return PolicyValueNetwork(
        num_pokemon=12,
        pokemon_features=24,
        field_features=26,
        num_actions=9,
        hidden_dim=config.hidden_dim,
        num_heads=config.num_attention_heads,
        num_layers=config.num_transformer_layers,
        embedding_dim=config.embedding_dim,
        num_species=config.num_species,
        num_moves=config.num_moves,
        num_abilities=config.num_abilities,
        num_items=config.num_items,
    ).to(device)


async def evaluate_hyperparameters(
    trial: Trial,
    timesteps: int,
    device: torch.device,
) -> float:
    """Train with given hyperparameters and return win rate."""
    # Sample hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    gamma = trial.suggest_float("gamma", 0.95, 0.999)
    gae_lambda = trial.suggest_float("gae_lambda", 0.9, 0.99)
    clip_epsilon = trial.suggest_float("clip_epsilon", 0.1, 0.3)
    entropy_coef = trial.suggest_float("entropy_coef", 0.001, 0.1, log=True)
    value_coef = trial.suggest_float("value_coef", 0.25, 1.0)
    num_epochs = trial.suggest_int("num_epochs", 2, 10)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    rollout_steps = trial.suggest_categorical("rollout_steps", [512, 1024, 2048, 4096])

    # Create config with sampled hyperparameters
    config = TrainingConfig(
        learning_rate=learning_rate,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_epsilon=clip_epsilon,
        entropy_coef=entropy_coef,
        value_coef=value_coef,
        num_epochs=num_epochs,
        batch_size=batch_size,
        rollout_steps=rollout_steps,
    )

    # Create model and trainer
    model = create_model_with_config(config, device)

    # Create unique directories for this trial
    trial_dir = f"data/tuning/trial_{trial.number}"

    trainer = Trainer(
        model=model,
        device=device,
        save_dir=f"{trial_dir}/checkpoints",
        log_dir=f"{trial_dir}/logs",
        self_play_dir=f"{trial_dir}/opponents",
        use_self_play=False,  # Faster tuning without self-play
        config=config,
    )

    try:
        # Train for specified timesteps
        await trainer.train(
            total_timesteps=timesteps,
            eval_interval=timesteps + 1,  # No intermediate eval
            save_interval=timesteps + 1,  # No intermediate save
        )

        # Return the best win rate achieved
        win_rate = trainer._best_win_rate

        # Report intermediate value for pruning
        trial.report(win_rate, step=timesteps)

        if trial.should_prune():
            raise optuna.TrialPruned()

        return win_rate

    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        return 0.0


def objective(trial: Trial, timesteps: int, device: torch.device) -> float:
    """Optuna objective function."""
    return asyncio.run(evaluate_hyperparameters(trial, timesteps, device))


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="Tune hyperparameters for the Pokemon Showdown bot",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--trials",
        "-n",
        type=int,
        default=20,
        help="Number of trials to run",
    )
    parser.add_argument(
        "--timesteps",
        "-t",
        type=int,
        default=20000,
        help="Timesteps per trial (shorter = faster but less accurate)",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default="showdown_bot_tuning",
        help="Name for the Optuna study",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default="sqlite:///data/tuning/optuna.db",
        help="Optuna storage URL",
    )
    parser.add_argument(
        "--load-if-exists",
        action="store_true",
        help="Continue existing study if it exists",
    )
    args = parser.parse_args()

    # Create tuning directory
    Path("data/tuning").mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"Using device: {device}")

    print("=" * 60)
    print("Pokemon Showdown RL Bot - Hyperparameter Tuning")
    print("=" * 60)
    print(f"Trials: {args.trials}")
    print(f"Timesteps per trial: {args.timesteps:,}")
    print(f"Study name: {args.study_name}")
    print("=" * 60)

    # Create or load study
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="maximize",  # Maximize win rate
        load_if_exists=args.load_if_exists,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
    )

    # Run optimization
    study.optimize(
        lambda trial: objective(trial, args.timesteps, device),
        n_trials=args.trials,
        show_progress_bar=True,
    )

    # Print results
    print("\n" + "=" * 60)
    print("Tuning Complete!")
    print("=" * 60)

    print("\nBest trial:")
    best_trial = study.best_trial
    print(f"  Win rate: {best_trial.value * 100:.1f}%")
    print(f"  Trial number: {best_trial.number}")

    print("\nBest hyperparameters:")
    for key, value in best_trial.params.items():
        print(f"  {key}: {value}")

    # Save best config
    best_config_path = Path("data/tuning/best_config.py")
    with open(best_config_path, "w") as f:
        f.write("# Best hyperparameters found by Optuna tuning\n")
        f.write("# Copy these to config.py or use --config flag\n\n")
        f.write("BEST_HYPERPARAMETERS = {\n")
        for key, value in best_trial.params.items():
            if isinstance(value, str):
                f.write(f'    "{key}": "{value}",\n')
            else:
                f.write(f'    "{key}": {value},\n')
        f.write("}\n")

    print(f"\nBest config saved to: {best_config_path}")

    # Show top 5 trials
    print("\nTop 5 trials:")
    trials_df = study.trials_dataframe()
    trials_df = trials_df.sort_values("value", ascending=False).head(5)
    for _, row in trials_df.iterrows():
        print(f"  Trial {int(row['number'])}: {row['value'] * 100:.1f}%")


if __name__ == "__main__":
    main()
