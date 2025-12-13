import os
import sys
import json
import csv
from dataclasses import dataclass
from typing import Dict, Any

import numpy as np
import torch
import gymnasium as gym


# ---------- Import PPOAgent and create_wds_env from your existing file ----------

# Adjust this import if your training file has a different name
from train_ppo_wds_pytorch import PPOAgent, create_wds_env


@dataclass
class EvalConfig:
    model_path: str = "models/ppo_wds_pytorch.pth"
    n_episodes: int = 50
    output_dir: str = "evaluations"


def load_config_for_model(model_path: str) -> Dict[str, Any]:
    """Load JSON config associated with a given .pth model."""
    config_path = model_path.replace(".pth", "_config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        return json.load(f)


def evaluate_and_log(cfg: EvalConfig):
    """Run evaluation, save per-episode stats and one trajectory to CSV."""

    print("=" * 70)
    print(f"Evaluating model: {cfg.model_path}")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1) Load training config for obs_dim/action_dim
    train_config = load_config_for_model(cfg.model_path)
    obs_dim = train_config["obs_dim"]
    action_dim = train_config["action_dim"]

    # 2) Create environment
    env = create_wds_env()
    print("Environment created:", env)

    # 3) Re-create agent and load weights
    agent = PPOAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        device=device,
    )
    agent.load_pth(cfg.model_path, map_location=device)
    agent.model.eval()
    print("Model loaded.")

    # 4) Prepare output directory and CSV files
    os.makedirs(cfg.output_dir, exist_ok=True)

    stats_path = os.path.join(cfg.output_dir, "episode_stats.csv")
    traj_path = os.path.join(cfg.output_dir, "episode0_trajectory.csv")

    # Per-episode statistics CSV: episode, total_reward, length
    with open(stats_path, "w", newline="") as f_stats:
        stats_writer = csv.writer(f_stats)
        stats_writer.writerow(["episode", "total_reward", "length"])

    # Trajectory CSV (for episode 0): step, reward, tank_level_0
    with open(traj_path, "w", newline="") as f_traj:
        traj_writer = csv.writer(f_traj)
        traj_writer.writerow(["step", "reward", "tank_level_0"])

    episode_rewards = []
    episode_lengths = []

    for ep in range(cfg.n_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        length = 0
        step_idx = 0

        while not done:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                action, _, _, _ = agent.model.get_action_and_value(obs_tensor)

            next_obs, reward, terminated, truncated, _ = env.step(action.cpu().item())
            done = terminated or truncated

            total_reward += reward
            length += 1

            # Log full trajectory only for episode 0 (for water-level plot)
            if ep == 0:
                # assuming first element in observation is a representative tank level
                tank_level_0 = float(obs[0])
                with open(traj_path, "a", newline="") as f_traj:
                    traj_writer = csv.writer(f_traj)
                    traj_writer.writerow([step_idx, reward, tank_level_0])

            obs = next_obs
            step_idx += 1

        episode_rewards.append(total_reward)
        episode_lengths.append(length)

        with open(stats_path, "a", newline="") as f_stats:
            stats_writer = csv.writer(f_stats)
            stats_writer.writerow([ep, total_reward, length])

        print(f"Episode {ep+1}/{cfg.n_episodes}: Reward={total_reward:.2f}, Length={length}")

    # Print summary
    mean_r = float(np.mean(episode_rewards))
    std_r = float(np.std(episode_rewards))
    mean_l = float(np.mean(episode_lengths))
    std_l = float(np.std(episode_lengths))

    print("\n" + "=" * 70)
    print("Evaluation summary:")
    print(f"  Mean Reward: {mean_r:.2f} ± {std_r:.2f}")
    print(f"  Mean Length: {mean_l:.1f} ± {std_l:.1f}")
    print(f"\nSaved per-episode stats to: {stats_path}")
    print(f"Saved episode 0 trajectory to: {traj_path}")
    print("=" * 70)

    env.close()


if __name__ == "__main__":
    # simple CLI: you can modify values here or later add argparse
    cfg = EvalConfig(
        model_path="models/ppo_wds_pytorch.pth",
        n_episodes=50,
        output_dir="evaluations",
    )
    evaluate_and_log(cfg)
