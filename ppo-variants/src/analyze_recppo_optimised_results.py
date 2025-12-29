#!/usr/bin/env python3

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

sns.set(style="whitegrid", context="talk")

def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)

# --------- Episode-level plots ---------

def plot_episode_returns(df_ep, outdir):
    _ensure_dir(outdir)

    # 1) Episode returns vs episode index
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_ep, x="episode", y="ep_reward")
    plt.xlabel("Episode")
    plt.ylabel("Episode return")
    plt.title("recPPO (optimized/tuned) – episode returns")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "recppo_tuned_episode_returns.png"))
    plt.close()

    # 2) Episode lengths vs episode index
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_ep, x="episode", y="ep_len")
    plt.xlabel("Episode")
    plt.ylabel("Episode length (steps)")
    plt.title("recPPO (optimized/tuned) – episode lengths")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "recppo_tuned_episode_lengths.png"))
    plt.close()

    # 3) Histogram of returns
    plt.figure(figsize=(10, 6))
    sns.histplot(df_ep["ep_reward"], kde=True, bins=30)
    plt.xlabel("Episode return")
    plt.ylabel("Count")
    plt.title("recPPO (optimized/tuned) – distribution of episode returns")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "recppo_tuned_ep_return_distribution.png"))
    plt.close()

# --------- Per-step rewards (proxy curve) ---------

def plot_step_rewards(df_train, outdir):
    """
    df_train has columns: ['timestep', 'episode', 'reward']

    """
    _ensure_dir(outdir)

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_train, x="timestep", y="reward")
    plt.xlabel("Environment timestep")
    plt.ylabel("Reward")
    plt.title("recPPO (optimized/tuned) – per-step rewards")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "recppo_tuned_step_rewards.png"))
    plt.close()

# --------- Plotly interactive ---------

def plot_interactive_returns(df_ep, outdir):
    _ensure_dir(outdir)
    fig = px.line(
        df_ep,
        x="episode",
        y="ep_reward",
        title="recPPO (optimized/tuned) – episode returns (interactive)",
    )
    fig.write_html(os.path.join(outdir, "recppo_tuned_episode_returns.html"))

# --------- Main entry ---------

def main():
    # base project root (this file is in src/)
    project_root = os.path.dirname(os.path.dirname(__file__))

    episode_csv = os.path.join(
        project_root, "logs", "recppo", "recppo_episode_stats.csv"
    )
    train_csv = os.path.join(
        project_root, "logs", "recppo", "recppo_step_stats.csv"
    )
    outdir = os.path.join(project_root, "figures", "recppo_tuned")

    # ---- load CSVs ----
    df_ep_raw = pd.read_csv(episode_csv)   # columns: episode, return, length
    df_train_raw = pd.read_csv(train_csv)  # columns: step, episode, reward

    # Episode dataframe
    df_ep = df_ep_raw.rename(
        columns={
            "return": "ep_reward",
            "length": "ep_len",
        }
    )

    # Step dataframe (keep rewards as-is, just rename step)
    df_train = df_train_raw.rename(
        columns={
            "step": "timestep",
        }
    )

    # ---- make plots ----
    plot_episode_returns(df_ep, outdir)
    plot_step_rewards(df_train, outdir)
    plot_interactive_returns(df_ep, outdir)

    print(f"recPPO (tuned) plots saved to: {outdir}")

if __name__ == "__main__":
    main()
