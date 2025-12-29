#!/usr/bin/env python3
"""
SB3 RecurrentPPO analysis script.

Extracts episode returns/lengths from TensorBoard event files 
(since SB3 doesn't write CSVs like custom implementations).
Generates identical plots for fair comparison.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

sns.set(style="whitegrid", context="talk")

def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def extract_sb3_data(tensorboard_dir):
    """
    Extract episode returns/lengths from SB3 TensorBoard logs.
    
    SB3 logs:
    - rollout/ep_rew_mean (smoothed)
    - Monitor writes raw episodes to event files
    """
    events_path = None
    for root, dirs, files in os.walk(tensorboard_dir):
        for file in files:
            if file.startswith("events.out.tfevents"):
                events_path = os.path.join(root, file)
                break
        if events_path:
            break
    
    if not events_path:
        raise FileNotFoundError(f"No TensorBoard events found in {tensorboard_dir}")
    
    event_acc = EventAccumulator(events_path)
    event_acc.Reload()
    
    # Extract raw episode data from Monitor (SB3 standard)
    ep_data = []
    
    # rollout/ep_rew_mean, rollout/ep_len_mean (smoothed)
    if 'rollout/ep_rew_mean' in event_acc.Tags()['scalars']:
        rew_mean = event_acc.Scalars('rollout/ep_rew_mean')
        ep_data.extend([{'step': s.step, 'ep_reward': s.value, 'type': 'rew_mean'} 
                       for s in rew_mean])
    
    if 'rollout/ep_len_mean' in event_acc.Tags()['scalars']:
        len_mean = event_acc.Scalars('rollout/ep_len_mean')
        ep_data.extend([{'step': s.step, 'ep_len': s.value, 'type': 'len_mean'} 
                       for s in len_mean])
    
    df = pd.DataFrame(ep_data)
    return df

# --------- Episode-level plots (using TensorBoard data) ---------

def plot_episode_returns(df, outdir):
    _ensure_dir(outdir)
    
    # Plot smoothed returns vs timestep (SB3 format)
    plt.figure(figsize=(10, 6))
    if 'ep_reward' in df.columns:
        sns.lineplot(data=df[df['type']=='rew_mean'], x="step", y="ep_reward")
        plt.xlabel("Timestep")
        plt.ylabel("Episode return (mean)")
    plt.title("SB3 RecurrentPPO – rollout/ep_rew_mean")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "sb3_recppo_ep_rew_mean.png"))
    plt.close()

def plot_step_rewards(df, outdir):
    """
    SB3 doesn't log per-step rewards to CSV, so plot training diagnostics instead.
    """
    _ensure_dir(outdir)
    
    # Plot key training scalars if available
    event_path = None
    for root, dirs, files in os.walk(outdir.replace("figures/sb3_recppo", "logs/sb3_recppo")):
        for file in files:
            if file.startswith("events.out.tfevents"):
                event_path = os.path.join(root, file)
                break
        if event_path:
            break
    
    if event_path:
        event_acc = EventAccumulator(event_path)
        event_acc.Reload()
        
        scalars = {}
        for tag in ['train/policy_loss', 'train/value_loss', 'train/approx_kl', 'train/entropy_loss']:
            if tag in event_acc.Tags()['scalars']:
                scalars[tag] = event_acc.Scalars(tag)
        
        if scalars:
            df_train = []
            for tag, data in scalars.items():
                for s in data:
                    df_train.append({'step': s.step, 'value': s.value, 'metric': tag})
            
            df_train_df = pd.DataFrame(df_train)
            
            plt.figure(figsize=(12, 8))
            sns.lineplot(data=df_train_df, x="step", y="value", hue="metric")
            plt.xlabel("Timestep")
            plt.ylabel("Loss / Metric")
            plt.title("SB3 RecurrentPPO – Training Diagnostics")
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, "sb3_recppo_training_diagnostics.png"), bbox_inches='tight')
            plt.close()

# --------- Main entry ---------

def main():
    project_root = os.path.dirname(os.path.dirname(__file__))
    
    # Find SB3 RecPPO TensorBoard directory
    tensorboard_dir = os.path.join(project_root, "logs", "sb3_recppo")
    outdir = os.path.join(project_root, "figures", "sb3_recppo")
    
    print(f"Extracting SB3 RecPPO data from: {tensorboard_dir}")
    
    # Extract data from TensorBoard events
    df = extract_sb3_data(tensorboard_dir)
    
    # Generate plots
    plot_episode_returns(df, outdir)
    plot_step_rewards(df, outdir)
    
    print(f"SB3 RecPPO plots saved to: {outdir}")
    print("Note: Use TensorBoard for full metrics:")
    print(f"  tensorboard --logdir={os.path.join(project_root, 'logs/sb3_recppo')}")

if __name__ == "__main__":
    main()
