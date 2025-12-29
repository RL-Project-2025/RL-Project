#!/usr/bin/env python3
"""
SB3 RecurrentPPO (MlpLstmPolicy) baseline for WDSEnv 

Purpose: Strong production baseline to validate custom implementations.
Identical environment preprocessing as custom PPO/recPPO/TRPO baselines:
- RewardScalingWrapper + NormaliseObservation (mean=0, std=1 per obs dim)
- PPO paper Table 3 hyperparameters exactly [1]
- Monitor wrapper for ep_rew_mean/ep_len_mean logging (TensorBoard overlay)

SB3-Contrib RecurrentPPO [4]: production-tested LSTM PPO implementation.
Compares MlpLstmPolicy vs custom LSTM implementations.

[1] Schulman et al. (2017). Proximal Policy Optimization Algorithms. arXiv:1707.06347
[2] Engstrom et al. (2022). 37 Implementation Details of PPO. ICLR Blog Track.
[3] Raffin et al. (2021). Stable-Baselines3: Reliable RL implementations. JMLR.
[4] sb3-contrib RecurrentPPO: https://sb3-contrib.readthedocs.io/en/master/
[5] Daveonwave (2024). Gym4ReaL: WDSEnv benchmark suite.
[6] Salaorni (2021). Optimal real-time WDS control. Politecnico di Milano thesis.
"""

import os
import time
import numpy as np
import gymnasium as gym
import gym4real
from gym4real.envs.wds.utils import parameter_generator
from gym4real.envs.wds.reward_scaling_wrapper import RewardScalingWrapper
from Normalise import NormaliseObservation  # Matches custom baselines exactly
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib import RecurrentPPO  # Production recurrent PPO [4]

# =====================================================================
# PROJECT PATHS (matches custom PPO/TRPO structure exactly)
# =====================================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))  # .../rlc2
GYM_ROOT = os.path.join(PROJECT_ROOT, "gym4ReaL")
LOG_ROOT = os.path.join(PROJECT_ROOT, "logs", "sb3_recppo")
MODEL_ROOT = os.path.join(PROJECT_ROOT, "models")
os.makedirs(LOG_ROOT, exist_ok=True)
os.makedirs(MODEL_ROOT, exist_ok=True)
os.chdir(GYM_ROOT)

# =====================================================================
# WDSEnv FACTORY (IDENTICAL PREPROCESSING TO CUSTOM BASELINES) [5,6]
# =====================================================================
def make_wds_env(seed: int = 42):
    """
    Single WDSEnv with EXACT SAME preprocessing as custom PPO/recPPO/TRPO:
    - RewardScalingWrapper: reward normalization [5]
    - NormaliseObservation: obs normalization (mean=0, std=1) [custom]
    - Monitor: ep_rew_mean/ep_len_mean for TensorBoard overlay [2]
    """
    params = parameter_generator(
        hydraulic_step=3600,           # 1hr steps [6]
        duration=604800,               # 1-week episodes [6]
        seed=seed,
        world_options="gym4real/envs/wds/world_anytown.yaml",  # AnyTown network [6]
    )
    
    # Exact demand settings matching custom implementations
    params["demand_moving_average"] = False
    params["demand_exp_moving_average"] = True
    
    env = gym.make("gym4real/wds-v0", settings=params)
    env = RewardScalingWrapper(env)
    env = NormaliseObservation(env)
    env = Monitor(env)  # SB3 episode stats (matches custom CSV logging)
    return env

def make_vec_env(seed: int = 42):
    """SB3 VecEnv wrapper (single env, matches custom rollout size)."""
    return DummyVecEnv([lambda: make_wds_env(seed)])

# =====================================================================
# HYPERPARAMETERS (PPO PAPER TABLE 3 EXACTLY) [1]
# =====================================================================
TOTAL_TIMESTEPS = 200_000  # Matches custom baselines
ROLLOUT_STEPS = 2048       # n_steps from Table 3 [1]
SEED = 42

# PPO Table 3 hyperparameters + SB3 defaults (no tuning needed) [1,3]
HYPERPARAMS = {
    "n_steps": ROLLOUT_STEPS,      # 2048 [1]
    "batch_size": 64,              # Table 3 [1]
    "n_epochs": 10,                # Table 3 [1]
    "gamma": 0.99,                 # Table 3 [1]
    "gae_lambda": 0.95,            # GAE(λ) [2]
    "learning_rate": 3e-4,         # Table 3 [1]
    "clip_range": 0.2,             # Table 3 [1]
    "ent_coef": 0.01,              # Table 3 [1]
    "vf_coef": 0.5,                # Value loss weight [1]
    "max_grad_norm": 0.5,          # Gradient clipping [2, Detail #15]
}

# =====================================================================
# PROGRESS CALLBACK (matches custom printouts)
# =====================================================================
from stable_baselines3.common.callbacks import BaseCallback

class SimplePrintCallback(BaseCallback):
    """
    Minimal callback: prints ep_rew_mean/ep_len_mean/fps every 2048 steps.
    Mimics custom PPO/TRPO console output for easy comparison.
    Uses Monitor episode infos (r/l fields).
    """
    def __init__(self, print_every: int = 2048, verbose: int = 0):
        super().__init__(verbose)
        self.print_every = print_every
        self.start_time = None

    def _on_training_start(self) -> None:
        self.start_time = time.time()

    def _on_step(self) -> bool:
        if self.num_timesteps % self.print_every == 0 and self.num_timesteps > 0:
            # Extract episode stats from Monitor infos
            ep_rewards = []
            ep_lengths = []
            infos = self.locals.get("infos", [])
            for info in infos:
                if "episode" in info:
                    ep_rewards.append(info["episode"]["r"])
                    ep_lengths.append(info["episode"]["l"])

            if ep_rewards:
                ep_rew_mean = np.mean(ep_rewards)
                ep_len_mean = np.mean(ep_lengths)
            else:
                ep_rew_mean = float("nan")
                ep_len_mean = float("nan")

            elapsed = time.time() - self.start_time
            fps = self.num_timesteps / max(elapsed, 1e-8)

            print(
                f"[{self.num_timesteps:>7}/{TOTAL_TIMESTEPS}] "
                f"ep_rew={ep_rew_mean:>6.1f} | "
                f"ep_len={ep_len_mean:>5.1f} | "
                f"fps={fps:>4.1f}"
            )
        return True

# =====================================================================
# MAIN: SB3 RECURRENTPPO BASELINE TRAINING
# =====================================================================
if __name__ == "__main__":
    print("SB3 RecurrentPPO BASELINE (WDSEnv) - off-the-shelf comparison")
    print(f"Total timesteps: {TOTAL_TIMESTEPS:,}")
    print(f"PPO Table 3 hyperparameters: {HYPERPARAMS}")
    print(f"Policy: MlpLstmPolicy (LSTM recurrent PPO) [4]")
    print("-" * 80)

    # Identical VecEnv to custom baselines
    vec_env = make_vec_env(SEED)

    # SB3 TensorBoard logging (overlay with custom runs)
    tensorboard_log_dir = LOG_ROOT
    os.makedirs(tensorboard_log_dir, exist_ok=True)

    # RecurrentPPO: production LSTM implementation [4]
    model = RecurrentPPO(
        policy="MlpLstmPolicy",  # LSTM encoder → actor/critic [4]
        env=vec_env,
        **HYPERPARAMS,           # PPO Table 3 exactly [1]
        tensorboard_log=tensorboard_log_dir,
        verbose=1,
        seed=SEED,
        device="cuda" if torch.cuda.is_available() else "cpu",  # Match custom
    )

    callback = SimplePrintCallback(print_every=ROLLOUT_STEPS)

    tb_run_name = "sb3_recppo_wds"  # Matches custom naming
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        tb_log_name=tb_run_name,
        callback=callback,
    )

    # SB3 .zip format (for reproducibility)
    model_path = os.path.join(MODEL_ROOT, "sb3_recppo_wds.zip")
    model.save(model_path)

    print("\nSB3 RecurrentPPO complete!")
    print(f"Model saved: {model_path}")
    print(f"TensorBoard (overlay with custom): tensorboard --logdir={LOG_ROOT}")
