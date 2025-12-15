#!/usr/bin/env python3

"""
Recurrent PPO (recPPO) for WDSEnv, mirroring
base PPO structure.
"""

import os
import time
import csv

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
import yaml

# --------- project-level paths ---------

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))  # .../rlc2
LOG_ROOT = os.path.join(PROJECT_ROOT, "logs")
MODEL_ROOT = os.path.join(PROJECT_ROOT, "models")
os.makedirs(LOG_ROOT, exist_ok=True)
os.makedirs(MODEL_ROOT, exist_ok=True)

LOG_DIR = os.path.join(LOG_ROOT, "recppo")
os.makedirs(LOG_DIR, exist_ok=True)

EPISODE_CSV = os.path.join(LOG_DIR, "recppo_episode_stats.csv")
STEP_CSV    = os.path.join(LOG_DIR, "recppo_step_stats.csv")


# ========================= Recurrent Actor-Critic =========================

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=64, rnn_hidden=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
        )
        self.rnn = nn.LSTM(
            input_size=hidden,
            hidden_size=rnn_hidden,
            num_layers=1,
        )
        self.actor = nn.Linear(rnn_hidden, act_dim)
        self.critic = nn.Linear(rnn_hidden, 1)
        self.rnn_hidden = rnn_hidden

    def init_hidden(self, batch_size=1):
        h = torch.zeros(1, batch_size, self.rnn_hidden)
        c = torch.zeros(1, batch_size, self.rnn_hidden)
        return h, c

    def forward(self, x, hidden):
        T, B, _ = x.shape
        enc = self.encoder(x.view(T * B, -1))
        enc = enc.view(T, B, -1)
        rnn_out, hidden_out = self.rnn(enc, hidden)
        logits = self.actor(rnn_out)
        values = self.critic(rnn_out).squeeze(-1)
        return logits, values, hidden_out

    def act(self, obs, hidden):
        x = obs.unsqueeze(0).unsqueeze(0)
        logits, values, hidden_next = self.forward(x, hidden)
        logits = logits.squeeze(0).squeeze(0)
        values = values.squeeze(0).squeeze(0)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, values, hidden_next

    def evaluate(self, obs, actions, hidden):
        logits, values, _ = self.forward(obs, hidden)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy, values


# ========================= Rollout Buffer =========================

class RolloutBuffer:
    def __init__(self):
        self.clear()

    def store(self, obs, action, reward, log_prob, value, done):
        self.obs.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)

    def clear(self):
        self.obs, self.actions, self.rewards = [], [], []
        self.log_probs, self.values, self.dones = [], [], []

    def get(self):
        return (
            torch.tensor(np.array(self.obs), dtype=torch.float32),
            torch.tensor(self.actions, dtype=torch.long),
            torch.tensor(self.rewards, dtype=torch.float32),
            torch.tensor(self.log_probs, dtype=torch.float32),
            torch.tensor(self.values, dtype=torch.float32),
            torch.tensor(self.dones, dtype=torch.float32),
        )


# ========================= GAE =========================

def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    advantages = []
    gae = 0.0
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0.0
        else:
            next_value = values[t + 1] * (1 - dones[t])
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages.append(gae)
    advantages = advantages[::-1]
    advantages = torch.tensor(advantages, dtype=torch.float32)
    returns = advantages + values
    return advantages, returns


# ========================= PPO (Recurrent) =========================

class PPO:
    def __init__(self, obs_dim, act_dim, lr=3e-4, gamma=0.99, lam=0.95,
                 clip_eps=0.2, epochs=10, batch_size=64, ent_coef=0.01,
                 vf_coef=0.5, max_grad_norm=0.5):
        self.gamma, self.lam = gamma, lam
        self.clip_eps, self.epochs = clip_eps, epochs
        self.batch_size = batch_size
        self.ent_coef, self.vf_coef = ent_coef, vf_coef
        self.max_grad_norm = max_grad_norm

        self.ac = ActorCritic(obs_dim, act_dim)
        self.optimizer = optim.Adam(self.ac.parameters(), lr=lr)
        self.buffer = RolloutBuffer()

        self.h, self.c = self.ac.init_hidden(batch_size=1)

    def select_action(self, obs):
        obs_t = torch.tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            action, log_prob, value, (self.h, self.c) = self.ac.act(
                obs_t, (self.h, self.c)
            )
        return action.item(), log_prob.item(), value.item()

    def store(self, obs, action, reward, log_prob, value, done):
        self.buffer.store(obs, action, reward, log_prob, value, done)
        if done:
            self.h, self.c = self.ac.init_hidden(batch_size=1)

    def update(self):
        obs, actions, rewards, old_log_probs, values, dones = self.buffer.get()

        advantages, returns = compute_gae(
            rewards, values, dones, self.gamma, self.lam
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        n = len(obs)
        obs_seq = obs.view(n, 1, -1)
        actions_seq = actions.view(n, 1)
        old_log_probs_seq = old_log_probs.view(n, 1)
        adv_seq = advantages.view(n, 1)
        ret_seq = returns.view(n, 1)

        total_policy_loss, total_value_loss, total_entropy = 0.0, 0.0, 0.0
        total_approx_kl, total_clip_frac = 0.0, 0.0
        num_updates = 0

        for _ in range(self.epochs):
            h0, c0 = self.ac.init_hidden(batch_size=1)
            new_log_probs, entropy, new_values = self.ac.evaluate(
                obs_seq, actions_seq, (h0, c0)
            )

            log_ratio = new_log_probs - old_log_probs_seq
            ratio = log_ratio.exp()

            with torch.no_grad():
                approx_kl = ((ratio - 1) - log_ratio).mean()
                clip_frac = ((ratio - 1).abs() > self.clip_eps).float().mean()

            surr1 = ratio * adv_seq
            surr2 = torch.clamp(
                ratio, 1 - self.clip_eps, 1 + self.clip_eps
            ) * adv_seq
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = ((new_values - ret_seq) ** 2).mean()
            entropy_loss = -entropy.mean()

            loss = (
                policy_loss
                + self.vf_coef * value_loss
                + self.ent_coef * entropy_loss
            )

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.ac.parameters(), self.max_grad_norm)
            self.optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += (-entropy_loss).item()
            total_approx_kl += approx_kl.item()
            total_clip_frac += clip_frac.item()
            num_updates += 1

        self.buffer.clear()

        return {
            "policy_loss": total_policy_loss / num_updates,
            "value_loss": total_value_loss / num_updates,
            "entropy": total_entropy / num_updates,
            "approx_kl": total_approx_kl / num_updates,
            "clip_fraction": total_clip_frac / num_updates,
        }

    def save(self, path):
        torch.save(
            {
                "model_state_dict": self.ac.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )

    def load(self, path):
        checkpoint = torch.load(path)
        self.ac.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.h, self.c = self.ac.init_hidden(batch_size=1)


# ========================= Training Loop =========================

def train_ppo(env, total_timesteps=200000, rollout_steps=2048, log_dir=None):
    if log_dir is None:
        log_dir = LOG_ROOT

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    run_name = f"recppo_scratch_{int(time.time())}"
    writer = SummaryWriter(os.path.join(log_dir, run_name))

    agent = PPO(obs_dim, act_dim)

    writer.add_text(
        "hyperparameters",
        f"lr={agent.optimizer.param_groups[0]['lr']}, gamma={agent.gamma}, "
        f"lam={agent.lam}, clip_eps={agent.clip_eps}, epochs={agent.epochs}, "
        f"batch_size={agent.batch_size}, ent_coef={agent.ent_coef}, "
        f"vf_coef={agent.vf_coef}",
    )

    ep_file = open(EPISODE_CSV, "w", newline="")
    ep_writer = csv.writer(ep_file)
    ep_writer.writerow(["episode", "return", "length"])

    step_file = open(STEP_CSV, "w", newline="")
    step_writer = csv.writer(step_file)
    step_writer.writerow(["step", "episode", "reward"])

    obs, _ = env.reset()
    ep_reward, ep_len = 0.0, 0
    ep_rewards, ep_lengths = [], []
    timestep = 0
    start_time = time.time()
    episode_idx = 0

    while timestep < total_timesteps:
        for _ in range(rollout_steps):
            action, log_prob, value = agent.select_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.store(obs, action, reward, log_prob, value, done)

            obs = next_obs
            ep_reward += reward
            ep_len += 1
            timestep += 1

            step_writer.writerow([timestep, episode_idx, float(reward)])

            if done:
                ep_rewards.append(ep_reward)
                ep_lengths.append(ep_len)

                writer.add_scalar("rollout/ep_rew", ep_reward, timestep)
                writer.add_scalar("rollout/ep_len", ep_len, timestep)

                ep_writer.writerow(
                    [episode_idx, float(ep_reward), int(ep_len)]
                )
                episode_idx += 1

                obs, _ = env.reset()
                ep_reward, ep_len = 0.0, 0

            if timestep >= total_timesteps:
                break

        if len(agent.buffer.obs) == rollout_steps:
            stats = agent.update()

            writer.add_scalar("train/policy_loss", stats["policy_loss"], timestep)
            writer.add_scalar("train/value_loss", stats["value_loss"], timestep)
            writer.add_scalar("train/entropy", stats["entropy"], timestep)
            writer.add_scalar("train/approx_kl", stats["approx_kl"], timestep)
            writer.add_scalar(
                "train/clip_fraction", stats["clip_fraction"], timestep
            )

            elapsed = time.time() - start_time
            fps = timestep / elapsed
            writer.add_scalar("time/fps", fps, timestep)

    ep_file.close()
    step_file.close()
    writer.close()
    return agent


# ========================= WDSEnv Entry Point =========================

# ========================= WDSEnv Entry Point =========================

if __name__ == "__main__":
    import gymnasium as gym
    import gym4real
    from gym4real.envs.wds.utils import parameter_generator
    from gym4real.envs.wds.reward_scaling_wrapper import RewardScalingWrapper

    # make cwd = gym4ReaL so all 'gym4real/...' paths work as in rlc1
    PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))  # .../rlc2
    GYM_ROOT = os.path.join(PROJECT_ROOT, "gym4ReaL")
    os.chdir(GYM_ROOT)

    params = parameter_generator(
        hydraulic_step=3600,
        duration=604800,
        seed=42,
        world_options="gym4real/envs/wds/world_anytown.yaml",
    )

    base_env = gym.make("gym4real/wds-v0", settings=params)
    env = RewardScalingWrapper(base_env)

    # logs and models go to rlc2/logs and rlc2/models
    log_dir = os.path.join("..", "logs")
    model_path = os.path.join("..", "models", "recppo_scratch.pt")

    agent = train_ppo(env, total_timesteps=50000, log_dir=log_dir)
    agent.save(model_path)
    print(f"\nTraining complete. Model saved to {model_path}")
    print("View logs: tensorboard --logdir=../logs")
