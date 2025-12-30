# train_rollout.py
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from collections import deque
from tqdm import trange

from networks.actor_critic import ActorCriticNet
from algorithms.a2c_rollout import A2CRollout
from envs.make_env import make_env


# Helper function to unnormalise rewards for tensorboard
def unnormalize_reward(env, reward):
    if hasattr(env, "unnormalize_reward"):
        raw = env.unnormalize_reward(reward)
        return float(raw)  # works for scalar or array
    return float(reward)




def train(
    total_updates: int = 100_000,     
    gamma: float = 0.99,
    lr: float = 3e-4,
    hidden_dim: int = 128,
    n_steps: int = 5,
    device: str = "cpu",
):
    run_name = f"a2c_rollout_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=f"runs/{run_name}")

    env = make_env(
        use_wrapper=False,
        use_normalisation=True,
        reward_scaling=True,
    )

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    network = ActorCriticNet(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)

    agent = A2CRollout(
        env=env,
        network=network,
        optimizer=optimizer,
        gamma=gamma,
        n_steps=n_steps,
        device=device,
        normalize_advantage=True,
    )

    # Track completed episodes (VecEnv length = 1 here)
    ep_return = 0.0
    ep_returns_window = deque(maxlen=50)

    pbar = trange(total_updates, desc="Training A2C (rollout)")
    for update in pbar:
        log_probs, values, rewards, entropies, dones, last_value = agent.collect_rollout()
        metrics = agent.update(log_probs, values, rewards, entropies, dones, last_value)

        # episode return tracking (from rollout rewards/dones)
        # rewards: list of tensors shape (n_envs,), n_envs=1
        # dones:   list of tensors shape (n_envs,)
        for r_t, d_t in zip(rewards, dones):
            r_np = r_t.detach().cpu().numpy()
            raw_r = unnormalize_reward(env, r_np)

            ep_return += raw_r

            if float(d_t.item()) == 1.0:
                ep_returns_window.append(ep_return)
                ep_return = 0.0

        ep_rew_mean = float(np.mean(ep_returns_window)) if len(ep_returns_window) > 0 else np.nan

        # TensorBoard
        writer.add_scalar("rollout/ep_rew_mean", ep_rew_mean, update)
        writer.add_scalar("train/policy_loss", metrics["policy_loss"], update)
        writer.add_scalar("train/value_loss", metrics["value_loss"], update)
        writer.add_scalar("train/entropy_loss", -metrics["entropy"], update)  # SB3 logs entropy_loss negative

        pbar.set_postfix(ep_rew_mean=f"{ep_rew_mean:.2f}" if ep_rew_mean == ep_rew_mean else "nan",
                         policy_loss=f"{metrics['policy_loss']:.3f}")

    torch.save(network.state_dict(), f"models/{run_name}.pt")
    env.save(f"models/{run_name}_vecnormalize.pkl")
    writer.close()


if __name__ == "__main__":
    train()
