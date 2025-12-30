import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from collections import deque
from tqdm import trange, tqdm  # Am using this for progress bar during training

from networks.actor_critic import ActorCriticNet
from algorithms.a2c_mc import A2C
from envs.make_env import make_env

# Helper function so I can get unnormalised rewards for tensorboard
def get_unnormalised_episode_return(env, rewards):
    """
    Convert normalised rewards back to environment scale.
    """
    if hasattr(env, "unnormalize_reward"):
        rewards = [env.unnormalize_reward(r) for r in rewards]
    return sum(r.item() for r in rewards)


def train(
    num_episodes: int = 1000,
    gamma: float = 0.99,
    lr: float = 3e-4,
    hidden_dim: int = 128,
    device: str = "cpu",
):
    
    # For the rew/mean
    return_buffer = deque(maxlen=50)

    # Timestamped run name
    run_name = f"a2c_mc_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=f"runs/{run_name}")

    # Environment
    env = make_env(
        use_wrapper=False,          # HourlyDecisionWrapper OFF (theoretically invalid)
        use_normalisation=True,     # VecNormalize ON
        reward_scaling=True,        # RewardScalingWrapper ON
    )
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Network
    network = ActorCriticNet(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
    )

    # Optimizer
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)

    # A2C algorithm
    agent = A2C(
        env=env,
        network=network,
        optimizer=optimizer,
        gamma=gamma,
        device=device,
    )

    global_step = 0

    # Slightly tweaked the standard for loop logic
    # This gives a progress bar for clarity
    pbar = trange(num_episodes, desc="Training A2C")
    for episode in pbar:
        log_probs, values, rewards, entropies = agent.run_episode()
        metrics = agent.update(log_probs, values, rewards, entropies)

        episode_return_norm = sum(r.item() for r in rewards)
        episode_return_raw = get_unnormalised_episode_return(env, rewards)
        return_buffer.append(episode_return_raw)
        mean_return = np.mean(return_buffer)

        writer.add_scalar("rollout/ep_rew_mean", mean_return, episode)
        writer.add_scalar("train/policy_loss", metrics["policy_loss"], episode)
        writer.add_scalar("train/value_loss", metrics["value_loss"], episode)
        writer.add_scalar("train/entropy", metrics["entropy"], episode)


        pbar.set_postfix(
            ep_return=f"{episode_return_norm:.2f}",
            policy_loss=f"{metrics['policy_loss']:.3f}"
        )

        if episode % 50 == 0:
            tqdm.write(
                f"Episode {episode:5d} | "
                f"Return: {episode_return_norm:8.2f}"
            )

        global_step += 1

    # Save model
    torch.save(network.state_dict(), f"models/{run_name}.pt")

    # save the VecNormalise stats
    env.save(f"models/{run_name}_vecnormalize.pkl")

    writer.close()


if __name__ == "__main__":
    train()
