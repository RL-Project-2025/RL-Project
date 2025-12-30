# eval_rollout.py
import torch
import numpy as np

from envs.make_env import make_env
from networks.actor_critic import ActorCriticNet
from stable_baselines3.common.vec_env import VecNormalize

from evaluation.evaluate_random import evaluate_random_policy
from evaluation.kpi_analysis import analyse_kpis
from evaluation.heuristics import evaluate_heuristic, simple_heuristic


def evaluate_policy_greedy(
    model_path: str,
    vecnorm_path: str,
    num_episodes: int = 50,
    device: str = "cpu",
):
    env = make_env(use_wrapper=False, use_normalisation=True, reward_scaling=True)
    env = VecNormalize.load(vecnorm_path, env)
    env.training = False
    env.norm_reward = False

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    network = ActorCriticNet(obs_dim=obs_dim, action_dim=action_dim).to(device)
    network.load_state_dict(torch.load(model_path, map_location=device))
    network.eval()

    returns = []
    for _ in range(num_episodes):
        obs = env.reset()
        done = False
        ep_return = 0.0

        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
            with torch.no_grad():
                dist, _ = network(obs_tensor)
                action = torch.argmax(dist.probs, dim=-1)

            obs, reward, done, _ = env.step([int(action.item())])
            ep_return += reward[0]
            done = done[0]

        returns.append(ep_return)

    return np.array(returns)


def evaluate_random_policy_wrapped(vecnorm_path: str, num_episodes: int = 50):
    env = make_env(use_wrapper=False, use_normalisation=True, reward_scaling=True)
    env = VecNormalize.load(vecnorm_path, env)
    env.training = False
    env.norm_reward = False
    return evaluate_random_policy(env, episodes=num_episodes)


def evaluate_heuristic_policy(vecnorm_path: str, heuristic_fn, num_episodes: int = 50):
    env = make_env(use_wrapper=False, use_normalisation=True, reward_scaling=True)
    env = VecNormalize.load(vecnorm_path, env)
    env.training = False
    env.norm_reward = False
    return evaluate_heuristic(env, heuristic_fn=heuristic_fn, episodes=num_episodes)


if __name__ == "__main__":
    MODEL_PATH = "models/a2c_rollout_20251230_000030.pt"
    VECNORM_PATH = "models/a2c_rollout_20251230_000030_vecnormalize.pkl"
    N = 50

    rollout_returns = evaluate_policy_greedy(MODEL_PATH, VECNORM_PATH, num_episodes=N)
    random_results = evaluate_random_policy_wrapped(VECNORM_PATH, num_episodes=N)
    heuristic_results = evaluate_heuristic_policy(VECNORM_PATH, simple_heuristic, num_episodes=N)

    print("\nA2C (rollout):")
    print(f"Mean: {rollout_returns.mean():.2f}, Std: {rollout_returns.std():.2f}")

    print("\nRandom:")
    print(f"Mean: {random_results['mean_reward']:.2f}, Std: {random_results['std_reward']:.2f}")

    print("\nHeuristic:")
    print(f"Mean: {heuristic_results['mean_reward']:.2f}, Std: {heuristic_results['std_reward']:.2f}")

    analyse_kpis(
        rollout_returns,
        random_results["all_rewards"],
        heuristic_results["all_rewards"],
    )
