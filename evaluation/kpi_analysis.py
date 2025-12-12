import numpy as np


def simple_heuristic(state):
    """
    Simple rule-based policy:
    - Turn pump ON if tank level < 50%
    """
    tank_level = state[0]
    return 1 if tank_level < 0.5 else 0


def evaluate_heuristic(env, heuristic_fn, episodes=5):
    """
    Evaluate a heuristic policy on a VecEnv-based environment.
    Returns episode-level KPIs.
    """

    episode_rewards = []
    episode_lengths = []

    for _ in range(episodes):
        obs = env.reset()
        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            # VecEnv
            single_obs = obs[0]                 # unbatch
            action = heuristic_fn(single_obs)   # heuristic decision
            action = np.array([action])         # re-batch

            obs, reward, done, info = env.step(action)

            total_reward += reward[0]           
            steps += 1

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)

    return {
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "mean_length": float(np.mean(episode_lengths)),
        "std_length": float(np.std(episode_lengths)),
        "all_rewards": episode_rewards,
        "all_lengths": episode_lengths,
    }
