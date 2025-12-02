import numpy as np

def evaluate_random_policy(env, episodes=5):
    rewards = []
    lengths = []

    for _ in range(episodes):
        obs, info = env.reset()
        done = False
        total_r = 0
        length = 0

        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_r += reward
            length += 1

        rewards.append(total_r)
        lengths.append(length)

    return {
        "mean_reward": np.mean(rewards),
        "std_reward": np.std(rewards),
        "mean_length": np.mean(lengths),
        "std_length": np.std(lengths),
        "all_rewards": rewards
    }
