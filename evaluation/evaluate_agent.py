import numpy as np

def evaluate_model(model, env, episodes=20):
    rewards = []
    lengths = []

    for _ in range(episodes):
        obs, info = env.reset()
        done = False
        total_r = 0
        step_count = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_r += reward
            step_count += 1

        rewards.append(total_r)
        lengths.append(step_count)

    return {
        "mean_reward": np.mean(rewards),
        "std_reward": np.std(rewards),
        "mean_length": np.mean(lengths),
        "std_length": np.std(lengths),
        "all_rewards": rewards
    }
