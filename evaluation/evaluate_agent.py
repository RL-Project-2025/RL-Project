import numpy as np

def evaluate_model(model, env, episodes=5):
    rewards = []
    lengths = []

    for _ in range(episodes):
        obs = env.reset()
        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            # SB3 returns batched actions already
            action, _ = model.predict(obs, deterministic=True)

            obs, reward, done, info = env.step(action)

            total_reward += reward[0]   # VecEnv → index 0
            steps += 1

        rewards.append(total_reward)
        lengths.append(steps)

    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_length": float(np.mean(lengths)),
        "std_length": float(np.std(lengths)),
        "all_rewards": rewards,
    }
