
import numpy as np

def simple_heuristic(state):
    # Turn pump ON if tank level < 50%
    tank_level = state[0]
    if tank_level < 0.5:
        return 1
    return 0


def evaluate_heuristic(env, heuristic_fn, episodes=5):
    rewards = []

    for _ in range(episodes):
        obs = env.reset()
        done = False
        total_r = 0.0

        while not done:
            # Unbatch observation
            single_obs = obs[0]

            # Heuristic acts on single state
            action = heuristic_fn(single_obs)

            # Re-batch action for VecEnv
            action = np.array([action])

            # Step env
            obs, reward, done, info = env.step(action)

            # Extract scalar reward
            total_r += reward[0]

        rewards.append(total_r)

    return {
        "mean_reward": float(sum(rewards) / len(rewards)),
        "all_rewards": rewards
    }

