# import numpy as np

# def evaluate_random_policy(env, episodes=5):
#     rewards = []
#     lengths = []

#     for _ in range(episodes):
#         obs = env.reset()
#         done = False
#         total_reward = 0.0
#         steps = 0

#         while not done:
#             action = np.array([env.action_space.sample()])  # <-- FIX
#             obs, reward, done, info = env.step(action)

#             total_reward += reward[0]
#             steps += 1

#         rewards.append(total_reward)
#         lengths.append(steps)

#     return {
#         "mean_reward": float(np.mean(rewards)),
#         "std_reward": float(np.std(rewards)),
#         "mean_length": float(np.mean(lengths)),
#         "std_length": float(np.std(lengths)),
#         "all_rewards": rewards,
#     }

import numpy as np


def evaluate_random_policy(env, episodes=5):
    rewards = []
    lengths = []

    for _ in range(episodes):
        obs = env.reset()
        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            # Sample random action
            action = np.array([env.action_space.sample()])

            obs, reward, done, _ = env.step(action)

            total_reward += reward[0]
            steps += 1
            done = done[0]

        rewards.append(total_reward)
        lengths.append(steps)

    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_length": float(np.mean(lengths)),
        "std_length": float(np.std(lengths)),
        "all_rewards": rewards,
    }
