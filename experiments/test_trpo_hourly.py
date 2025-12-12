# import gymnasium as gym
# import gym4real
# from gym4real.envs.wds.utils import parameter_generator
# import os
# from sb3_contrib import TRPO
# from evaluation.evaluate_random import evaluate_random_policy
# from evaluation.evaluate_agent import evaluate_model
# from evaluation.heuristics import evaluate_heuristic, simple_heuristic
# from gym4real.envs.wds.hourly_wrapper import HourlyDecisionWrapper


# print(">>> test_trpo.py is running...")

# # Load env
# package_root = os.path.dirname(gym4real.__file__)
# world_file = os.path.join(package_root, "envs", "wds", "world_anytown_fixed.yaml")

# params = parameter_generator(world_file)

# base_env = gym.make("gym4real/wds-v0", settings=params)
# env = HourlyDecisionWrapper(base_env)


# # Measuring ep length to check for 168 or not
# obs, info = env.reset()
# step_count = 0
# done = False

# while not done:
#     obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
#     done = terminated or truncated
#     step_count += 1

# print("\n=== Environment Episode Length Test ===")
# print("Episode length =", step_count)

# # Load and eval
# model = TRPO.load("models/trpo_wds.zip")

# obs, info = env.reset()
# done = False
# total_reward = 0

# while not done:
#     action, _ = model.predict(obs, deterministic=True)
#     obs, reward, terminated, truncated, info = env.step(action)
#     done = terminated or truncated
#     total_reward += reward

# print("\n=== TRPO Evaluation ===")
# print("Total episode reward =", total_reward)

# # Evaluations
# print("\n=== Random Policy Baseline (5 episodes) ===")
# print(evaluate_random_policy(env, episodes=5))

# print("\n=== TRPO Evaluation (5 episodes) ===")
# print(evaluate_model(model, env, episodes=5))

# print("\n=== Heuristic Evaluation (5 episodes) ===")
# print(evaluate_heuristic(env, simple_heuristic, episodes=5))

# env.close()

import gymnasium as gym
import gym4real
from gym4real.envs.wds.utils import parameter_generator
import os

from sb3_contrib import TRPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from evaluation.evaluate_random import evaluate_random_policy
from evaluation.evaluate_agent import evaluate_model
from evaluation.heuristics import evaluate_heuristic, simple_heuristic

from gym4real.envs.wds.hourly_wrapper import HourlyDecisionWrapper


print(">>> test_trpo_hourly.py is running...")

# Load env
package_root = os.path.dirname(gym4real.__file__)
world_file = os.path.join(
    package_root,
    "envs", "wds", "world_anytown_fixed.yaml"
)

params = parameter_generator(world_file)


# Create hourly-wrapped environment
def make_env():
    base_env = gym.make("gym4real/wds-v0", settings=params)
    return HourlyDecisionWrapper(base_env)

env = DummyVecEnv([make_env])


# Load VecNormalize stats from training
env = VecNormalize.load(
    "models/trpo_wds_hourly_vecnorm.pkl",
    env
)


# Important: evaluation mode
env.training = False
env.norm_reward = False


# Episode length sanity check-
obs = env.reset()
step_count = 0
done = False

while not done:
    obs, reward, done, info = env.step([env.action_space.sample()])
    step_count += 1

print("\n=== Environment Episode Length Test ===")
print("Episode length =", step_count)  # should be 168


# Load trained model
model = TRPO.load("models/trpo_wds_hourly")


# Single-episode evaluation
obs = env.reset()
done = False
total_reward = 0.0

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    total_reward += reward[0]

print("\n=== TRPO Evaluation (single episode) ===")
print("Total episode reward =", total_reward)


# Baseline comparisons
print("\n=== Random Policy Baseline (5 episodes) ===")
print(evaluate_random_policy(env, episodes=5))

print("\n=== TRPO Evaluation (5 episodes) ===")
print(evaluate_model(model, env, episodes=5))

print("\n=== Heuristic Evaluation (5 episodes) ===")
print(evaluate_heuristic(env, simple_heuristic, episodes=5))

env.close()
