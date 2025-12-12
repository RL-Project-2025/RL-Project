import gymnasium as gym
import gym4real
from gym4real.envs.wds.utils import parameter_generator
import os

from sb3_contrib import TRPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from evaluation.evaluate_random import evaluate_random_policy
from evaluation.evaluate_agent import evaluate_model
from evaluation.heuristics import evaluate_heuristic, simple_heuristic

from gym4real.envs.wds.reward_scaling_wrapper import RewardScalingWrapper


print(">>> test_trpo_scaled.py is running...")

# Load env
package_root = os.path.dirname(gym4real.__file__)
world_file = os.path.join(
    package_root,
    "envs", "wds", "world_anytown_fixed.yaml"
)

params = parameter_generator(world_file)


# SMDP environment
def make_env():
    base_env = gym.make("gym4real/wds-v0", settings=params)
    return RewardScalingWrapper(base_env)

env = DummyVecEnv([make_env])


# Load VecNormalize stats from scaled training
env = VecNormalize.load(
    "models/trpo_wds_scaled_vecnorm.pkl",
    env
)

# Evaluation mode
env.training = False
env.norm_reward = False


# Episode length sanity check
obs = env.reset()
step_count = 0
done = False

while not done:
    obs, reward, done, info = env.step([env.action_space.sample()])
    step_count += 1

print("\n=== Environment Episode Length Test (SMDP) ===")
print("Episode steps =", step_count)


# Load trained model
model = TRPO.load("models/trpo_wds_scaled")


# Single-episode evaluation
obs = env.reset()
done = False
total_reward = 0.0

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    total_reward += reward[0]

print("\n=== TRPO Scaled Evaluation (single episode) ===")
print("Total episode reward =", total_reward)


# Baselines
print("\n=== Random Policy Baseline (5 episodes) ===")
print(evaluate_random_policy(env, episodes=5))

print("\n=== TRPO Scaled Evaluation (5 episodes) ===")
print(evaluate_model(model, env, episodes=5))

print("\n=== Heuristic Evaluation (5 episodes) ===")
print(evaluate_heuristic(env, simple_heuristic, episodes=5))

env.close()