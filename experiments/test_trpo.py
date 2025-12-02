import gymnasium as gym
import gym4real
from gym4real.envs.wds.utils import parameter_generator
import os
from sb3_contrib import TRPO
from evaluation.evaluate_random import evaluate_random_policy
from evaluation.evaluate_agent import evaluate_model
from evaluation.heuristics import evaluate_heuristic, simple_heuristic


print(">>> test_trpo.py is running...")

# Load env
package_root = os.path.dirname(gym4real.__file__)
world_file = os.path.join(package_root, "envs", "wds", "world_anytown_fixed.yaml")

params = parameter_generator(world_file)
env = gym.make("gym4real/wds-v0", settings=params)

# Measuring ep length to check for 168 or not
obs, info = env.reset()
step_count = 0
done = False

while not done:
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    done = terminated or truncated
    step_count += 1

print("\n=== Environment Episode Length Test ===")
print("Episode length =", step_count)

# Load and eval
model = TRPO.load("models/trpo_wds.zip")

obs, info = env.reset()
done = False
total_reward = 0

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    total_reward += reward

print("\n=== TRPO Evaluation ===")
print("Total episode reward =", total_reward)

# Evaluations
print("\n=== Random Policy Baseline (5 episodes) ===")
print(evaluate_random_policy(env, episodes=5))

print("\n=== TRPO Evaluation (5 episodes) ===")
print(evaluate_model(model, env, episodes=5))

print("\n=== Heuristic Evaluation (5 episodes) ===")
print(evaluate_heuristic(env, simple_heuristic, episodes=5))

env.close()

