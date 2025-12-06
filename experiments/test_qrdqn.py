import gymnasium as gym
import gym4real
from gym4real.envs.wds.utils import parameter_generator
import os
from sb3_contrib import QRDQN
from evaluation.evaluate_random import evaluate_random_policy
from evaluation.evaluate_agent import evaluate_model
from evaluation.heuristics import evaluate_heuristic, simple_heuristic
from gym4real.envs.wds.hourly_wrapper import HourlyDecisionWrapper


print(">>> test_qrdqn.py is running...")


# Load env
package_root = os.path.dirname(gym4real.__file__)
world_file = os.path.join(package_root, "envs", "wds", "world_anytown_fixed.yaml")

params = parameter_generator(world_file)

base_env = gym.make("gym4real/wds-v0", settings=params)
env = HourlyDecisionWrapper(base_env)


# Load QR-DQN model
model_path = "models/qrdqn_wds.zip"
if not os.path.exists(model_path):
    raise FileNotFoundError(
        f"{model_path} not found. Train first with: python experiments/train_qrdqn.py"
    )

model = QRDQN.load(model_path)

# Single-episode run
obs, info = env.reset()
done = False
total_reward = 0

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    total_reward += reward

print("\n=== QR-DQN Single-Episode Run ===")
print("Total episode reward =", total_reward)

# Full eval
print("\n=== Random Policy Baseline (5 episodes) ===")
print(evaluate_random_policy(env, episodes=5))

print("\n=== QR-DQN Evaluation (5 episodes) ===")
print(evaluate_model(model, env, episodes=5))

print("\n=== Heuristic Evaluation (5 episodes) ===")
print(evaluate_heuristic(env, simple_heuristic, episodes=5))

env.close()
