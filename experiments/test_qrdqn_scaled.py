import os
import gymnasium as gym
import gym4real

from gym4real.envs.wds.utils import parameter_generator
from gym4real.envs.wds.reward_scaling_wrapper import RewardScalingWrapper

from sb3_contrib import QRDQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from evaluation.evaluate_random import evaluate_random_policy
from evaluation.evaluate_agent import evaluate_model
from evaluation.heuristics import evaluate_heuristic, simple_heuristic


print(">>> test_qrdqn_scaled.py is running...")


# Environment
def make_env():
    package_root = os.path.dirname(gym4real.__file__)
    world_file = os.path.join(
        package_root, "envs", "wds", "world_anytown_fixed.yaml"
    )

    params = parameter_generator(world_file)

    env = gym.make("gym4real/wds-v0", settings=params)
    env = RewardScalingWrapper(env)  

    return env


env = DummyVecEnv([make_env])

# Load VecNormalize statistics from training
env = VecNormalize.load(
    "models/qrdqn_wds_scaled_vecnorm.pkl",
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
    action = env.action_space.sample()
    obs, reward, done, info = env.step([action])
    step_count += 1

print("\n=== Environment Episode Length Test (SMDP) ===")
print("Episode length =", step_count)
print("(expected: variable, NOT 168)")


# Load model
model_path = "models/qrdqn_wds_scaled"
if not os.path.exists(model_path + ".zip"):
    raise FileNotFoundError(
        f"{model_path}.zip not found. Train first with train_qrdqn_scaled.py"
    )

model = QRDQN.load(model_path)


# Single-episode evaluation
obs = env.reset()
done = False
total_reward = 0.0

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    total_reward += reward[0]

print("\n=== QR-DQN Scaled Evaluation (single episode) ===")
print("Total episode reward =", total_reward)


# Baseline comparisons
print("\n=== Random Policy Baseline (5 episodes) ===")
print(evaluate_random_policy(env, episodes=5))

print("\n=== QR-DQN Scaled Evaluation (5 episodes) ===")
print(evaluate_model(model, env, episodes=5))

print("\n=== Heuristic Evaluation (5 episodes) ===")
print(evaluate_heuristic(env, simple_heuristic, episodes=5))


env.close()
