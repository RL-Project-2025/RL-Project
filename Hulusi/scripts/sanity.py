import os
if os.path.exists('gym4ReaL'):
    os.chdir('gym4ReaL')
import gymnasium as gym
import gym4real
from gym4real.envs.wds.utils import parameter_generator

params = parameter_generator(
    hydraulic_step=3600,
    duration=604800,
    seed=42,
    world_options='gym4real/envs/wds/world_anytown.yaml'
)
env = gym.make('gym4real/wds-v0', settings=params)

print("=== Action Space ===")
print(f"Type: {env.action_space}")
print(f"n: {env.action_space.n}")

print("\n=== Observation Space ===")
print(f"Type: {env.observation_space}")
print(f"Shape: {env.observation_space.shape}")
print(f"Low: {env.observation_space.low}")
print(f"High: {env.observation_space.high}")

obs, info = env.reset()
print(f"\n=== Sample Observation ===")
print(f"Shape: {obs.shape}")
print(f"Values: {obs}")

env.close()
