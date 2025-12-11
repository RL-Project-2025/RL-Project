import os
import gymnasium as gym
import gym4real
from gym4real.envs.wds.utils import parameter_generator

if os.path.exists('gym4ReaL'):
    os.chdir('gym4ReaL')


params = parameter_generator(
    hydraulic_step=3600,
    duration=604800,
    seed=42,
    world_options='gym4real/envs/wds/world_anytown.yaml'
)
env = gym.make('gym4real/wds-v0', settings=params)

print("=== Observation Space ===")
print(f"Type: {type(env.observation_space)}")
print(f"Shape: {env.observation_space.shape}")
print(f"Dtype: {env.observation_space.dtype}")
print(f"Low: {env.observation_space.low}")
print(f"High: {env.observation_space.high}")

print("\n=== Action Space ===")
print(f"Type: {type(env.action_space)}")
if hasattr(env.action_space, 'n'):
    print(f"n: {env.action_space.n}")
if hasattr(env.action_space, 'shape'):
    print(f"Shape: {env.action_space.shape}")

print("\n=== Sample Observation ===")
obs, info = env.reset()
print(f"Obs shape: {obs.shape}")
print(f"Obs dtype: {obs.dtype}")
print(f"Obs min: {obs.min():.4f}, max: {obs.max():.4f}")
print(f"Obs values: {obs}")

print("\n=== Reset Info ===")
print(info)
