#!/usr/bin/env python3
import os
import gymnasium as gym
import gym4real
from gym4real.envs.wds.utils import parameter_generator
from gym4real.envs.wds.hourly_wrapper import HourlyDecisionWrapper

if os.path.exists('gym4ReaL'):
    os.chdir('gym4ReaL')

params = parameter_generator(
    hydraulic_step=3600,
    duration=604800,
    seed=42,
    world_options='gym4real/envs/wds/world_anytown.yaml'
)
base_env = gym.make('gym4real/wds-v0', settings=params)
env = HourlyDecisionWrapper(base_env)

print(">>> Running one debug episode...")
obs, info = env.reset()
last_t = info.get("elapsed_time", 0.0)
print(f"step 0: elapsed_time = {last_t}")

step = 0
while True:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    step += 1
    t = info.get("elapsed_time", 0.0)
    dt = t - last_t
    print(f"step {step:4d}: elapsed_time = {t:8.1f}, Δt = {dt:6.1f}, reward = {reward:.3f}")
    last_t = t
    if terminated or truncated:
        print("\n=== Episode ended ===")
        print(f"Total steps: {step}")
        print(f"Final elapsed_time: {t}")
        break

env.close()
