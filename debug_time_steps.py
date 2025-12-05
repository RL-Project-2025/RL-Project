import os
import gymnasium as gym
import gym4real
from gym4real.envs.wds.utils import parameter_generator

package_root = os.path.dirname(gym4real.__file__)
world_file = os.path.join(package_root, "envs", "wds", "world_anytown_fixed.yaml")

params = parameter_generator(world_file)
env = gym.make("gym4real/wds-v0", settings=params)

print(">>> Running one debug episode...")
obs, info = env.reset()
last_t = info.get("elapsed_time", 0.0)
print(f"step 0: elapsed_time = {last_t}")

step = 0
while True:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    step += 1
    t = info.get("elapsed_time", None)
    dt = None if t is None else t - last_t
    print(f"step {step:4d}: elapsed_time = {t:8.1f}, Δt = {dt:6.1f}, reward = {reward:.3f}")
    last_t = t

    if terminated or truncated:
        print("\n=== Episode ended ===")
        print(f"Total steps: {step}")
        print(f"Final elapsed_time: {t}")
        break

env.close()