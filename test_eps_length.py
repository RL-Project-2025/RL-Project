import gymnasium as gym
import gym4real
from gym4real.envs.wds.utils import parameter_generator
import os

print(">>> Checking episode length...")

package_root = os.path.dirname(gym4real.__file__)
world_file = os.path.join(package_root, "envs", "wds", "world_anytown_fixed.yaml")

params = parameter_generator(world_file)
env = gym.make("gym4real/wds-v0", settings=params)

obs, info = env.reset()
step_count = 0
done = False

while not done:
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    done = terminated or truncated
    step_count += 1

print("\n=== Episode Length ===")
print(step_count)

env.close()
