import gymnasium as gym
import gym4real
from gym4real.envs.wds.utils import parameter_generator
import os

from sb3_contrib import TRPO

# Load config YAML
package_root = os.path.dirname(gym4real.__file__)
world_file = os.path.join(package_root, "envs", "wds", "world_anytown_fixed.yaml")

params = parameter_generator(world_file)

# Create Env
env = gym.make("gym4real/wds-v0", settings=params)

# Create Model
model = TRPO(
    policy="MlpPolicy",
    env=env,
    gamma=0.99,
    learning_rate=3e-4,
    verbose=1,
    batch_size=128,   
)

# Training
TIMESTEPS = 150_000   
model.learn(total_timesteps=TIMESTEPS)

# Save Model
model.save("models/trpo_wds")
env.close()

print("Training complete. Model saved as trpo_wds.zip")
