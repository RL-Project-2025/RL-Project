import gymnasium as gym
import gym4real
from gym4real.envs.wds.utils import parameter_generator
import os

from sb3_contrib import TRPO
from stable_baselines3.common.logger import configure


# Load config YAML
package_root = os.path.dirname(gym4real.__file__)
world_file = os.path.join(package_root, "envs", "wds", "world_anytown_fixed.yaml")

params = parameter_generator(world_file)

# Create Env
env = gym.make("gym4real/wds-v0", settings=params)

# Create a logs/ folder 
log_path = "logs/trpo_wds/"
os.makedirs(log_path, exist_ok=True)

# Tell SB3 to log to TensorBoard format
new_logger = configure(log_path, ["tensorboard"])

# Create Model
model = TRPO(
    policy="MlpPolicy",
    env=env,
    gamma=0.99,
    learning_rate=3e-4,
    verbose=1,
    batch_size=128,   
)


model.set_logger(new_logger)

# Training
TIMESTEPS = 40000  
model.learn(total_timesteps=TIMESTEPS, progress_bar=True)

# Save Model
model.save("models/trpo_wds")
env.close()

print("Training complete. Model saved as trpo_wds.zip")
