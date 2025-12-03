import os
import gymnasium as gym
import gym4real
from gym4real.envs.wds.utils import parameter_generator
from sb3_contrib import QRDQN
from stable_baselines3.common.logger import configure

print(">>> train_qrdqn.py is running...")

# Load env
package_root = os.path.dirname(gym4real.__file__)
world_file = os.path.join(package_root, "envs", "wds", "world_anytown_fixed.yaml")

params = parameter_generator(world_file)
env = gym.make("gym4real/wds-v0", settings=params)

# Create a logs/ folder 
log_path = "logs/trpo_wds/"
os.makedirs(log_path, exist_ok=True)

# Tell SB3 to log to TensorBoard format
new_logger = configure(log_path, ["tensorboard"])

# Create QR-DQN model
model = QRDQN(
    policy="MlpPolicy",
    env=env,
    learning_rate=1e-4,
    buffer_size=100_000,
    learning_starts=10_000,
    batch_size=256,
    gamma=0.99,
    train_freq=1,
    gradient_steps=1,
    target_update_interval=1_000,
    exploration_fraction=0.3,
    exploration_final_eps=0.05,
    verbose=1,
)

model.set_logger(new_logger)

# Training
TIMESTEPS = 20000  
model.learn(total_timesteps=TIMESTEPS, progress_bar=True)

# Save model
os.makedirs("models", exist_ok=True)
model.save("models/qrdqn_wds.zip")
env.close()

print("Training complete. Model saved as models/qrdqn_wds.zip")
