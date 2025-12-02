# experiments/train_qrdqn.py

import os
import gymnasium as gym
import gym4real
from gym4real.envs.wds.utils import parameter_generator
from sb3_contrib import QRDQN

print(">>> train_qrdqn.py is running...")

# ---------------------------
# Load environment settings
# ---------------------------
package_root = os.path.dirname(gym4real.__file__)
world_file = os.path.join(package_root, "envs", "wds", "world_anytown_fixed.yaml")

params = parameter_generator(world_file)
env = gym.make("gym4real/wds-v0", settings=params)

# ---------------------------
# Create QR-DQN model
# ---------------------------
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

# ---------------------------
# Train model
# ---------------------------
TIMESTEPS = 150_000  # placeholder; we may bump this later
model.learn(total_timesteps=TIMESTEPS)

# ---------------------------
# Save model
# ---------------------------
os.makedirs("models", exist_ok=True)
model.save("models/qrdqn_wds.zip")
env.close()

print("Training complete. Model saved as models/qrdqn_wds.zip")
