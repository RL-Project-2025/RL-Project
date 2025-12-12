import os
import gymnasium as gym
import gym4real

from gym4real.envs.wds.utils import parameter_generator
from gym4real.envs.wds.reward_scaling_wrapper import RewardScalingWrapper

from sb3_contrib import QRDQN
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize


print(">>> train_qrdqn_scaled.py is running...")


# Environment
def make_env():
    package_root = os.path.dirname(gym4real.__file__)
    world_file = os.path.join(
        package_root, "envs", "wds", "world_anytown_fixed.yaml"
    )

    params = parameter_generator(world_file)

    env = gym.make("gym4real/wds-v0", settings=params)

    # KEEP EPANET SMDP, scale rewards by dt
    env = RewardScalingWrapper(env)

    return env


# Vectorised + monitored + normalised
env = DummyVecEnv([make_env])
env = VecMonitor(env)
env = VecNormalize(
    env,
    norm_obs=True,
    norm_reward=True,
    clip_obs=10.0
)


# Logging
log_path = "logs/qrdqn_wds_scaled/"
os.makedirs(log_path, exist_ok=True)

logger = configure(log_path, ["tensorboard"])


# Model
model = QRDQN(
    policy="MlpPolicy",
    env=env,
    learning_rate=1e-4,
    buffer_size=100_000,
    learning_starts=10_000,
    batch_size=256,

    gamma=1.0,           
    train_freq=1,
    gradient_steps=1,
    target_update_interval=1_000,
    exploration_fraction=0.3,
    exploration_final_eps=0.05,

    verbose=1,
)

model.set_logger(logger)


# Training
TIMESTEPS = 10_000
model.learn(total_timesteps=TIMESTEPS, progress_bar=True)


# Save
os.makedirs("models", exist_ok=True)

model.save("models/qrdqn_wds_scaled")
env.save("models/qrdqn_wds_scaled_vecnorm.pkl")

env.close()

print("Training complete. Model saved as models/qrdqn_wds_scaled")
