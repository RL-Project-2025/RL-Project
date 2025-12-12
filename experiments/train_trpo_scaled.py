import os
import gymnasium as gym
import gym4real

from gym4real.envs.wds.utils import parameter_generator
from gym4real.envs.wds.reward_scaling_wrapper import RewardScalingWrapper

from sb3_contrib import TRPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize


# Environment
def make_env():
    package_root = os.path.dirname(gym4real.__file__)
    world_file = os.path.join(
        package_root,
        "envs", "wds", "world_anytown_fixed.yaml"
    )

    params = parameter_generator(world_file)

    env = gym.make("gym4real/wds-v0", settings=params)
    env = RewardScalingWrapper(env)   

    return env


# Vectorise + normalise
env = DummyVecEnv([make_env])
env = VecMonitor(env)

env = VecNormalize(
    env,
    norm_obs=True,
    norm_reward=True,
    clip_obs=10.0
)


# Logging
log_path = "logs/trpo_wds_scaled/"
os.makedirs(log_path, exist_ok=True)

logger = configure(log_path, ["tensorboard"])


# Model
model = TRPO(
    policy="MlpPolicy",
    env=env,
    gamma=1.0, 
    learning_rate=3e-4,
    batch_size=128,
    verbose=1,
)

model.set_logger(logger)


# Training
TIMESTEPS = 10_000
model.learn(total_timesteps=TIMESTEPS, progress_bar=True)


# Save
model.save("models/trpo_wds_scaled")
env.save("models/trpo_wds_scaled_vecnorm.pkl")

env.close()

print("Training complete. Model saved as trpo_wds_scaled")
