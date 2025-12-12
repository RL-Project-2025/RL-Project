# INSTRUCTIONS
# SHOULD BE PLACED INTO A FOLDER CALLED 'envs' AT ROOT OF DIRECTORY
# FOR CALLING AND USAGE EXAMPLES OF HOW TO USE THE WRAPPERS
# PLEASE SEE MY 'experiments' FOLDER (Reece branch)
# TO LOOK INTO THE BASELINE MODELS USED FOR TESTING COMPARISON, SEE MY 'evaluation' FOLDER

import os
import gymnasium as gym
import gym4real
from gym4real.envs.wds.utils import parameter_generator
from gym4real.envs.wds.hourly_wrapper import HourlyDecisionWrapper
from gym4real.envs.wds.reward_scaling_wrapper import RewardScalingWrapper
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize


def make_env(
    use_wrapper: bool,
    use_normalisation: bool,
    reward_scaling: bool = False,
):
    def _init():
        package_root = os.path.dirname(gym4real.__file__)
        world_file = os.path.join(
            package_root, "envs", "wds", "world_anytown_fixed.yaml"
        )

        params = parameter_generator(world_file)
        env = gym.make("gym4real/wds-v0", settings=params)

        if reward_scaling:
            env = RewardScalingWrapper(env) 

        if use_wrapper:
            env = HourlyDecisionWrapper(env)

        return env

    env = DummyVecEnv([_init])
    env = VecMonitor(env)

    if use_normalisation:
        env = VecNormalize(env, norm_obs=True, norm_reward=True)

    return env
