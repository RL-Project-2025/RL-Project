import os
import gymnasium as gym
import gym4real
from gym4real.envs.wds.utils import parameter_generator
# from gym4real.envs.wds.hourly_wrapper import HourlyDecisionWrapper - not using right ? 
from gym4real.envs.wds.reward_scaling_wrapper import RewardScalingWrapper
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize


# ***** will debug why the implementation isn't working with the wrappers 
# current implementation works with unwrapped env - getting episode rewards of 140 / 167 - with lows of 72 
# (expected for A3C performance to fluctuate according to literature)


# def make_env(
#     # use_wrapper: bool,
#     use_normalisation: bool,
#     reward_scaling: bool = False,
# ):
    # def _init():
    #     package_root = os.path.dirname(gym4real.__file__)
    #     world_file = os.path.join(
    #         package_root, "envs", "wds", "world_anytown_fixed.yaml"     # CHANGE TO YOUR YAML
    #     )

    #     params = parameter_generator(world_file)
    #     env = gym.make("gym4real/wds-v0", settings=params)

    #     # ADDRESS THIS TypeError: 'RewardScalingWrapper' object is not callable
    #     # if reward_scaling:
    #     #     env = RewardScalingWrapper(env) 

    #     # if use_wrapper:
    #     #     env = HourlyDecisionWrapper(env)

    #     return env

    # env = DummyVecEnv([_init()]) #added brackets here - otherwise results in NoneType object
    # env = VecMonitor(env)

    # if use_normalisation:
    #     env = VecNormalize(env, norm_obs=True, norm_reward=True)

    # return env

def make_env(use_normalisation: bool,
                reward_scaling: bool = False,):
    
    params = parameter_generator(
        hydraulic_step=3600,
        duration=3600 * 24 * 7,
        seed=42,
        world_options='gym4real/envs/wds/world_anytown.yaml'
    )

    env = gym.make('gym4real/wds-v0', **{'settings': params})

    # triggers: ValueError: expected sequence of length 27 at dim 1 (got 4)
    # if reward_scaling:
    #     env = RewardScalingWrapper(env) 

    
    # triggers TypeError: 'int' object is not subscriptable - when performing the agent's chosen action on the env
    # # test to normalise rewards and observations 
    # env = DummyVecEnv([lambda: env])
    # env = VecMonitor(env)
    # env = VecNormalize(env, norm_obs=True, norm_reward=True)
    # # end of code implementing normalisation test

    return env