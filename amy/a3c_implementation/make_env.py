import gymnasium as gym
from gym4real.envs.wds.utils import parameter_generator
# from gym4real.envs.wds.hourly_wrapper import HourlyDecisionWrapper - not using right ? 
from gym4real.envs.wds.reward_scaling_wrapper import RewardScalingWrapper
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize

def make_env(use_normalisation: bool,
                reward_scaling: bool = False,):
    
    params = parameter_generator(
        hydraulic_step=3600,
        duration=3600 * 24 * 7,
        seed=42,
        world_options='gym4real/envs/wds/world_anytown.yaml'
    )

    env = gym.make('gym4real/wds-v0', **{'settings': params})

    # multiprocessing now working with RewardScalingWrapper
    # although curiously it makes A3C performance worse (atm)
    if reward_scaling:
        env = RewardScalingWrapper(env) 

    # currently debugging this 
    # if use_normalisation:
        # triggers TypeError: 'int' object is not subscriptable - when performing the agent's chosen action on the env
        # test to normalise rewards and observations 
        # env = DummyVecEnv([lambda: env])
        # env = VecMonitor(env)
        # env = VecNormalize(env, norm_obs=True, norm_reward=True)
        # end of code implementing normalisation test

    return env