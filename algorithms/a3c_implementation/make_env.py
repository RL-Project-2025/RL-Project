import gymnasium as gym
from gym4real.envs.wds.utils import parameter_generator
from gym4real.envs.wds.reward_scaling_wrapper import RewardScalingWrapper
import Normalise #for type hinting
from Normalise import NormaliseObservation

def make_env(use_normalisation: bool,
                reward_scaling: bool,
                use_ema: bool) -> Normalise.NormaliseObservation:
    """
    Wraps the environment according to the parameters + adjusts the env to use either regular moving average or exponential moving average

    Returns: a wrapped version of the gym environment 
    """
    
    params = parameter_generator(
        hydraulic_step=3600,
        duration=3600 * 24 * 7,
        seed=42,
        world_options='gym4real/envs/wds/world_anytown.yaml'
    )

    if use_ema:
        params['demand_moving_average'] = False
        params['demand_exp_moving_average'] = True

    env = gym.make('gym4real/wds-v0', **{'settings': params})

    if reward_scaling:
        env = RewardScalingWrapper(env) 

    if use_normalisation:
        env = NormaliseObservation(env)

    return env