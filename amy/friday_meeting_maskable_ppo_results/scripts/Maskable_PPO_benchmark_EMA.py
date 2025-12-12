import gymnasium as gym
import numpy as np
import os
import pandas as pd #used to extract tank levels
import gym4real
from gym4real.envs.wds.utils import parameter_generator
from stable_baselines3.common.evaluation import evaluate_policy
# maskable PPO currently in the experimental features section of SB3
# need to run pip install sb3-contrib to execute this file
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback #a drop-in replacement for EvalCallback
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from gymnasium.wrappers import TimeLimit

# ************* Utility funcs / classes
def create_env():
    # params = parameter_generator(world_file)
    params = parameter_generator(
        hydraulic_step=3600,
        duration=3600 * 24 * 7,
        seed=42,
        world_options='gym4real/envs/wds/world_anytown.yaml'
    )
    env = gym.make('gym4real/wds-v0', **{'settings': params})

    return env

# Maskable PPO requires a dynamic mask that prevents the agent from taking 
# illegal/invalid actions in a given state 
def mask_fn(env: gym.Env) -> np.ndarray:
    NUM_ACTIONS = env.action_space.n #currently returns 4 as the YAML file specifies 2 pumps 
    mask = np.ones(NUM_ACTIONS, dtype=bool)
    base_env = env.unwrapped #required to access the low level environment properties
    CRITICAL_FACTOR = 0.98
    pumps = base_env._wn.pumps.keys() #the pumps we need to trace back to from the tanks 

    for tank in base_env._wn.tanks.keys():

        tank_level = base_env._wn.nodes[tank].level
        # the variable above can sometimes be a scalar - or a pandas series 
        # so need to extract it differenttly for it to be appropriate for bool check later on
        if isinstance(tank_level, pd.Series):
            current_level = tank_level.iloc[0] 
        elif isinstance(tank_level, (float, int)):
            current_level = tank_level

        max_tank_capacity = base_env._wn.tanks[tank].maxlevel
        threshold = max_tank_capacity * CRITICAL_FACTOR
        is_tank_full = current_level >= threshold

        pumps_feeding_tank = tank_pump_tracer(base_env, tank, pumps) #identified that both pumps are connected to both tanks

        if is_tank_full:
            # do not allow pumps to turn on if tank is full 
            # True marks an action as valid 
            # False marks an action as invalid 

            mask = [True, False, False, False]
            # refactor this to use pumps_feeding_tank
            # (used pumps_feeding_tank to identify that both pumps are connected to a resevoir and are both conncted to each tank)

        else:
            # no risk of overflow if tank not full - so agent is free to perform any action
            mask = [True, True, True, True]

    return mask

def tank_pump_tracer(unwrapped_env, 
                     tank_id, 
                     pump_ids_list) -> set:
    """
    Searches through the environment to determine which pumps feed the given tank
    """
    feeding_pumps = set()
    visited_nodes = []
    queue = [unwrapped_env._wn.nodes[tank_id]]

    # starting at the tank
    previous_node = unwrapped_env._wn.nodes[tank_id]
    while queue:
        current_node = queue.pop(0)

        for link_obj in current_node.__dict__['links']:
            link_id = link_obj.__dict__['uid']

            
            if link_id.startswith("PIPE"):
                connecting_pipe = unwrapped_env._wn.pipes[link_id]

                pipe_destination = connecting_pipe.__dict__['to_node']
                pipe_destination_id = pipe_destination.__dict__['uid']

                pipe_origin = connecting_pipe.__dict__['from_node'] 
                pipe_origin_id = pipe_origin.__dict__['uid']

                if pipe_destination != previous_node:
                    # proceed with the other pipe 
                    if pipe_destination_id not in visited_nodes:
                        visited_nodes.append(pipe_destination_id) 
                        queue.append(pipe_destination)
                        previous_node = pipe_destination

                if pipe_origin != previous_node:
                    if pipe_origin_id not in visited_nodes:
                        visited_nodes.append(pipe_origin_id) 
                        queue.append(pipe_origin)
                        previous_node = pipe_origin
                    

            elif link_id in pump_ids_list:
                pump_link = unwrapped_env._wn.pumps[link_id]
                
                # Access the downstream and source of the pump
                pump_destination = pump_link.__dict__['to_node']  
                pump_destination_id = pump_destination.__dict__['uid']
                pump_source_obj = pump_link.__dict__['from_node']
                pump_source_id = pump_source_obj.__dict__['uid']

                # is the current junction the destination of the pump
                if pump_destination_id == current_node.__dict__['uid']:
                    # if the pump's source is a resevoir it is feeding the tank
                    if pump_source_id.startswith('R'):
                        feeding_pumps.add(link_id)
                        continue
                
                    # need to add to visited nodes here ? so that this works for other networks
                    # where there is an intermediary pump that is not connected to a resevoir / end point ?

    # print('Pumps feeding Tank ', tank_id, ':', feeding_pumps) #print statement for debugging
    return feeding_pumps


# ************* Train the model 
env = create_env()
env = ActionMasker(env, mask_fn)  # Wrap to enable masking

# env = TimeLimit(env, max_episode_steps=250)  # Give it 250 steps to finish 168 hours

NUM_TIMESTEPS = 100000
MODEL_FILE_NAME = "Maskable_PPO_model_" + str(NUM_TIMESTEPS) + "_ts"

# Invalid action masks prevent the agent from taking illegal/invalid actions in a current state 
# this could be useful if eg the agent tries to pump more water into a tank that is already full
# invalid action masking could be helpful in real world deployment scenarios where invalid actions could have severe consequences
maskable_PPO_model = MaskablePPO(MaskableActorCriticPolicy, env, gamma=0.4, seed=32, verbose=1)
maskable_PPO_model.learn(total_timesteps=NUM_TIMESTEPS)
maskable_PPO_model.save(MODEL_FILE_NAME)

# # ************* Load the model as a test
del maskable_PPO_model #delete current model to test if loading works 
maskable_PPO_model = MaskablePPO.load(MODEL_FILE_NAME, env=env)

# ************* Evaluate the model
NUM_EVAL_EP = 20
# evaluate_policy() runs the policy for n_eval_episodes episodes and outputs the average and std return per episode
mean_reward, std_reward = evaluate_policy(maskable_PPO_model, maskable_PPO_model.get_env(), n_eval_episodes=NUM_EVAL_EP) #reward_threshold=90, warn=False
print(mean_reward, std_reward)