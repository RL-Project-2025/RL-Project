import gymnasium as gym
import numpy as np
from gym4real.envs.wds.utils import parameter_generator
from stable_baselines3.common.evaluation import evaluate_policy
# maskable PPO currently in the experimental features section of SB3
# need to run pip install sb3-contrib to execute this file
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback #a drop-in replacement for EvalCallback
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker



# ************* Utility funcs / classes
def create_env():
    params = parameter_generator()
    env = gym.make('gym4real/wds-v0', **{'settings': params})
    obs,info = env.reset()
    done = False
    return env

# MASK_FN() WORK IN PROGRESS:
# Maskable PPO requires a dynamic mask that prevents the agent from taking illegal/invalid actions 
# in a given state - currently building a proof of concept mask function for the WDSEnv 
# working on tracing which pumps are connected to which tanks 
# so that the agent is not allowed to eg pump water into a tank that is already full
def mask_fn(env: gym.Env) -> np.ndarray:
    NUM_ACTIONS = env.action_space.n #currently returns 4 as the YAML file specifies 2 pumps 
    mask = np.ones(NUM_ACTIONS, dtype=bool)
    base_env = env.unwrapped #required to access the low level environment properties
    CRITICAL_FACTOR = 0.98
    pumps = base_env._wn.pumps.keys() #the pumps we need to trace back to from the tanks 

    for tank in base_env._wn.tanks.keys():
        tank_level = base_env._wn.nodes[tank].level
        max_tank_capacity = base_env._wn.tanks[tank].maxlevel
        threshold = max_tank_capacity * CRITICAL_FACTOR
        is_tank_full = tank_level >= threshold

        # find which pumps are connected to the tank
        # pattern across the network is pipe --> junction --> pipe ...
        pipes_connecting_to_tank = base_env._wn.nodes[tank].__dict__['links']

        for connecting_pipe_id in pipes_connecting_to_tank.keys():
            current_node_id = connecting_pipe_id
            traversed_pipes = [current_node_id]

            while current_node_id not in pumps:
                print(current_node_id) #debugging - will remove this line when finished
                if current_node_id.startswith("P"): 
                    #process pipe 
                    connecting_pipe = base_env._wn.pipes[current_node_id]
                    pipe_from_node = connecting_pipe.__dict__['from_node']
                    previous_source_id = pipe_from_node.__dict__['uid']
                    current_node_id = previous_source_id
                    if current_node_id not in traversed_pipes:
                        traversed_pipes.append(current_node_id)

                elif current_node_id.startswith("J"):
                    # process junction
                    junction_obj = base_env._wn.junctions[current_node_id]
                    # Check connected pipes for an un-traversed path
                    for pipe_obj in junction_obj.__dict__['links']:
                        pipe_id = pipe_obj.__dict__['uid']
                        if pipe_id not in traversed_pipes:
                            current_node_id = pipe_id
                            traversed_pipes.append(current_node_id)
                            break
                    else:
                        # if a dead end is found
                        print(f"ERROR: Junction {current_node_id} is a dead end.")
                        break #stop the while loop if a dead end is found

            # should print a pump here - working on it :)
            print(current_node_id)

    # attributes of the unwrapped environment - helpful ref
    # print(base_env._wn.__dict__.keys())
    # dict_keys(['ep', 'inputfile', 'rptfile', 'binfile', 'vertices', 'nodes', 'junctions', 
    #            'reservoirs', 'tanks', 'links', 'pipes', 'valves', 'pumps', 'curves', 'patterns', 
    #            'solved', 'solved_for_simtime', 'times', 'duration', 'hydraulic_step', 'pattern_step', 'interactive'])
    # tanks_list = ["T41", "T42"]
    return mask

# ************* Train the model 
env = create_env()
mask_fn(env)
env = ActionMasker(env, mask_fn)  # Wrap to enable masking

NUM_TIMESTEPS = 20000
MODEL_FILE_NAME = "Maskable_PPO_model_" + str(NUM_TIMESTEPS) + "_ts"

# Invalid action masks prevent the agent from taking illegal/invalid actions in a current state 
# this could be useful if eg the agent tries to pump more water into a tank that is already full
# invalid action masking could be helpful in real world deployment scenarios where invalid actions could have severe consequences
maskable_PPO_model = MaskablePPO(MaskableActorCriticPolicy, env, gamma=0.4, seed=32, verbose=1)
maskable_PPO_model.learn(total_timesteps=NUM_TIMESTEPS)
maskable_PPO_model.save(MODEL_FILE_NAME)

# ************* Load the model as a test
del maskable_PPO_model #delete current model to test if loading works 
maskable_PPO_model = maskable_PPO_model.load(MODEL_FILE_NAME, env=env)

# ************* Evaluate the model
NUM_EVAL_EP = 20
# evaluate_policy() runs the policy for n_eval_episodes episodes and outputs the average and std return per episode
mean_reward, std_reward = evaluate_policy(maskable_PPO_model, maskable_PPO_model.get_env(), n_eval_episodes=NUM_EVAL_EP) #reward_threshold=90, warn=False
print(mean_reward, std_reward)