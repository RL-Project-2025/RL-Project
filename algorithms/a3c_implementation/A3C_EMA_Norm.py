from local_agent import LocalAgent 
from make_env import make_env
from a2c import ActorCritic
from shared_optimiser import SharedAdam
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter 
import math
import time
from datetime import timedelta
import torch
from logging_util import convert_all_logs_to_single_file


if __name__ == '__main__': # 'if clause protection' needed here otherwise multiprocessig triggers an error

    # ******************** HYPERPARAMETERS
    MAX_TIMESTEP_COUNT = 200000
    EPISODE_DURATION = 168
    MAX_EPISODE_COUNT = math.floor(MAX_TIMESTEP_COUNT / EPISODE_DURATION) #safety net to prevent infinite looping
    #the paper introducing A3C suggests global agent should be updated every 5 actions [section 8 "Experimental setup": https://arxiv.org/pdf/1602.01783]
    GLOBAL_AGENT_UPDATE_INTERVAL = 5
    GAMMA = 0.99 #place more emphasis on long term outcomes
    LEARNING_RATE = 1e-5 #smaller learning rate as batch size is 5 (as recommended by the A3C paper)
    NUM_LOCAL_AGENTS = mp.cpu_count() #might need to limit here for HEX
    IS_TB_LOGGING = True
    IS_SCALING_REWARDS = True

    # boolean flags for our chosen model variations
    IS_NORMALISING_REWARDS = True
    IS_USING_EMA = True
    # ********************

    # ******************** SET UP FILE PATHS FOR TB LOGGING AND MODEL SAVING
    RUN_NAME = "A3C_EMA_Norm"
    LOG_DIR = './a3c_logs'
    MODEL_DIR_PATH = './a3c_model_files'
    # ********************

    writer = SummaryWriter(f"{LOG_DIR}/{RUN_NAME}")

    # set up an env just to extract the dimensions of the action and observation space (rather than hardcoding the number in)
    env = make_env(use_normalisation=True, reward_scaling=True, use_ema=True) # (this env is not used by local agents)
    obs_space_dim = env.observation_space.shape[0]
    action_space_dim = env.action_space.n

    # set up global actor critic agent
    torch.manual_seed(42) #A3C appears to be very sensitive to weight initialisation - so set a seed to make experiments comparable
    global_agent = ActorCritic(obs_space_dim, action_space_dim, gamma = GAMMA)
    global_agent.share_memory()
    shared_optimiser = SharedAdam(global_agent.parameters(), lr = LEARNING_RATE)
    global_episode_index = mp.Value('i', 0) #i=unsinged integer here 
    global_timestep_index = mp.Value('i', 0) #i=unsinged integer here 

    # set up multiple local agents
    local_agents = [LocalAgent(
                                global_actor_critic =  global_agent,
                                shared_optimiser = shared_optimiser,
                                obs_space_dim = obs_space_dim,
                                action_space_dim = action_space_dim, 
                                gamma = GAMMA,
                                worker_name = f"worker_{i}",
                                global_episode_idx = global_episode_index,
                                global_timestep_idx = global_timestep_index,
                                is_normalising_rewards = IS_NORMALISING_REWARDS,
                                is_scaling_rewards = IS_SCALING_REWARDS,
                                is_using_ema = IS_USING_EMA,
                                max_episode_count = MAX_EPISODE_COUNT,
                                global_update_interval = GLOBAL_AGENT_UPDATE_INTERVAL,
                                is_logging = IS_TB_LOGGING,
                                log_dir = LOG_DIR,
                                logging_run_name = RUN_NAME) 
                    for i in range(NUM_LOCAL_AGENTS)]
    
    # start multithreaded process
    start_time = time.time()
    [local_agent.start() for local_agent in local_agents]
    [local_agent.join() for local_agent in local_agents]
    end_time = time.time()
    elapsed_time = end_time-start_time

    writer.add_text('hyperparameters',
        f"lr={LEARNING_RATE}, gamma={GAMMA}, "
        f"global_agent_update_interval={GLOBAL_AGENT_UPDATE_INTERVAL}, "
        f"max_episode_count={MAX_EPISODE_COUNT}, "
        f"is_normalising_rewards={IS_NORMALISING_REWARDS}, is_scaling_rewards={IS_SCALING_REWARDS}, "
        f"is_using_ema={IS_USING_EMA}, num_local_agents={NUM_LOCAL_AGENTS}, "
        f"execution_time={str(timedelta(seconds=elapsed_time))}")
    writer.close()

    # A3C logging requires a work around to avoid pickling errors raised by torch.multiprocessing
    convert_all_logs_to_single_file(input_dir=f"{LOG_DIR}/{RUN_NAME}", output_dir=f"{LOG_DIR}/{RUN_NAME}_compiled")
    # save the model to .pt file
    torch.save(global_agent.state_dict(), f"{MODEL_DIR_PATH}/{RUN_NAME}.pt")

