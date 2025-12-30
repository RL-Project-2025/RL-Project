from local_agent import LocalAgent 
from make_env import make_env
from a2c import ActorCritic
from shared_optimiser import SharedAdam
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter 
import time
import torch
from logging_util import convert_all_logs_to_single_file


if __name__ == '__main__': # 'if clause protection' needed here otherwise it triggers the following error:
    # RuntimeError: 
        # An attempt has been made to start a new process before the
        # current process has finished its bootstrapping phase.
    # as explained by https://docs.pytorch.org/docs/stable/notes/windows.html#multiprocessing-error-without-if-clause-protection 

    # ******************** HYPERPARAMETERS
    MAX_EPISODE_COUNT = 200000 #safety net to prevent infinite looping
    #the paper introducing A3C suggests global agent should be updated every 5 actions
    # ref section 8 "Experimental setup" of the paper: https://arxiv.org/pdf/1602.01783 
    GLOBAL_AGENT_UPDATE_INTERVAL = 5 #tried 20 but no performance improvements 
    GAMMA = 0.99 #place more emphasis on long term outcomes
    LEARNING_RATE = 1e-4
    NUM_LOCAL_AGENTS = mp.cpu_count() #might need to limit here for HEX

    IS_NORMALISING_REWARDS = True
    IS_SCALING_REWARDS = True
    IS_USING_EMA = True
    IS_TB_LOGGING = True
    # ********************

    # ******************** SET UP FILE PATHS FOR TB LOGGING AND MODEL SAVING
    RUN_NAME = f"a3c_{int(time.time())}"
    LOG_DIR = './a3c_logs'
    MODEL_DIR_PATH = './a3c_model_files'
    # ********************

    writer = SummaryWriter(f"{LOG_DIR}/{RUN_NAME}")
    writer.add_text('hyperparameters',
        f"lr={LEARNING_RATE}, gamma={GAMMA}, "
        f"global_agent_update_interval={GLOBAL_AGENT_UPDATE_INTERVAL}, "
        f"max_episode_count={MAX_EPISODE_COUNT}, "
        f"is_normalising_rewards={IS_NORMALISING_REWARDS}, is_scaling_rewards={IS_SCALING_REWARDS}, "
        f"is_using_ema={IS_USING_EMA}, num_local_agents={NUM_LOCAL_AGENTS}")


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

    # set up multiple local agents
    local_agents = [LocalAgent(
                                global_actor_critic =  global_agent,
                                shared_optimiser = shared_optimiser,
                                obs_space_dim = obs_space_dim,
                                action_space_dim = action_space_dim, 
                                gamma = GAMMA,
                                worker_name = f"worker_{i}",
                                global_episode_idx = global_episode_index,
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
    [local_agent.start() for local_agent in local_agents]
    [local_agent.join() for local_agent in local_agents]

    writer.close()

    convert_all_logs_to_single_file(input_dir=f"{LOG_DIR}/{RUN_NAME}", output_dir=f"{LOG_DIR}/{RUN_NAME}_compiled")
    torch.save(global_agent.state_dict(), f"{MODEL_DIR_PATH}/{RUN_NAME}.pt")

