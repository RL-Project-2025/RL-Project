from local_agent import LocalAgent 
from make_env import make_env
from a2c import ActorCritic
from shared_optimiser import SharedAdam
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter 
import time
import torch

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
    IS_USING_EMA = False
    IS_TB_LOGGING = True
    # ********************

    # set up tensorboard logging 
    run_name = f"a3c_{int(time.time())}"
    log_dir = './a3c_logs'
    writer = SummaryWriter(f"{log_dir}/{run_name}")
    writer.add_text('hyperparameters',
        f"lr={LEARNING_RATE}, gamma={GAMMA}, "
        f"global_agent_update_interval={GLOBAL_AGENT_UPDATE_INTERVAL}"
        f"max_episode_count={MAX_EPISODE_COUNT}"
        f"is_normalising_rewards={IS_NORMALISING_REWARDS}, is_scaling_rewards={IS_SCALING_REWARDS}"
        f"is_using_ema={IS_USING_EMA}")


    # get the observation and action space dimensions
    env = make_env(use_normalisation=True, reward_scaling=True, use_ema=True) # used to extract the dimensions of the action space (rather than hardcoding the number in)
    obs_space_dim = env.observation_space.shape[0]
    action_space_dim = env.action_space.n

    # set up global actor critic agent
    torch.manual_seed(42) #A3C appears to be very sensitive to weight initialisation - so set a seed to make experiments comparable
    global_agent = ActorCritic(obs_space_dim, action_space_dim, gamma = GAMMA)
    global_agent.share_memory()
    shared_optimiser = SharedAdam(global_agent.parameters(), ) #lr = LEARNING_RATE
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
                                log_dir = log_dir,
                                logging_run_name = run_name) 
                    for i in range(NUM_LOCAL_AGENTS)]
    
    # start multithreaded process
    [local_agent.start() for local_agent in local_agents]
    [local_agent.join() for local_agent in local_agents]

    writer.close()