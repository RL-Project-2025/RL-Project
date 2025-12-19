from local_agent import LocalAgent 
from make_env import make_env
from a2c import ActorCritic
from shared_optimiser import SharedAdam
import torch.multiprocessing as mp

if __name__ == '__main__': # 'if clause protection' needed here otherwise it triggers the following error:
    # RuntimeError: 
        # An attempt has been made to start a new process before the
        # current process has finished its bootstrapping phase.
    # as explained by https://docs.pytorch.org/docs/stable/notes/windows.html#multiprocessing-error-without-if-clause-protection 

    # **** HYPERPARAMETERS
    MAX_EPISODE_COUNT = 200000 #safety net to prevent infinite looping 
    #the paper introducing A3C suggests global agent should be updated every 5 actions
    # ref section 8 "Experimental setup" of the paper: https://arxiv.org/pdf/1602.01783 
    GLOBAL_AGENT_UPDATE_INTERVAL = 5 
    GAMMA = 0.99
    # ********************

    env = make_env(use_normalisation=True, reward_scaling=True) # used to extract the dimensions of the action space (rather than hardcoding the number in)
    obs_space_dim = env.observation_space.shape[0]
    action_space_dim = env.action_space.n

    global_agent = ActorCritic(obs_space_dim, action_space_dim, gamma = GAMMA)
    global_agent.share_memory()
    shared_optimiser = SharedAdam(global_agent.parameters()) #use the default learning rate for now
    global_episode_index = mp.Value('i', 0) #i=unsinged integer here 

    num_cores = mp.cpu_count()
    local_agents = [LocalAgent(
                                global_actor_critic =  global_agent,
                                shared_optimiser = shared_optimiser,
                                obs_space_dim = obs_space_dim,
                                action_space_dim = action_space_dim, 
                                gamma = GAMMA,
                                worker_name = f"worker_{i}",
                                global_episode_idx = global_episode_index,
                                is_normalising_rewards = True,
                                is_scaling_rewards = True,
                                max_episode_count = MAX_EPISODE_COUNT,
                                global_update_interval = GLOBAL_AGENT_UPDATE_INTERVAL) 
                    for i in range(num_cores)]
    
    # start multithreaded process
    [local_agent.start() for local_agent in local_agents]
    [local_agent.join() for local_agent in local_agents]