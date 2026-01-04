# Imports for logging
from torch.utils.tensorboard import SummaryWriter 
import math
import time
from datetime import timedelta
import os
from tbparse import SummaryReader #!pip install tbparse

# Imports for A3C Implementation 
import torch.multiprocessing as mp
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import softmax
import torch.optim as optim
from torch.distributions import Categorical
import torch.nn.functional as F

# Imports related to environment setup
import gymnasium as gym
from gym4real.envs.wds.utils import parameter_generator
from gym4real.envs.wds.reward_scaling_wrapper import RewardScalingWrapper

# **************************************** UTILITY FUNCTIONS
# class to normalise the observations of the env
class RunningMeanStd:

    '''
        Goal: Track running mean and variance using Welford's parallel online algorithm
    '''

    def __init__(self, shape=()):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4 # Div by 0 bad, so initially setting it to this

    # Update the stats with a batch of samples
    def update(self, x):
        x = np.asarray(x)
        if x.ndim == 1:
            x = x.reshape(1, -1) # As its a single obs, itd be a batch of 1

        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]

        self.update_from_moments(batch_mean, batch_var, batch_count)

    # Chan's parallel algorithm for combining statistics.
    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        self.mean = self.mean + delta * batch_count / total_count # New mea being the weighted combination

        # New var being the parallel algorithm
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + (delta ** 2) * self.count * batch_count / total_count # Sum of squared deviations from mean!
        self.var = M2 / total_count

        self.count = total_count

    def normalise(self, x):
        return (x - self.mean) / np.sqrt(self.var + 1e-8) # z-score norm


class NormaliseObservation(gym.Wrapper):
    '''
        Gym wrapper for obs norm using running statistics.
        
        Aim:
            During training: updates stats and normalises
            During evaluation: normalises only (set training=False)
    '''

    def __init__(self, env):
        super().__init__(env)
        self.rms = RunningMeanStd(shape=env.observation_space.shape)
        self.training = True

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.training:
            self.rms.update(obs)
        return self.rms.normalise(obs), reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if self.training:
            self.rms.update(obs)
        return self.rms.normalise(obs), info
    
def make_env(use_normalisation: bool,
                reward_scaling: bool,
                use_ema: bool) -> NormaliseObservation:
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

def convert_all_logs_to_single_file(input_dir: str, output_dir: str):
    """
    Combines multiple events.out.tfevents files into a single file 

    A3C implementation currently results in multiple events.out.tfevents files (due to a work around addressing the multiprocessing pickling error)
    when training for 200k timesteps the number of events.out.tfevents files exceeds the buffer size of the tensorboard UI ;
    so this function merges the multiple files into a single file ready for display in the tensorboard UI

    Params:
        input_dir: the path to the folder which currently stores the multiple events.out.tfevents files
        output_dir: folder path for the resulting single events.out.tfevent file to go in
    """
    # create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    reader = SummaryReader(input_dir)
    tensorboard_writer = SummaryWriter(output_dir)
    # extract scalars
    df_scalars = reader.scalars
    for index, row in df_scalars.iterrows():
        tensorboard_writer.add_scalar(row['tag'], row['value'], int(row['step']))

    # extract tensors
    df_tensors = reader.tensors
    for index, row in df_tensors.iterrows():
        tensorboard_writer.add_text(row['tag'], str(row['value']), int(row['step']))

    tensorboard_writer.close()
    print(f"Logs compiled into directory: {output_dir}")
# **************************************** END OF UTILITY FUNCTIONS 


# **************************************** A3C IMPLEMENTATION
class RolloutBuffer:
    '''
    Branching slightly off of the papers implementation, fundementally I believe the buffer is purely meant to store and retrieve.
    Which is why compute_gae() is a standalone and get() returns the full buffer.
    '''
    def __init__(self):
        self.clear()
    
    # Aims to append a single timestep data
    def store(self, obs, action, reward, log_prob, value, done):
        self.obs.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)
    
    # Reset for next rollout
    def clear(self):
        self.obs, self.actions, self.rewards = [], [], []
        self.log_probs, self.values, self.dones = [], [], []

    
    # Algorithm 1 (Page 5): "optimize it with minibatch SGD... for K epochs."
    # Which means we would need to shuffle and yield mini batches
    # As of now get() returns the full buffer, which means ill handle batching in the training loop instead 
    def get(self):
        return (
            torch.tensor(np.array(self.obs), dtype=torch.float32),
            torch.tensor(self.actions, dtype=torch.long),
            torch.tensor(self.rewards, dtype=torch.float32),
            torch.tensor(self.log_probs, dtype=torch.float32),
            torch.tensor(self.values, dtype=torch.float32),
            torch.tensor(self.dones, dtype=torch.float32)
        )
    
class ActorCritic(nn.Module):
    '''
    The Actor-Critic class aims to follow the papers implementation
    '''

    # Paper notes(Page 6): We dont share parameters between the policy and value function
    # However the problem we have at hand is not an Atari problem but rather a discrete problem! So sharing should be fine
    # For discrete actions theres no log_std, which only matters when theres a cont action space as it aims to output a Gaussian
    def __init__(self, obs_dim, act_dim, gamma, hidden=64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh()
        )
        self.actor = nn.Linear(hidden, act_dim)

        self.critic = nn.Linear(hidden, 1)

        # **** A2C class modified to have rollout buffer as class property
        self.rollout_buffer = RolloutBuffer()
        self.gamma = gamma
    
    # Returns both policy dist params and value estimate
    def forward(self, x):
        h = self.shared(x)
        return self.actor(h), self.critic(h)
    
    # SAMPLE action from policy for enviroment interaction
    def act(self, obs):
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), value.squeeze(-1)
    
    # Get action from policy for env interaction
    # Returns actions needed for env.step (action, log_prob, value) and stores the internals (log_prob & value) for later use
    def select_action(self, obs):
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0) # use of unsqueeze as act() expects batched input and single obs is shape (obs dim) NOT (1, obs_dim)

        with torch.no_grad(): # No grad to reduce memory consumption as I wont be be calling backward():
            action, log_prob, value = self.act(obs_t)
        return action.item(), log_prob.item(), value.item() #Converting to scalars before returning
    
    def calc_discounted_return(self, terminal_flag: int) -> torch.Tensor:
        """
            Calculates the discounted reward as set out in the paper (page 14) https://arxiv.org/pdf/1602.01783

            Returns an array of discounted rewards for the batch of observations (states) stored in the rollout buffer 
        """
        
        rollout_buffer_obs = torch.tensor(np.array(self.rollout_buffer.obs), dtype=torch.float32) #convert to tensor ready for the forward pass
        logits, expected_returns = self.forward(rollout_buffer_obs)
        # loop backward (as suggested by the pseudocode in page 14 of the paper https://arxiv.org/pdf/1602.01783)
        # reverse and convert to numpy to detach from computational graph
        discounted_return = 0
        batch_discounted_returns = []
        if terminal_flag != 1: #if not in the terminal state
            discounted_return = expected_returns[-1].detach().item()

        reversed_exp_returns = expected_returns.detach().flatten().tolist()[::-1]
            
        for i, exp_return in enumerate(reversed_exp_returns):
            if i!=0 and terminal_flag == 1:
                discounted_return = exp_return + self.gamma*discounted_return
            batch_discounted_returns.append(discounted_return)          

        batch_discounted_returns.reverse()

        # OLD return statement
        # return torch.tensor(batch_discounted_returns, dtype=torch.float32)
        # experimented with scaling the batch_discounted_returns and found that this significantly reduced the noise in both loss functions (actor and critic)
        return torch.tensor(batch_discounted_returns, dtype=torch.float32) * 0.01
        
    
    # **ADDED METHOD
    def calc_loss(self, terminal_flag: int) -> tuple[float, float]:
        """
            Method added so that local agents can compute loss on the fly and send updates to the global agent

            Returns actor loss and critic loss seperately for the purposes of logging
        """
        batch_discounted_rewards = self.calc_discounted_return(terminal_flag)

        rollout_buffer_obs = torch.tensor(np.array(self.rollout_buffer.obs), dtype=torch.float32) #convert to tensor ready for the forward pass
        logits, vals = self.forward(rollout_buffer_obs)
        #difference between returns and values - will be used in calculation of both actor and critic loss
        advantages = batch_discounted_rewards - vals.squeeze(-1) # use squeeze to match dimensions

        # calculate L2 loss for the critic
        # critic_loss = advantages**2

        # switch to L1 loss to address exploding gradients
        critic_loss = F.smooth_l1_loss(vals.view(-1), batch_discounted_rewards.view(-1))

        # calculate loss for the actor 
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(torch.tensor(self.rollout_buffer.actions, dtype=torch.float32))
        actor_loss = -(log_probs*advantages)

        # get entropy ready for loss calculation 
        entropy = dist.entropy().mean()

        return critic_loss.mean(), actor_loss.mean(), entropy

# the SharedAdam class is sourced from: https://github.com/philtabor/Youtube-Code-Repository/blob/master/ReinforcementLearning/PolicyGradient/A3C/pytorch/a3c.py
# the class is used to share optimiser parameters between the global and local agent - ready for multiprocessing
class SharedAdam(torch.optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8,
            weight_decay=0):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps,
                weight_decay=weight_decay)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()
# end of code reference

class LocalAgent(mp.Process):
    """
    Initialises a worker that will be run by PyTorch multiprocessing
    
    """
    def __init__(self,
                 global_actor_critic: ActorCritic,
                 shared_optimiser: SharedAdam,
                 obs_space_dim: int,
                 action_space_dim: int, 
                 gamma: float,
                 worker_name: str,
                 global_episode_idx: int,
                 global_timestep_idx: int,
                 is_normalising_rewards: bool,
                 is_scaling_rewards: bool,
                 is_using_ema: bool,
                 max_episode_count: int,
                 global_update_interval: int,
                 is_logging : bool = False, 
                 log_dir: str = None,
                 logging_run_name: str = None,
                 ) -> None:
        """
        Initialises the instance properties needed for a local agent 

        Params:
            global_actor_critic: a reference to the global actor critic network 
            shared_optimiser: an optimiser that is shared between the global agent and local agents
            obs_space_dim: dimension of the observation space
            action_space_dim: dimension of the action space 
            gamma: discount factor 
            worker_name: name of the local agent that torch.multiprocessing will asign
            global_episode_idx: total number of episoder run across all local agents
            is_normalising_rewards: boolean flag to be passed to the make_env function 
            is_scaling_rewards: boolean flag to be passed to the make_env function 
            is_using_ema: boolean flag to determine whether the environment should use regular moving average or exponential moving average
            max_episode_count: maximum number of episodes to be run by all agents
            global_update_interval: the interval at which a local agent should update the gradients of the global agent
            is_logging: boolean flag to stipulate whether results should be logged (--> can set to false when debugging)
            log_dir: the folder which episode rewards should be logged to - used to initialise the SummaryWriter
            log_run_name: the name of the run for logging purposes
        
        """
        super().__init__()

        self.gamma = gamma
        self.local_agent = ActorCritic(obs_space_dim, action_space_dim, self.gamma)
        # use orthogonal initialisation for all linear layers in the local agent - to mitigate exploding gradients
        for name, module in self.local_agent.named_modules():
            if type(module) == torch.nn.modules.linear.Linear:
                torch.nn.init.orthogonal_(module.weight, 1)

        self.global_agent = global_actor_critic #save this ready to 'send updates' to the global agent
        self.worker_name = worker_name
        self.global_episode_idx = global_episode_idx
        self.global_timestep_idx = global_timestep_idx
        self.shared_optimiser = shared_optimiser
        self.is_normalising_rewards = is_normalising_rewards
        self.is_scaling_rewards = is_scaling_rewards
        self.is_using_ema = is_using_ema
        self.log_dir = log_dir
        self.run_name = logging_run_name
        self.is_logging = is_logging

        self.MAX_EPISODE_COUNT = max_episode_count
        self.GLOBAL_AGENT_UPDATE_INTERVAL = global_update_interval
        self.CRITIC_COEF = 0.5
        self.ENTROPY_COEF = 0.01
        self.MAX_GRAD_NORM = 0.5

        # attempted defining SummaryWriter as class property but it lead to Pickling errors when PyTorch multiprocessing called .start()
        # self.summary_writer = SummaryWriter(f"{self.log_dir}/{self.run_name}")


    def run(self) -> None:
        """
        this function that will be run by the worker start() function of torch.multiprocessing

        run() initialises a new environment, unique for this local agent - it then trains the agent on the 
        env and updates the global agent at the specified interval
        """
        # the approach used in the run() function follows the general approach laid out in: https://www.youtube.com/watch?v=OcIx_TBu90Q&t=1482s
        # however our implementation of run() has been modified to:
        # - work with our custom implementation of a2c 
        # - add entropy to the loss calculation 
        # - add a weight (critic_coef to the loss calculation)
        # - clip the gradients of the local and global agents to mitigate exploding gradients
        # - log results to tensorboard 


        # had to move environment creation to the run function - otherwise it triggers the error:
        # make a local copy of the environment ready for the local agent to explore 
        # AttributeError: Can't get local object 'CDLL.__init__.<locals>._FuncPtr'
        self.local_env = make_env(use_normalisation=self.is_normalising_rewards, 
                                  reward_scaling=self.is_scaling_rewards,
                                  use_ema=self.is_using_ema)
        
        local_time_step = 1
        while self.global_episode_idx.value < self.MAX_EPISODE_COUNT:

            done = False
            obs, info = self.local_env.reset()
            episode_reward = 0
            self.local_agent.rollout_buffer.clear() # clear agent memory ready for start of episode

            while not done:
                action, log_prob, value = self.local_agent.select_action(obs)
                next_obs, reward, terminated, truncated, _ = self.local_env.step(action)
                done = terminated or truncated
                self.local_agent.rollout_buffer.store(obs, action, reward, log_prob, value, done)
                episode_reward += reward

                is_time_to_update = local_time_step % self.GLOBAL_AGENT_UPDATE_INTERVAL ==0
                if is_time_to_update or done:
                    # calculate loss for local agent 
                    critic_loss, actor_loss, entropy = self.local_agent.calc_loss(done)
                    loss = (self.CRITIC_COEF * critic_loss) + actor_loss - (self.ENTROPY_COEF * entropy)
                    self.shared_optimiser.zero_grad()
                    loss.backward()

                    # update global agent with local agent's gradients 
                    agent_parameters = zip(self.local_agent.parameters(), self.global_agent.parameters())
                    for local_param, global_param in agent_parameters:
                        global_param._grad = local_param.grad

                    # address outlier gradients that could cause unstable updates
                    nn.utils.clip_grad_norm_(self.local_agent.parameters(), self.MAX_GRAD_NORM)
                    nn.utils.clip_grad_norm_(self.global_agent.parameters(), self.MAX_GRAD_NORM)
                    self.shared_optimiser.step()
                    # load the latest parameters from the global agent to the local agent 
                    # use state_dict here as it will map the parameters to the layers 
                    global_agent_state_dict = self.global_agent.state_dict()
                    self.local_agent.load_state_dict(global_agent_state_dict)

                    # clear memory ready for next episode 
                    self.local_agent.rollout_buffer.clear()

                local_time_step += 1
                obs = next_obs
                with self.global_timestep_idx.get_lock():
                    self.global_timestep_idx.value += 1
                

            # when episode has finished 
            # print to console so can monitor progress of model
            print(self.worker_name, "Episode reward: ", episode_reward)

            if self.is_logging:
                # passing a reference to the SummaryWriter in the LocalAgent __init__ parameters triggered the following error:
                # TypeError: cannot pickle '_thread.lock' object
                # this seems to be a incompatibility issue between TensorBoard SummaryWriter and PyTroch multiprocessing
                # using a workaround suggested in the PyTorch discussion forums:
                # https://discuss.pytorch.org/t/thread-lock-object-cannot-be-pickled-when-using-pytorch-multiprocessing-package-with-spawn-method/184953/3
                SummaryWriter(f"{self.log_dir}/{self.run_name}").add_scalar('episode reward', episode_reward, self.global_timestep_idx.value)

                # global agent does not calculate loss as it does not perform actions - instead it aggregates the gradients from local agents 
                # so log the local agent loss that is used for the update in each episode 
                SummaryWriter(f"{self.log_dir}/{self.run_name}").add_scalar('Actor loss', actor_loss.item(), self.global_timestep_idx.value)
                SummaryWriter(f"{self.log_dir}/{self.run_name}").add_scalar('Critic loss', critic_loss.item(), self.global_timestep_idx.value)
                SummaryWriter(f"{self.log_dir}/{self.run_name}").add_scalar('Loss', loss.item(), self.global_timestep_idx.value)

                SummaryWriter(f"{self.log_dir}/{self.run_name}").add_scalar('Entropy', entropy.item(), self.global_timestep_idx.value)

            with self.global_episode_idx.get_lock(): #lock the variable before updating as another local agent could be trying to access this variable 
                # print(self.global_episode_idx.value) #can be uncommented to check if training stops before MAX_EPISODE_COUNT
                self.global_episode_idx.value += 1 #update the value of the episode index and not the local reference to it


# **************************************** END OF A3C IMPLEMENTATION



# ****************************************TRAINING SCRIPT
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
    IS_NORMALISING_REWARDS = False
    IS_USING_EMA = True
    # ********************

    # ******************** SET UP FILE PATHS FOR TB LOGGING AND MODEL SAVING
    RUN_NAME = "A3C_EMA_NoNorm"
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
# **************************************** END OF TRAINING SCRIPT