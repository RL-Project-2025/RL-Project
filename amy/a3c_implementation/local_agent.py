from a2c import ActorCritic
import torch.multiprocessing as mp
from shared_optimiser import SharedAdam
from make_env import make_env
from torch.utils.tensorboard import SummaryWriter 

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
        self.global_agent = global_actor_critic #save this ready to 'send updates' to the global agent
        self.worker_name = worker_name
        self.global_episode_idx = global_episode_idx
        
        self.shared_optimiser = shared_optimiser
        

        self.is_normalising_rewards = is_normalising_rewards
        self.is_scaling_rewards = is_scaling_rewards
        self.is_using_ema = is_using_ema
        self.MAX_EPISODE_COUNT = max_episode_count
        self.GLOBAL_AGENT_UPDATE_INTERVAL = global_update_interval
        self.log_dir = log_dir
        self.run_name = logging_run_name
        self.is_logging = is_logging


    def run(self) -> None:
        """
        this function that will be run by the worker start() function of torch.multiprocessing

        run() initialises a new environment, unique for this local agent - it then trains the agent on the 
        env and updates the global agent at the specified interval
        """

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
                    loss = self.local_agent.calc_loss(done)
                    self.shared_optimiser.zero_grad()
                    loss.backward()

                    # update global agent with local agent's gradients 
                    agent_parameters = zip(self.local_agent.parameters(), self.global_agent.parameters())
                    for local_param, global_param in agent_parameters:
                        global_param._grad = local_param.grad

                    self.shared_optimiser.step()
                    # load the latest parameters from the global agent to the local agent 
                    # use state_dict here as it will map the parameters to the layers 
                    global_agent_state_dict = self.global_agent.state_dict()
                    self.local_agent.load_state_dict(global_agent_state_dict)

                    # clear memory ready for next episode 
                    self.local_agent.rollout_buffer.clear()

                local_time_step += 1
                obs = next_obs

            # when episode has finished 
            # print to console so can monitor progress of model
            print(self.worker_name, "Episode reward: ", episode_reward)

            if self.is_logging:
                # passing a reference to the SummaryWriter in the LocalAgent __init__ parameters triggered the following error:
                # TypeError: cannot pickle '_thread.lock' object
                # this seems to be a incompatibility issue between TensorBoard SummaryWriter and PyTroch multiprocessing
                # using a workaround suggested in the PyTorch discussion forums:
                # https://discuss.pytorch.org/t/thread-lock-object-cannot-be-pickled-when-using-pytorch-multiprocessing-package-with-spawn-method/184953/3
                SummaryWriter(f"{self.log_dir}/{self.run_name}").add_scalar('episode reward', episode_reward, self.global_episode_idx.value)
            
            with self.global_episode_idx.get_lock(): #lock the variable before updating as another local agent could be trying to access this variable 
                # print(self.global_episode_idx.value) #can be uncommented to check if training stops before MAX_EPISODE_COUNT
                self.global_episode_idx.value += 1 #update the value of the episode index and not the local reference to it






