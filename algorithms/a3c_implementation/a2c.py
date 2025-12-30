import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import softmax
import torch.optim as optim
from torch.distributions import Categorical
from rollout_buffer import RolloutBuffer
import torch.nn.functional as F

# Modified from Hulusi's branch: added methods to calculate the discounted return and loss; 
# ready for local agents that need to do this on the fly for A3C
# also added rollbout buffer property to the class
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
