#!/usr/bin/env python3

# Credits to: https://arxiv.org/pdf/1707.06347

# Looking back,
# Super clean, 4 components
# ActorCritic -> RolloutBuff -> compute_gae -> PPO -> train_ppo

# Would be funny if I did the entire thing in numpy
import numpy as np
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

class ActorCritic(nn.Module):
    '''
    The Actor-Critic class aims to follow the papers implementation
    '''

    # Paper notes(Page 6): We dont share parameters between the policy and value function
    # However the problem we have at hand is not an Atari problem but rather a discrete problem! So sharing should be fine
    # For discrete actions theres no log_std, which only matters when theres a cont action space as it aims to output a Gaussian
    def __init__(self, obs_dim, act_dim, hidden=64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh()
        )
        self.actor = nn.Linear(hidden, act_dim)
        self.critic = nn.Linear(hidden, 1)
    
    # Returns both policy dist params and value estimate
    # Needed for the combined loss L^{CLIP+VF+S} (Equation 9, page 5)
    def forward(self, x):
        h = self.shared(x)
        return self.actor(h), self.critic(h)
    
    # SAMPLE action from policy for enviroment interaction
    # Also return log_prob (needed for ratio r_t) and value (for GAE)
    def act(self, obs):
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), value.squeeze(-1)
    
    # Given stored (s,a) pairs from rollout, recompute:
    # log π_θ(a|s) under CURRENT params (for ratio computation)
    # V(s) under current params (for value loss)
    # entropy (for exploration bonus, Equation 9)
    def evaluate(self, obs, actions):
        logits, values = self.forward(obs)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), values.squeeze(-1)

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

# The reason this is a standalone function is because its pure computation.
# Idk, seems cleaner to me not to put it into the rollout buffer.
def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    advantages = []
    gae = 0
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0 # Assuming final timestep is always terminal/ No bootstrap needed!
        else:
            next_value = values[t + 1] * (1 - dones[t])
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        # advantages.insert(0, gae), O(n^2) append
        advantages.append(gae)
    # For future me:
    # Values coming from get() are already tensors, so advantage + value works. But if they are on different devices and I want returns Ill probably get a device mismatch.
    # Look into passing in 'devices' to advantages = torch.tensor...
        # Removed all references to devices, as we will be running this on CPU # Ignore previous comments, still usefull for future reference/potential debugging advantages = advantages[::-1] advantages = torch.tensor(advantages, dtype=torch.float32)
    advantages = advantages[::-1]
    advantages = torch.tensor(advantages, dtype=torch.float32)
    returns = advantages + values
    return advantages, returns

class PPO:
    '''
    PPO with clipped surrogate objective (Algorithm 1, page 5).
    
    Uses Actor-Critic with shared layers, GAE for adv estimation, and combined loss L^(CLIP+VF+S) (Eq 9, Page 5).

    Expect alot of comments throughout.
    '''
    def __init__(self, obs_dim, act_dim, lr=3e-4, gamma=0.99, lam=0.95,
                 clip_eps=0.2, epochs=10, batch_size=64, ent_coef=0.01,
                 vf_coef=0.5, max_grad_norm=0.5):
        '''
        Hyper-params from Table 3 (Page 10)
        Adam Stepsize(Learning rate): 3x10^-4
        Discount(Gamma): .99
        GAE Param(Lambda): .95
        Batch: 64
        Epochs: 10 (Defaulting to 10, because the paper said so)
        Clipping (Epsilon): .2 (Core PPO hyperparam, paper said .2 performed the best, up for experimentation on our end)
        
        Added/Changed:
        Gradient Clipping(max_grad_norm): .5 (NOT IN PAPER, added for stability!)
        vf_coeff: Atari used 1, made it .5
        ent_coeff: .01 
        '''
        self.gamma, self.lam = gamma, lam
        self.clip_eps, self.epochs = clip_eps, epochs
        self.batch_size = batch_size
        self.ent_coef, self.vf_coef = ent_coef, vf_coef
        self.max_grad_norm = max_grad_norm # Note to self, remember to use this in update(), right before self.optimizer.step()
    
        # Please dont mind the naming    
        self.ac = ActorCritic(obs_dim, act_dim)
        self.optimizer = optim.Adam(self.ac.parameters(), lr=lr)
        self.buffer = RolloutBuffer()
    
    # Get action from policy for env interaction
    # Returns actions needed for env.step (action, log_prob, value) and stores the internals (log_prob & value) for later use
    def select_action(self, obs):
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0) # Absolute genius use of unsqueeze by me as act() expects batched input and single obs is shape (obs dim) NOT (1, obs_dim)

        with torch.no_grad(): # No grad to reduce memory consumption as I wont be be calling backward():
            action, log_prob, value = self.ac.act(obs_t)
        return action.item(), log_prob.item(), value.item() #Converting to scalars before returning
    
    # Simply storing to buffer 
    def store(self, obs, action, reward, log_prob, value, done):
        self.buffer.store(obs, action, reward, log_prob, value, done)
    
    def update(self):
        '''
        Adding a roadmap to self for this function
        
        1. Retrieve the stored data from the buffer
        2. Compute advantages and returns
        3. Normalising advantages(NOT in the paper but standard practice)
        
        4. K-Epochs of minibatch updates (Algorithm 1):
            for each epoch:
                shuffle data
                for each minibatch:
                a) evaluate actions under CURRENT policy (get new log_probs, entropy, values)
                b) compute ratio r_t = exp(new_log_prob - old_log_prob)
                c) clipped surrogate loss L^CLIP (Equation 7, page 3)
                d) value loss L^VF (basically squared error)
                e) entropy bonus S
                f) combined loss (Eq 9): L^CLIP - c1*L^VF + c2*S
                g) backprop + gradient clip + optimizer step

        5. Clear buffer for next rollout
        '''
        obs, actions, rewards, old_log_probs, values, dones = self.buffer.get()
        
        #advantages, returns = compute_gae(rewards, values, dones, self.gamma, self.lam)
        #advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        # AI sas I shouldnt be normalizing on the CPU then moving to device as it is wastefull
            # For documentation: removed `advantages, returns = advantages.to(self.device), returns.to(self.device)` from here too
        advantages, returns = compute_gae(rewards, values, dones, self.gamma, self.lam)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)       
        
        n = len(obs)
        # Tracking variables, very usefull for debugging. Need to upddate in inner loop
        total_policy_loss, total_value_loss, total_entropy = 0, 0, 0
        total_approx_kl, total_clip_frac = 0, 0
        num_updates = 0
        
        for _ in range(self.epochs):
            indices = torch.randperm(n)
            for start in range(0, n, self.batch_size):
                idx = indices[start:start + self.batch_size]
                
                new_log_probs, entropy, new_values = self.ac.evaluate(obs[idx], actions[idx])
                log_ratio = new_log_probs - old_log_probs[idx]
                ratio = log_ratio.exp()
                
                # Note for future self
                # The KL approximation (line 211) is the second order Taylor expansion of KL divergence. Usefull for monitoring. NOT IN PAPER
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean()
                    clip_frac = ((ratio - 1).abs() > self.clip_eps).float().mean()
                
                
                surr1 = ratio * advantages[idx]
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages[idx]
                policy_loss = -torch.min(surr1, surr2).mean()
 
                # Works out the same as Eq 9 Page 5 as we already negated policy_loss and entropy_loss
                value_loss = ((new_values - returns[idx]) ** 2).mean()
                entropy_loss = -entropy.mean() # - is correct. WE WANT TO MAX Entropy. combined with +ent_coef*ent_loss means we are sbstracting entropy from loss to maximize it
                
                # We remembered to clip, amazing.
                loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss
                
                self.optimizer.zero_grad() # Clear gradients from last step
                loss.backward() # Compute gradients of loss w.r.t all parameters
                nn.utils.clip_grad_norm_(self.ac.parameters(), self.max_grad_norm) # STOP THE EXPLODING GRADIENTS!!!
                self.optimizer.step() # Update params (Something along the lines of teta <- teta - lr * grad)
                
                # Accum metrics, again note to self .item() extracts the Python scalar from a 0 dimensional tensor.
                # Will also leave slight mental notes on usecases for each
                total_policy_loss += policy_loss.item() # Is the surrogate objective improving?
                total_value_loss += value_loss.item() # Is the critic fitting returning well?
                total_entropy += entropy.mean().item() # Is the policy still exploring? (Decreasing entropy might mean premature convergence)
                total_approx_kl += approx_kl.item() # Is the policy drifting too fat? (High kl = increase the clip OR lower lr)
                total_clip_frac += clip_frac.item() # How often is clipping active? (0% means its useless, 100% means it updates too aggressive)
                num_updates += 1
        
        self.buffer.clear()
        
        # If we wanted to sanity check we could also include num_updates in the return dictionary
        return {
            'policy_loss': total_policy_loss / num_updates,
            'value_loss': total_value_loss / num_updates,
            'entropy': total_entropy / num_updates,
            'approx_kl': total_approx_kl / num_updates,
            'clip_fraction': total_clip_frac / num_updates
        }
   
    # 2 clean functions
    # The only note one could add is we could save/load total_timesteps or episode count to track progress/resume training if cut.
    # But for basic checkpointing this should work just fine
 
    def save(self, path):
        torch.save({
            'model_state_dict': self.ac.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)
    
    def load(self, path):
        checkpoint = torch.load(path)
        self.ac.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

def train_ppo(env, total_timesteps = 200000, rollout_steps=2048, log_dir='../logs'):
    '''
    Algo 1, Page 5
    1. Collect T timesteps (The Horizon)
    2. Update the policy
    3. Repeat until total_timesteps reached
    '''

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    
    run_name = f"ppo_scratch_{int(time.time())}"
    writer = SummaryWriter(f"{log_dir}/{run_name}")
    
    agent = PPO(obs_dim, act_dim)

    # Log hyperparameters, for tensor_logs
    # Dont do the same mistake as I did, meant to log the string literals but actual PPO uses defaults so either pass hyperparams explicitly to train() & PPO OR pull them directly from the agent
    writer.add_text('hyperparameters',
        f"lr={agent.optimizer.param_groups[0]['lr']}, gamma={agent.gamma}, "
        f"lam={agent.lam}, clip_eps={agent.clip_eps}, epochs={agent.epochs}, "
        f"batch_size={agent.batch_size}, ent_coef={agent.ent_coef}, vf_coef={agent.vf_coef}")
    
    obs, _ = env.reset()
    ep_reward, ep_len = 0, 0
    ep_rewards, ep_lengths = [], []
    timestep = 0
    iteration = 0 # Super usefull for logging the x axis
    start_time = time.time()

    while timestep < total_timesteps:
        for i in range(rollout_steps):
            action, log_prob, value = agent.select_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.store(obs, action, reward, log_prob, value, done)
            
            obs = next_obs
            ep_reward += reward
            ep_len += 1
            timestep += 1
            
            if done:
                ep_rewards.append(ep_reward)
                ep_lengths.append(ep_len)
                writer.add_scalar('rollout/ep_rew', ep_reward, timestep)
                writer.add_scalar('rollout/ep_len', ep_len, timestep)
                obs, _ = env.reset()
                ep_reward, ep_len = 0, 0
            
            #if timestep >= total_timesteps:
            #    break
            # AI said this break is kind of ok but then agent.update() runs with a partial buffer.
            
            if timestep >= total_timesteps:
                break

        # Now AI says this is redundant and too safe, but its ok. After all the limiting factor is the enviroment
        if len(agent.buffer.obs) == rollout_steps:  # Only update with full buffer
            stats = agent.update()
            iteration += 1
            
        # The logging was outside the loop before the initial AI fix.
        # Then realised it would crash if stats didnt exist

            # Log training stats
            writer.add_scalar('train/policy_loss', stats['policy_loss'], timestep)
            writer.add_scalar('train/value_loss', stats['value_loss'], timestep)
            writer.add_scalar('train/entropy', stats['entropy'], timestep)
            writer.add_scalar('train/approx_kl', stats['approx_kl'], timestep)
            writer.add_scalar('train/clip_fraction', stats['clip_fraction'], timestep)
            
            elapsed = time.time() - start_time
            fps = timestep / elapsed
            writer.add_scalar('time/fps', fps, timestep)
       
        # Same issue as before, stats might not exist 
            # Rollout stats (averaged over recent episodes)
            if ep_rewards:
                recent_rews = ep_rewards[-10:]
                recent_lens = ep_lengths[-10:]
                writer.add_scalar('rollout/ep_rew_mean', np.mean(recent_rews), timestep)
                writer.add_scalar('rollout/ep_len_mean', np.mean(recent_lens), timestep)
                
                print(f"[{timestep:>7}/{total_timesteps}] "
                      f"ep_rew={np.mean(recent_rews):>6.1f} | "
                      f"ep_len={np.mean(recent_lens):>5.1f} | "
                      f"policy_loss={stats['policy_loss']:>7.4f} | "
                      f"value_loss={stats['value_loss']:>7.2f} | "
                      f"entropy={stats['entropy']:>5.3f} | "
                      f"kl={stats['approx_kl']:>.4f} | "
                      f"clip={stats['clip_fraction']:>.3f} | "
                      f"fps={fps:>4.1f}")
    
    writer.close()
    return agent

# Litteraly SB3 PPO.py test code
if __name__ == '__main__':
    import os
    import gymnasium as gym
    import gym4real
    from gym4real.envs.wds.utils import parameter_generator

    if os.path.exists("gym4ReaL"):
        os.chdir("gym4ReaL")
    
    params = parameter_generator(
        hydraulic_step=3600,
        duration=604800,
        seed=42,
        world_options="gym4real/envs/wds/world_anytown.yaml"
    )
    
    env = gym.make("gym4real/wds-v0", settings=params)
    
    agent = train_ppo(env, total_timesteps=200000, log_dir="../logs")
    agent.save("../models/ppo_scratch.pt")
    print(f"\nTraining complete. Model saved to ../models/ppo_scratch.pt")
    print(f"View logs: tensorboard --logdir=../logs")
