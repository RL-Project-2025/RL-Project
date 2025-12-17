#!/usr/bin/env python3

# Credits to: https://arxiv.org/pdf/1502.05477

import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

# Sorry for anyone trying to make sense of the mathematical formulas in the form of comments.
# I couldnt copy and paste symbols like theta, sigma etc...

class ActorCritic(nn.Module):
    '''
    Seperate actor and critic networks for TRPO.
    '''

    def __init__(self, obs_dim, act_dim, hidden=64):
        super().__init__()
        # Actor network (policy) - TRPO aims to update this via natural gradient
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, act_dim)
        )
        # Critic network (value function), updates via standard SDG
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1)
        )
   
    # Returns both policy dist params and value estimate 
    def forward(self, x):
        return self.actor(x), self.critic(x)
   
    # SAMPLE action from policy for enviroment interaction
    # Also return log_prob and value 
    def act(self, obs):
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), value.squeeze(-1)
    
    # Compute log_prob, entropy, V(s)
    def evaluate(self, obs, actions):
        logits, values = self.forward(obs)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), values.squeeze(-1)
    # Return the policy distribution for the KL computation    
    def get_distribution(self, obs):
        logits = self.actor(obs)
        return Categorical(logits=logits)

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

    # Algo 1, Page 3 "Solve the constrained optimization problem" once per iter!
    def get(self):
        return (
            torch.tensor(np.array(self.obs), dtype=torch.float32),
            torch.tensor(self.actions, dtype=torch.long),
            torch.tensor(self.rewards, dtype=torch.float32),
            torch.tensor(self.log_probs, dtype=torch.float32),
            torch.tensor(self.values, dtype=torch.float32),
            torch.tensor(self.dones, dtype=torch.float32)
        )

def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """Identical to PPO, reference GAE from Schulman et al. 2015."""
    advantages = []
    gae = 0
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = last_value * (1 - dones[t]) # This time, bootstrap if not done
        else:
            next_value = values[t + 1] * (1 - dones[t])
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages.append(gae)
    advantages = advantages[::-1]
    advantages = torch.tensor(advantages, dtype=torch.float32)
    returns = advantages + values
    return advantages, returns

def flat_params(model):
    '''
    Extract all parameters from model into a single 1D tensor.

    As line search operates on theta as a single vector, rather than a collection of weights/biases.
    
    1. Save theta_old before update
    2. Compute theta_new = theta_old + alpha * step_dir
    3. Restore theta_old if line search fails!
    '''
    
    # 1. Iterate over model.parameters()
    # 2. Flatten each param with .view(-1)
    # 3. Concatenate into a single 1D tensor!


    # p.data.view(-1) detaches from graph, which is aimed because we are saving/restoring
    # The gradients we get back are attached to the graph with create_graph=True! Which is what enables the fisher vector product!
    return torch.cat([p.data.view(-1) for p in model.parameters()])

def set_flat_params(model, flat_params):
    '''
    Set model parameters from a flat 1D tensor.
    Inverse of flat_params().
    '''
    
    #Surely theres a more ptimal way of doing this
    idx = 0
    for p in model.parameters():
        numel = p.numel()
        p.data.copy_(flat_params[idx:idx + numel].view(p.shape))
        idx += numel

def flat_grad(loss, params, retain_graph=False, create_graph=False):
    '''
    Compute grad of loss w.r.t. params, return 1D tensor.

    Why not loss.backward()? GREAT QUESTION!
        We need the gradient as a vector g for CG computation
        We often need to retain_graph = True for multiple grad calculations
    '''

    # 1. torch.autograd.grad(loss, params, retain_graph=retain_graph)
    # 2. Flatten and concatenate, same as flat_params
    # 3. Handle None grads (replace with zeros)

    grads = torch.autograd.grad(loss, params, retain_graph=retain_graph, create_graph=create_graph)
    
    flat_grads = []
    for g, p in zip(grads, params):
        if g is not None:
            flat_grads.append(g.view(-1))
        else:
            # param didnt contribute to loss! grad is zero!
            flat_grads.append(torch.zeros(p.numel()))
    
    return torch.cat(flat_grads)

def compute_kl(actor, obs, old_dist):
    '''
    Compute mean KL divergence between old and current policy.
    
    Eq 12, Page 4:
        D^ρ_KL(θ_old, θ) = E_s~ρ [D_KL(π_θold(·|s) || π_θ(·|s))]
    
    For categorical distrobutions:
        KL(p||q) = Σ p(x) * log(p(x)/q(x))
    '''

    # Im not flagging obs to have requires_grad = False, yes in theory it should let the gradient flow through policy params only but I dont believe such lies told by the goverment. 
    new_dist = actor.get_Distribution(obs)

    # kl_divergence expects (old, new) and computs KL(old || new)
    kl = torch.distributions.kl_divergence(old_dist, new_dist)
    return kl.mean()

    

def fisher_vector_product(actor, obs, old_dist, vector, damping=0.1):
    '''
    Compute F @ vector, where F is the FISHER INFORMATION MATRIX

    According to the paper, this is the core computation of TRPO. We never form F explicitly  sa its |theta|.|theta|( too large ). 
    Instead we compute Fv for arbitrary v
    
    From TRPO paper, Appendix C.1, Page 14:
        F is the Hessian of the KL divergence: F_ij - sigma^2 D_{KL} / sigma.theta_i.sigma.thetha_j
        We compute this via two backpropagations (gradient of gradient)

    The trick (Pearlmutter 1994, "Fast exact multiplication by the Hessian"):
        1. Compute g = nabla_theta KL
        2. Compute dot product: g . v (a scalar)
        3. Compute nabla_theta (g . ) = Hv (the Hessian-vector product)

    Damping(not in original paper,) recommended by a colleague):
        1. Return (F + damping * I)
        2. Ensures numerical stability, makes f positive definite
        3. Equivalent to adding L2 regularization
    '''

    # 1. Compute KL div between pol and old_pol
    #   KL = E[log pi_theta(a|s) - log pi_theta_old(a|s)]  (using old_log_probs)
    #   Or use the analytic KL for your distribution (Categorical)    

    # 2. Compute gradient of KL w.r.t. policy params
    #   kl_grad = flat_grad(kl, actor.parameters(), retain_graph=True)

    # 3. Compute dot prod(ofc): kl_grad . vector

    # 4. Compute gradient of the dot prod ( aims to give Hessian vector product )
    #   This requires the retain_graph=True in step 2!

    # Credits to: https://arxiv.org/pdf/1206.6464 The paper cites it but still.
    # 5. Add damping: return hessian_vector_product + damping * vector  

    kl = compute_kl(actor, obs, old_dist)

    params = list(actor.parameters())
    kl_grad = flat_grad(kl, params, retain_graph=True, create_graph=True)

    kl_grad_v = (kl_grad * vector).sum()
    
    # Yes we are calling this function multiple times on the same graph, retain_graph=True isnt redundant. 
    hvp = flat_grad(kl_grad_v, params, retain_graph=True)
    
    # Damping: (F + λI)v for numerical stability
    # Aims to ensure positive definiteness when F is near singular!
    return hvp + damping * vector
    
def conjugate_gradient(fvp_fn, g, n_iters=10, residual_tol=1e-10):
    '''
    Solve Fx = g for x, where F is positive definite.
    
    TRPO paper, Appendix C, Page 13:
    "We use the conjugate gradient algorithm followed by a line search"

    The algorithm(standard CG, not specific to the TRPO paper):
        Start with x=0
        Iterate and improve x using only Fv products (fisher_vector_product)
        Converges in at most abs(theta) iters, but 10 is most of the time enough

    Reference: Couldnt find anything better than wikipedia, but tbf it was more than enough.
    https://en.wikipedia.org/wiki/Conjugate_gradient_method
    '''

    x = torch.zeros_like(g)
    r = g.clone()
    p = r.clone()
    r_dot_r = torch.dor(r, r)

    for i in range(n_iters):
        Fp = fvp_fn(p)
        alpha = r_dot_r / (torch.dot(p, Fp) + 1e-8)
        x = x + alpha * p
        r = r - alpha * Fp

        r_dot_r_new = torch.dot(r, r)
        if r_dot_r_new < residual_tol:
            break

        beta = r_dot_r_new / r_dot_r
        p = r + beta * p              
        r_dot_r = r_dot_r_new

    return x

def line_search(actor, get_loss, get_kl, old_params, expected_improve, step_dir, max_step, max_backtracks=10, accept_ratio=0.1, delta=0.01):
    '''
    Backtracking line search to find step size constraints.
    
    From TRPO paper, Appendix C, Page 14:
    "We perform the line search on the objective L_Thetaold(theta) - X[D_KL <= delta]"

    Starting with max step size beta = .sqrt(2.delta / s^T . F . s), we shrink until:
        Actual loss improves(not just surrogate!)
        KL constraint is satisfied: D_KL(theta_old || theta_new) <= delta

    The "accept_ratio" (also called c1 or Armijo constant):
        Requires L_new > L_old + c * expected_improvement
        Ensures we get meaningful improvement, not just numerical noise
        Typically .1 for TRPO
    '''

    old_loss = get_loss().item()
    #expected_improve = (flat_grad(get_loss(), actor.parameters()) * step_dir).sum().item()
    # Computed get_loss and its gradient, but I already have the gradient from before CG!
    
    expected_improve = (policy_grad * step_dir).sum().item()

    for i in range(max_backtracks):
        step_frac = 0.5 ** i
        # Idk at this point, max_step scalar is  beta=(2.delta/s^T.F.s)^(1/2) right???, so step dir should aread be a step direction from CG!
        #new_params = old_params + step_frac * max_step * step_dir
        new_params = old_params + step_frac * step_dir #Step frac meant to account for reduced step size
        set_flat_params(actor, new_params)
        
        new_loss = get_loss().item()
        new_kl = get_kl().item()
        
        actual_improve = new_loss - old_loss
        # Looking back >= seems to be only correct if get_loss returns the surrogate objective. Just make sure that is the case please. 
        if new_kl <= delta and actual_improve >= accept_ratio * step_frac * expected_improve:
            return step_frac * max_step
    
    set_flat_params(actor, old_params)
    return 0.0
    
class TRPO:
    '''
    Finally TRPO!

    From paper, Section 6, P 5:

    1. Collect rollout!
    2. Compute GAE
    3. Compute policy gradient g
    4. Compute search directions s
    5. Compute max step beta
    6. Line search to find actual step
    7. Update value function (can/should use multiple SGD steps)
    '''

    def __init__(self, obs_dim, act_dim, gamma=.99, lam=.95, delta=.01, cg_iters=10, damping=0.1, vf_lr=1e-3, vf_epochs=5, max_backtracks=10, accept_ratio=0.1):
        '''
        Hyperparams from paper, appendix E, table 2, P 15:
        '''
        
        self.gamma = gamma
        self.lam = lam
        self.delta = delta  # KL constraint threshold 
        self.cg_iters = cg_iters
        self.damping = damping
        self.vf_epochs = vf_epochs
        self.max_backtracks = max_backtracks
        self.accept_ratio = accept_ratio
        
        self.ac = ActorCritic(obs_dim, act_dim)

        self.vf_optimizer = optim.Adam(self.ac.critic.parameters(), lr=vf_lr)

        self.buffer = RolloutBuffer()
        
    def select_action(self, obs):
        """Identical to PPO."""
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action, log_prob, value = self.ac.act(obs_t)
        return action.item(), log_prob.item(), value.item()
     
    def store(self, obs, action, reward, log_prob, value, done):
        """Identical to PPO."""
        self.buffer.store(obs, action, reward, log_prob, value, done)
     
    def update(self):
        '''
        1. Get rollout data
        2. Compute GAE
        3. Normalize advantages

        # Pol update
        4. Compute surrogate loss L
        5. Compute pol grad g
        6. Build FVP function
        7. Compute search direction via CG
        8. Compute max step size
        9. Line search to find valid step
        10. Apply the update

        11. Multiple epochs of value functions updates
        
        12. Clear buffer        
        '''
        
        obs, actions, rewards, old_log_probs, values, dones = self.buffer.get()
        advantages, returns = compute_gae(rewards, values, dones, self.gamma, self.lam)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        with torch.no_grad():
            # old_dist.logits is already detached due to no_grad context
            old_dist = self.ac.get_distribution(obs)

        # Bootstraping value for GAE, for truncated episodes
        with torch.no_grad():
            _, _, last_value = self.ac.act(obs[-1].unsqueeze(0))
            advantages, returns = compute_gae(rewards, values, dones, last_value, self.gamma, self.lam)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        def get_surrogate_loss():
            new_log_probs, _, _ = self.ac.evaluate(obs, actions)
            ratio = torch.exp(new_log_probs - old_log_probs)
            return (ratio * advantages).mean()
        
        def get_kl():
            return compute_kl(self.ac.actor, obs, old_dist)

        loss = get_surrogate_loss()
        policy_params = list(self.ac.actor.parameters())
        g = flat_grad(loss, policy_params, retain_graph=True)
        
        def fvp_fn(v):
            return fisher_vector_product(self.ac.actor, obs, old_dist, v, self.damping)
        
        step_dir = conjugate_gradient(fvp_fn, g, n_iters=self.cg_iters)

        sFs = torch.dot(step_dir, fvp_fn(step_dir)) # Appendix C, Page 14, sFs is s^T A s where A is damped fisher matrix
        max_step = torch.sqrt(2 * self.delta / (sFs + 1e-8))

        old_params = flat_params(self.ac.actor)
    
        # Mismatch with line search signature, will fix if clarity is needed
        step_taken = line_search(actor=self.ac.actor, get_loss=get_surrogate_loss, get_kl=get_kl, old_params=old_params, step_dir=step_dir, max_step=max_step, max_backtracks=self.max_backtracks, accept_ratio=self.accept_ratio, delta=self.delta)
        
        total_vf_loss = 0
        for _ in range(self.vf_epochs):
            _, _, new_values = self.ac.evaluate(obs, actions)
            vf_loss = ((new_values - returns) ** 2).mean()
                
            self.vf_optimizer.zero_grad()
            vf_loss.backward()
            self.vf_optimizer.step()
            total_vf_loss += vf_loss.item()

        with torch.no_grad():
            final_kl = get_kl().item()
            final_loss = get_surrogate_loss().item()
        
        self.buffer.clear()
        
        return {
            'policy_loss': final_loss,
            'value_loss': total_vf_loss / self.vf_epochs,
            'kl': final_kl,
            'step_size': step_taken if isinstance(step_taken, float) else step_taken.item(),
            'max_step': max_step.item()
        }
    
    def save(self, path):
        torch.save({
            'actor_state_dict': self.ac.actor.state_dict(),
            'critic_state_dict': self.ac.critic.state_dict(),
            'vf_optimizer_state_dict': self.vf_optimizer.state_dict()
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.ac.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.ac.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.vf_optimizer.load_state_dict(checkpoint['vf_optimizer_state_dict'])

def train_trpo(env, total_timesteps=200000, rollout_steps=2048, log_dir='../logs'):
    '''
    Only diffs:
        Create trpo agent rather than ppo
        Logging will include different metrics
        No epochs parameter
    '''
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    
    run_name = f"trpo_scratch_{int(time.time())}"
    writer = SummaryWriter(f"{log_dir}/{run_name}")
    
    agent = TRPO(obs_dim, act_dim)
    
    writer.add_text('hyperparameters',
        f"gamma={agent.gamma}, lam={agent.lam}, delta={agent.delta}, "
        f"cg_iters={agent.cg_iters}, damping={agent.damping}, "
        f"vf_epochs={agent.vf_epochs}")
    
    obs, _ = env.reset()
    ep_reward, ep_len = 0, 0
    ep_rewards, ep_lengths = [], []
    timestep = 0
    iteration = 0
    start_time = time.time()
    
    while timestep < total_timesteps:
        for _ in range(rollout_steps):
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
            
            if timestep >= total_timesteps:
                break
        
        # This could cause issues if episode ends right before rollout_steps
        if len(agent.buffer.obs) == rollout_steps:
            stats = agent.update()
            iteration += 1
            
            writer.add_scalar('train/policy_loss', stats['policy_loss'], timestep)
            writer.add_scalar('train/value_loss', stats['value_loss'], timestep)
            writer.add_scalar('train/kl', stats['kl'], timestep)
            writer.add_scalar('train/step_size', stats['step_size'], timestep)
            writer.add_scalar('train/max_step', stats['max_step'], timestep)
           
            # For trpo specifically, not that I think we will be actively parameter tuning but here we go anyway.
            # writer.add_scalar('train/sFs', stats.get('sFs', 0), timestep)
            # writer.add_scalar('train/grad_norm', stats.get('grad_norm', 0), timestep)

            elapsed = time.time() - start_time
            fps = timestep / elapsed
            writer.add_scalar('time/fps', fps, timestep)
            
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
                      f"kl={stats['kl']:>.4f} | "
                      f"step={stats['step_size']:>.4f} | "
                      f"fps={fps:>4.1f}")
    
    writer.close()
    return agent

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
        world_options="gym4real/envs/wds/world_anytown.yaml",
    )

    params['demand_moving_average'] = False
    params['demand_exp_moving_average'] = True

    env = gym.make("gym4real/wds-v0", settings=params)

    agent = train_trpo(env, total_timesteps=200000, log_dir="../logs")
    agent.save("../models/trpo_scratch.pt")
    print(f"\nTraining complete. Model saved to ../models/trpo_scratch.pt")
    print(f"View logs: tensorboard --logdir=../logs")
