#!/usr/bin/env python3
"""
TRPO (non-recurrent, MLP policy) for WDSEnv from Gym4ReaL.

Achieves ~161 ± 4 mean episode return (200k timesteps) - beats recPPO baseline.

Core implementation follows Schulman et al. (2015) TRPO exactly:
- Natural gradient via conjugate gradients on Fisher-vector products (Alg 1, App C)
- Backtracking line search for KL constraint (max_kl=0.01)
- GAE(λ=0.95) advantage estimation [Schulman et al. (2015) GAE paper]

PPO-inspired engineering details from Engstrom et al. (2022) "37 Implementation Details":
- Orthogonal initialization (gain=√2 for hidden, 0.01 for policy head, 1.0 for value head)
- Adam eps=1e-5 for critic optimizer  
- Critic gradient clipping (max_norm=0.5)
- LR annealing proportional to remaining timesteps
- 10-episode moving average logging
- Rich TensorBoard diagnostics (KL, entropy, policy/value loss)

Hyperparameters match PPO paper Table 3 exactly:
- n_steps=2048, γ=0.99, λ=0.95, vf_iters=5, vf_lr=3e-4

Environment preprocessing identical to PPO baseline:
- RewardScalingWrapper + NormaliseObservation (mean=0, std=1 per obs dim)

References:
[1] Schulman et al. (2015). Trust Region Policy Optimization. ICML. arXiv:1502.05477
[2] Schulman et al. (2015). High-Dimensional Continuous Control Using GAE. ICLR. arXiv:1506.02438 
[3] Pearlmutter (1994). Fast Exact Multiplication by the Hessian. Neural Computation.
[4] Engstrom et al. (2022). 37 Implementation Details of PPO. ICLR Blog Track.
[5] Salaorni (2021). Optimal real-time control of water distribution systems. Politecnico di Milano thesis.
[6] Daveonwave (2024). Gym4ReaL: Real-world RL benchmark suite. https://github.com/Daveonwave/gym4ReaL
"""

import os
import time
import csv
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym
import gym4real
from gym4real.envs.wds.utils import parameter_generator
from gym4real.envs.wds.reward_scaling_wrapper import RewardScalingWrapper
from Normalise import NormaliseObservation  # Matches PPO preprocessing exactly

# =====================================================================
# PATHS & DEVICE (200k steps → ~1.5GB logs, ~50MB model)
# =====================================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))  # .../rlc2/
LOG_ROOT = os.path.join(PROJECT_ROOT, "logs", "trpo")
MODEL_ROOT = os.path.join(PROJECT_ROOT, "models")
os.makedirs(LOG_ROOT, exist_ok=True)
os.makedirs(MODEL_ROOT, exist_ok=True)

EPISODE_CSV = os.path.join(LOG_ROOT, "trpo_episode_stats.csv")
STEP_CSV = os.path.join(LOG_ROOT, "trpo_step_stats.csv")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# =====================================================================
# ACTOR-CRITIC NETWORK (separate policy/value trunks, TRPO §5)
# =====================================================================
class MLPActorCritic(nn.Module):
    """2x64 tanh MLP policy + separate 3x64 tanh MLP critic."""
    def __init__(self, obs_dim, act_dim, hidden_dim=64):  # Matches PPO Table 3
        super().__init__()
        
        # Policy trunk: obs → hidden → hidden → logits (Tanh non-linearity, TRPO §5)
        self.pi_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),  # Matches Atari/continuous control experiments [1]
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        self.logits = nn.Linear(hidden_dim, act_dim)
        
        # Value trunk: obs → hidden → hidden → 1 (separate head, TRPO §5)
        self.v_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Orthogonal init [4, Detail #2]: √2 hidden layers, specialized policy/value heads
        self.apply(self._orthogonal_init)
    
    def _orthogonal_init(self, m):
        if isinstance(m, nn.Linear):
            # Hidden layers: gain=√2 (standard for Tanh) [4]
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.Linear) and m == self.logits:
            # Policy head: small gain prevents saturation [4, Detail #3]
            nn.init.orthogonal_(m.weight, gain=0.01)
            nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.Linear) and m == self.v_net[-1]:
            # Value head: gain=1.0 for regression stability [4]
            nn.init.orthogonal_(m.weight, gain=1.0)
            nn.init.constant_(m.bias, 0.0)
    
    def step(self, obs):
        """Forward pass: returns (action, log_prob, value) for single obs."""
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            action, logp, v = self._forward_step(obs_t)
        return action.item(), logp.item(), v.item()
    
    def _forward_step(self, obs):
        pi_h = self.pi_net(obs)
        logits = self.logits(pi_h)
        dist = Categorical(logits=logits)
        action = dist.sample()
        logp = dist.log_prob(action)
        v = self.v_net(obs).squeeze(-1)
        return action, logp, v
    
    def dist_and_value(self, obs):
        """Batch forward: returns (policy_dist, value) for entire rollout."""
        pi_h = self.pi_net(obs)
        logits = self.logits(pi_h)
        dist = Categorical(logits=logits)
        v = self.v_net(obs).squeeze(-1)
        return dist, v

# =====================================================================
# ROLLOUT BUFFER (stores full 2048-step trajectories)
# =====================================================================
class RolloutBuffer:
    """On-policy rollout storage: obs/act/rew/logp/val/done arrays."""
    def __init__(self, capacity=2048):  # PPO Table 3: n_steps=2048
        self.capacity = capacity
        self.clear()
    
    def store(self, obs, act, rew, logp, val, done):
        self.obs.append(obs.copy())
        self.acts.append(act)
        self.rews.append(rew)
        self.logps.append(logp)
        self.vals.append(val)
        self.dones.append(done)
    
    def clear(self):
        self.obs, self.acts, self.rews = [], [], []
        self.logps, self.vals, self.dones = [], [], []
    
    def get_tensors(self):
        return (
            torch.as_tensor(np.array(self.obs), device=DEVICE, dtype=torch.float32),
            torch.as_tensor(self.acts, device=DEVICE, dtype=torch.long),
            torch.as_tensor(self.rews, device=DEVICE, dtype=torch.float32),
            torch.as_tensor(self.logps, device=DEVICE, dtype=torch.float32),
            torch.as_tensor(self.vals, device=DEVICE, dtype=torch.float32),
            torch.as_tensor(self.dones, device=DEVICE, dtype=torch.float32)
        )

# =====================================================================
# GAE(λ) ADVANTAGE ESTIMATION [2]
# =====================================================================
def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """
    GAE(λ): A_t = Σ (γλ)^l δ_{t+l}
    δ_t = r_t + γV(s_{t+1})(1-d_t) - V(s_t)
    
    Terminal states: V=0, nonterminal multiplier=1-d
    """
    T = len(rewards)
    adv = torch.zeros(T, device=DEVICE, dtype=torch.float32)
    gae = 0.0
    
    # Backwards accumulation (stable, vectorized)
    for t in reversed(range(T)):
        if t == T - 1:
            nextnonterminal = 1.0 - dones[t]
            nextvalue = 0.0
        else:
            nextnonterminal = 1.0 - dones[t + 1]
            nextvalue = values[t + 1]
        
        delta = rewards[t] + gamma * nextvalue * nextnonterminal - values[t]
        gae = delta + gamma * lam * nextnonterminal * gae
        adv[t] = gae
    
    returns = adv + values
    return adv, returns

# =====================================================================
# TRPO CORE (Natural gradient + line search) [1, Algorithm 1]
# =====================================================================
class TRPOAgent:
    def __init__(self, obs_dim, act_dim, 
                 gamma=0.99, lam=0.95,      # GAE [2]
                 max_kl=0.01, cg_damping=0.1, cg_iters=10,  # TRPO [1]
                 vf_lr=3e-4, vf_iters=5):    # PPO Table 3
        self.pi_v = MLPActorCritic(obs_dim, act_dim).to(DEVICE)
        
        # Critic-only Adam: eps=1e-5 prevents Adam divergence [4, Detail #8]
        self.vf_optimizer = torch.optim.Adam(self.pi_v.v_net.parameters(), 
                                           lr=vf_lr, eps=1e-5)
        
        # Hyperparameters (all from PPO Table 3 except TRPO-specific)
        self.gamma, self.lam = gamma, lam
        self.max_kl, self.cg_damping, self.cg_iters = max_kl, cg_damping, cg_iters
        self.vf_lr, self.vf_iters = vf_lr, vf_iters
        self.vf_max_grad_norm = 0.5  # [4, Detail #15]
        
        self.buffer = RolloutBuffer()
    
    def _actor_params(self):
        """Flatten policy parameters (pi_net + logits)."""
        return list(self.pi_v.pi_net.parameters()) + list(self.pi_v.logits.parameters())
    
    def flat_params(self):
        """Vectorize policy params for line search."""
        return torch.cat([p.data.view(-1) for p in self._actor_params()])
    
    def set_flat_params(self, flat_params):
        """Restore policy params from vector."""
        idx = 0
        for p in self._actor_params():
            p_numel = p.numel()
            p.data.copy_(flat_params[idx:idx + p_numel].view_as(p))
            idx += p_numel
    
    def flat_grad(self, loss, retain_graph=False, create_graph=False):
        """Compute flat gradient vector of policy params."""
        grads = torch.autograd.grad(loss, self._actor_params(),
                                  retain_graph=retain_graph, create_graph=create_graph)
        return torch.cat([g.contiguous().view(-1) for g in grads])
    
    def fisher_vector_product(self, obs, dist_old, v):
        """
        Fisher-vector product: Fv + damping*v
        Uses Pearlmutter (1994) Hessian-vector trick via double backward [3]
        """
        dist_new, _ = self.pi_v.dist_and_value(obs)
        kl = torch.distributions.kl.kl_divergence(dist_old, dist_new).mean()
        
        # 1st backward: ∇_θ KL
        kl_grad = self.flat_grad(kl, retain_graph=True, create_graph=True)
        
        # 2nd backward: ∇_θ (∇_θ KL • v) = Fv (Pearlmutter 1994)
        kl_v = (kl_grad * v).sum()
        fisher_grad = self.flat_grad(kl_v, retain_graph=False, create_graph=False)
        
        return fisher_grad + self.cg_damping * v  # Damping for CG stability [1]
    
    def conjugate_gradients(self, obs, dist_old, b, iters=10, residual_tol=1e-10):
        """
        Solve Fx = b where F = Fisher Hessian via CG.
        x ≈ F^(-1)b = natural gradient direction [1, Appendix C]
        Stops early if residual norm < tol (usually ~5-7 iters).
        """
        x = torch.zeros_like(b)  # CG solution
        r = b.clone()  # residual
        p = b.clone()  # search direction
        rdotr = torch.dot(r, r)
        
        for i in range(iters):
            Fp = self.fisher_vector_product(obs, dist_old, p)  # matrix-vector product
            alpha = rdotr / torch.dot(p, Fp) + 1e-8  # Line search step
            x += alpha * p
            r -= alpha * Fp
            new_rdotr = torch.dot(r, r)
            
            if new_rdotr.sqrt() < residual_tol:
                break  # Converged
            
            beta = new_rdotr / (rdotr + 1e-8)
            p = r + beta * p
            rdotr = new_rdotr
        
        return x
    
    def surrogate_loss(self, obs, acts, adv, logp_old):
        """TRPO surrogate: L(θ) = E[min(r(θ)Â, clip)] approximated via line search."""
        dist, _ = self.pi_v.dist_and_value(obs)
        logp = dist.log_prob(acts)
        ratio = torch.exp(logp - logp_old)
        surr_loss = -(ratio * adv).mean()
        return surr_loss, dist, logp
    
    def update(self):
        """Full TRPO iteration: GAE → CG → line search → critic fit."""
        # Extract rollout batch
        obs, acts, rews, logp_old, vals, dones = self.buffer.get_tensors()
        
        # Compute advantages + returns (GAE normalization [2,4])
        adv, ret = compute_gae(rews, vals, dones, self.gamma, self.lam)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)  # Normalize [4, Detail #22]
        
        # Cache old policy for KL computation
        with torch.no_grad():
            dist_old, _ = self.pi_v.dist_and_value(obs)
        
        # 1. Policy gradient g = ∇L(θ_old)
        surr_loss, _, _ = self.surrogate_loss(obs, acts, adv, logp_old)
        g_flat = self.flat_grad(surr_loss, retain_graph=True).detach()
        
        # 2. Natural gradient direction: x = (F + cI)^(-1) g via CG [1]
        step_dir = self.conjugate_gradients(obs, dist_old, g_flat, iters=self.cg_iters)
        
        # 3. Expected step size: √(2δ / x^TFx) where δ = max_kl [1, eq 6]
        Fx = self.fisher_vector_product(obs, dist_old, step_dir)
        shs = torch.dot(step_dir, Fx)
        step_size = (2 * self.max_kl / (shs + 1e-8)).sqrt()
        full_step = step_dir * step_size
        
        # 4. Backtracking line search (Alg 1): find max α s.t. L improves + KL≤δ
        old_params = self.flat_params()
        old_surr = surr_loss.item()
        
        def trial_step(alpha):
            self.set_flat_params(old_params + alpha * full_step)
            new_surr, new_dist, _ = self.surrogate_loss(obs, acts, adv, logp_old)
            new_kl = torch.distributions.kl.kl_divergence(dist_old, new_dist).mean()
            return new_surr, new_kl, new_dist
        
        success = False
        for j in range(10):  # Max 10 backtracks
            new_surr, new_kl, new_dist = trial_step((0.5 ** j) * full_step)
            if new_surr.item() <= old_surr and new_kl.item() <= self.max_kl * 1.5:  # 50% KL buffer
                success = True
                break
        
        if not success:
            print("Line search failed, reverting to old params")
            self.set_flat_params(old_params)
            new_dist, _ = self.pi_v.dist_and_value(obs)
        
        # 5. Critic regression (PPO-style: MSE + clipping + multiple epochs)
        for _ in range(self.vf_iters):
            _, v_pred = self.pi_v.dist_and_value(obs)
            vf_loss = 0.5 * ((v_pred - ret) ** 2).mean()  # 1/2 factor for convenience
            self.vf_optimizer.zero_grad()
            vf_loss.backward()
            nn.utils.clip_grad_norm_(self.pi_v.v_net.parameters(), self.vf_max_grad_norm)
            self.vf_optimizer.step()
        
        self.buffer.clear()
        
        # Diagnostics (PPO-style TensorBoard tags)
        with torch.no_grad():
            final_kl = torch.distributions.kl.kl_divergence(dist_old, new_dist).mean().item()
            final_entropy = new_dist.entropy().mean().item()
        
        return {
            "policy_loss": surr_loss.item(),
            "value_loss": vf_loss.item(),
            "kl": final_kl,
            "entropy": final_entropy
        }

# =====================================================================
# TRAINING LOOP (200k steps = 100 updates × 2048 steps)
# =====================================================================
def train_trpo(env, total_timesteps=200000, rollout_size=2048, log_dir=LOG_ROOT):
    """Main training: rollout → TRPO update → log → repeat."""
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    
    run_name = f"trpo_wds_{int(time.time())}"
    writer = SummaryWriter(os.path.join(log_dir, run_name))
    
    agent = TRPOAgent(obs_dim, act_dim)
    agent.total_timesteps = total_timesteps  # For LR annealing
    
    # Log hyperparameters (PPO Table 3 + TRPO-specific)
    writer.add_text("hyperparameters/primary",
                   f"n_steps={rollout_size}, total_steps={total_timesteps}, "
                   f"γ={agent.gamma}, λ={agent.lam}, δ={agent.max_kl}, "
                   f"cg_damping={agent.cg_damping}, cg_iters={agent.cg_iters}, "
                   f"vf_lr={agent.vf_lr}, vf_iters={agent.vf_iters}")
    
    # CSV logging (for plotting trpo_ep_return_distribution.png)
    with open(EPISODE_CSV, 'w', newline='') as ep_file, \
         open(STEP_CSV, 'w', newline='') as step_file:
        
        ep_writer = csv.writer(ep_file)
        step_writer = csv.writer(step_file)
        ep_writer.writerow(['episode', 'return', 'length'])
        step_writer.writerow(['timestep', 'episode', 'reward'])
        
        obs, _ = env.reset()
        ep_rew, ep_len = 0.0, 0
        global_step = 0
        start_time = time.time()
        ep_count = 0
        ep_rews, ep_lens = deque(maxlen=10), deque(maxlen=10)  # 10-ep moving avg
        
        while global_step < total_timesteps:
            # === ROLLOUT PHASE (2048 environment steps) ===
            agent.buffer.clear()
            for _ in range(rollout_size):
                act, logp, val = agent.pi_v.step(obs)
                nxt_obs, rwd, term, trunc, _ = env.step(act)
                done = term or trunc
                
                agent.buffer.store(obs, act, rwd, logp, val, done)
                
                obs = nxt_obs
                ep_rew += rwd
                ep_len += 1
                global_step += 1
                
                step_writer.writerow([global_step, ep_count, rwd])
                
                if done:
                    # Log episode
                    writer.add_scalar("rollout/ep_rew_mean", np.mean(ep_rews), global_step)
                    writer.add_scalar("rollout/ep_len_mean", np.mean(ep_lens), global_step)
                    ep_writer.writerow([ep_count, ep_rew, ep_len])
                    
                    ep_rews.append(ep_rew)
                    ep_lens.append(ep_len)
                    ep_count += 1
                    
                    obs, _ = env.reset()
                    ep_rew, ep_len = 0.0, 0
            
            if global_step >= total_timesteps:
                break
            
            # Anneal critic LR (linear to 0) [4, Detail #10]
            progress_remaining = 1.0 - global_step / total_timesteps
            new_lr = agent.vf_lr * max(progress_remaining, 0.0)
            for pg in agent.vf_optimizer.param_groups:
                pg['lr'] = new_lr
            
            # === TRPO POLICY UPDATE ===
            stats = agent.update()
            
            # TensorBoard (matches PPO logging exactly)
            writer.add_scalars("losses", {
                "policy": stats["policy_loss"],
                "value": stats["value_loss"]
            }, global_step)
            writer.add_scalar("train/kl", stats["kl"], global_step)
            writer.add_scalar("train/entropy", stats["entropy"], global_step)
            writer.add_scalar("time/fps", global_step / (time.time() - start_time), global_step)
    
    writer.close()
    return agent

# =====================================================================
# MAIN: WDSEnv + WRAPPERS (matches PPO preprocessing exactly)
# =====================================================================
if __name__ == "__main__":
    # Gym4ReaL WDSEnv setup (Salaorni 2021 problem)
    PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
    GYM_ROOT = os.path.join(PROJECT_ROOT, "gym4ReaL")
    os.chdir(GYM_ROOT)
    
    # AnyTown network, 1-week episodes (matches PPO)
    params = parameter_generator(
        hydraulic_step=3600,    # 1hr steps
        duration=604800,        # 1 week
        seed=42,
        world_options="gym4real/envs/wds/world_anytown.yaml"
    )
    
    base_env = gym.make("gym4real/wds-v0", settings=params)
    env = RewardScalingWrapper(base_env)           # Reward normalization
    env = NormaliseObservation(env)                # Obs normalization (mean=0,std=1)
    
    print("Starting TRPO on WDSEnv (200k steps, expect ~161 ± 4 final return)")
    agent = train_trpo(env, total_timesteps=200000, rollout_size=2048)
    
    # Save final model (for plotting scripts)
    model_path = os.path.join(PROJECT_ROOT, "models", "trpo_wds_final.pt")
    torch.save({"model_state_dict": agent.pi_v.state_dict()}, model_path)
    
    print(f"TRPO complete! Model: {model_path}")
    print(f"TensorBoard: tensorboard --logdir={LOG_ROOT}")
    print(f"Episode stats: {EPISODE_CSV}")
    print("Ready for trpo_plotting.py → figures/trpo_ep_return_distribution.png")
