#!/usr/bin/env python3

# Credits to: https://arxiv.org/pdf/1502.05477

import numpy as np
import torch
import torch.nn as nn

# Sorry for anyone trying to make sense of the mathematical formulas in the form of comments.
# I couldnt copy and paste symbols like theta, sigma etc...

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

    pass

def flat_grad(loss, params, retain_graph=False):
    '''
    Compute grad of loss w.r.t. params, return 1D tensor.

    Why not loss.backward()? GREAT QUESTION!
        We need the gradient as a vector g for CG computation
        We often need to retain_graph = True for multiple grad calculations
    '''

    # 1. torch.autograd.grad(loss, params, retain_graph=retain_graph)
    # 2. Flatten and concatenate, same as flat_params
    # 3. Handle None grads (replace with zeros)

    pass

def fisher_vector_product(actor, obs, old_log_probs, vector, damping=0.1)
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

    # 5. Add damping: return hessian_vector_product + damping * vector

    pass 

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

    pass

def line_search(actor, get_loss, get_kl, old_params, step_dir, max_step, max_backtracks=10, accept_ratio=0.1, delta=0.01):
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

    pass

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

    def __init__(self, '''may god help us in this section'''):
        '''
        Hyperparams from paper, appendix E, table 2, P 15:
        '''

    def select_action(self, obs):
        # Copy this from PPO implementation
    
    def store(self, obs, action, reward, log_prob, value, done):
        # Copy this from PPO implementation
    
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

        pass

    def save(self, path):
        # From PPO
        pass

    def load(self, path):
        # From PPO
        pass

def train_trpo(env, total_timestep=200000, rollout_steps=2048, log_dir='../logs'):
     # From PPO too

    '''
    Only diffs:
        Create trpo agent rather than ppo
        Logging will include different metrics
        No epochs parameter
    '''
    pass
