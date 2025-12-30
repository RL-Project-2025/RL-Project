# algorithms/a2c_rollout.py
import torch
import torch.nn.functional as F


class A2CRollout:
    """
    A2C with truncated n-step rollouts (SB3-style).

    Key choices:
    - Collect n_steps transitions (a rollout)
    - Bootstrap the return using V(s_{t+n}) if rollout doesn't end in terminal
    - Compute advantages = returns - values
    - Optionally normalize advantages (matches SB3 trick)
    """

    def __init__(
        self,
        env,
        network,
        optimizer,
        gamma: float = 0.99,
        n_steps: int = 5,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,  # SB3 clips gradients
        device: str = "cpu",
        normalize_advantage: bool = True,
    ):
        self.env = env
        self.network = network.to(device)
        self.optimizer = optimizer

        self.gamma = gamma
        self.n_steps = n_steps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.device = device
        self.normalize_advantage = normalize_advantage

        self.obs = self.env.reset()  # VecEnv returns batched obs

    def collect_rollout(self):
        """
        Collect n_steps transitions.
        Works with DummyVecEnv (single env) but keeps VecEnv batching.
        """
        log_probs = []
        values = []
        rewards = []
        entropies = []
        dones = []

        for _ in range(self.n_steps):
            obs_tensor = torch.tensor(self.obs, dtype=torch.float32, device=self.device)

            dist, value = self.network(obs_tensor)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()

            next_obs, reward, done, _ = self.env.step(action.cpu().numpy())

            # reward/done are vec-shaped; keep tensors on device
            reward_t = torch.tensor(reward, dtype=torch.float32, device=self.device).squeeze(-1)
            done_t = torch.tensor(done, dtype=torch.float32, device=self.device).squeeze(-1)

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward_t)
            entropies.append(entropy)
            dones.append(done_t)

            self.obs = next_obs

        # Bootstrap value from last observation
        with torch.no_grad():
            obs_tensor = torch.tensor(self.obs, dtype=torch.float32, device=self.device)
            _, last_value = self.network(obs_tensor)

        return log_probs, values, rewards, entropies, dones, last_value

    def compute_returns(self, rewards, dones, last_value):
        """
        n-step bootstrapped returns computed backwards:
            R_t = r_t + gamma * (1 - done_t) * R_{t+1}
        with initial R_{t+n} = last_value.
        """
        returns = []
        R = last_value  # shape: (n_envs,)
        for r, d in zip(reversed(rewards), reversed(dones)):
            R = r + self.gamma * (1.0 - d) * R
            returns.insert(0, R)
        return torch.stack(returns)  # shape: (n_steps, n_envs)

    def update(self, log_probs, values, rewards, entropies, dones, last_value):
        """
        One update per rollout.
        """
        returns = self.compute_returns(rewards, dones, last_value).detach()
        values = torch.stack(values)        # (n_steps, n_envs)
        log_probs = torch.stack(log_probs)  # (n_steps, n_envs)
        entropies = torch.stack(entropies)  # (n_steps, n_envs)

        advantages = returns - values

        if self.normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        policy_loss = -(log_probs * advantages.detach()).mean()
        value_loss = F.mse_loss(values, returns)
        entropy_loss = -entropies.mean()

        total_loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return {
            "total_loss": float(total_loss.item()),
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float(entropies.mean().item()),
        }
