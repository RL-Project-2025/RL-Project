import torch
import torch.nn.functional as F


class A2C:
    """
    Advantage Actor Critic (A2C) algorithm.

    This implementation uses:
    - GAE with λ=1, corresponding to Monte-Carlo advantage estimation.
    - Shared actor critic network
    - Entropy regularisation for exploration
    """

    def __init__(
        self,
        env,
        network,
        optimizer,
        gamma: float = 0.99,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        device: str = "cpu",
    ):
        self.env = env
        self.network = network.to(device)
        self.optimizer = optimizer

        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.device = device

    def run_episode(self):
        """
        Run a single on-policy episode and collect transitions.

        Returns:
            log_probs: list of log π(a_t | s_t)
            values: list of V(s_t)
            rewards: list of rewards
            entropies: list of policy entropies
        """
        state = self.env.reset()

        log_probs = []
        values = []
        rewards = []
        entropies = []

        done = False

        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32)

            dist, value = self.network(state_tensor)

            action = dist.sample()
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()

            next_state, reward, done, _ = self.env.step([action.item()])

            reward = reward[0]
            done = done[0]


            log_probs.append(log_prob)
            values.append(value)
            rewards.append(torch.tensor(reward, device=self.device))
            entropies.append(entropy)

            state = next_state

        return log_probs, values, rewards, entropies

    def compute_returns(self, rewards):
        """
        Compute Monte-Carlo discounted returns.

        Note:
            This corresponds to λ = 1 in Generalized Advantage Estimation (GAE),
            yielding unbiased but higher-variance advantage estimates.
            See:
                Schulman et al., 2015: "Generalized Advantage Estimation"
                Sutton & Barto, 2018: Reinforcement Learning: An Introduction
        """

        returns = []
        G = 0.0

        for reward in reversed(rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)

        return torch.stack(returns)

    def update(self, log_probs, values, rewards, entropies):
        """
        Perform a single A2C update step.
        """
        returns = self.compute_returns(rewards).detach()
        values = torch.cat(values)
        log_probs = torch.cat(log_probs)
        entropies = torch.cat(entropies)

        # Advantage estimation
        advantages = returns - values

        # Advantage normalisation - idea from stable baselines3
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)


        # Actor loss
        policy_loss = -(log_probs * advantages.detach()).mean()

        # Critic loss (value regression)
        value_loss = F.mse_loss(values, returns)

        # Entropy bonus
        entropy_loss = -entropies.mean()

        total_loss = (
            policy_loss
            + self.value_coef * value_loss
            + self.entropy_coef * entropy_loss
        )

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return {
            "total_loss": total_loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropies.mean().item(),
        }
