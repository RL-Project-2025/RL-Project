import torch
import torch.nn as nn
from torch.distributions import Categorical


class ActorCriticNet(nn.Module):
    """
    Shared-backbone Actor Critic network for discrete action spaces.
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()

        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Policy head (actor)
        self.policy_head = nn.Linear(hidden_dim, action_dim)

        # Value head (critic)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs):
        """
        Args:
            obs: Tensor of shape (batch_size, obs_dim)
        Returns:
            dist: Categorical distribution over actions
            value: Tensor of shape (batch_size,)
        """
        features = self.shared(obs)

        logits = self.policy_head(features)
        value = self.value_head(features).squeeze(-1)

        dist = Categorical(logits=logits)

        return dist, value
