"""
GreenCloudRL - Neural Network Architectures
Actor, Critic, and PPO networks for hierarchical RL agents.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import List, Tuple, Optional
import numpy as np


class ActorNetwork(nn.Module):
    """
    Policy network for the low-level A2C agent.
    Maps state -> action probabilities with invalid action masking.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_sizes: List[int] = [256, 256, 128]):
        super().__init__()

        layers = []
        prev_dim = state_dim
        for h in hidden_sizes:
            layers.extend([nn.Linear(prev_dim, h), nn.ReLU(), nn.LayerNorm(h)])
            prev_dim = h

        self.feature_net = nn.Sequential(*layers)
        self.action_head = nn.Linear(prev_dim, action_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        # Smaller init for action head
        nn.init.orthogonal_(self.action_head.weight, gain=0.01)

    def forward(self, state: torch.Tensor, action_mask: Optional[torch.Tensor] = None) -> Categorical:
        """
        Forward pass with optional action masking.
        
        Args:
            state: Batch of state vectors [B, state_dim]
            action_mask: Binary mask [B, action_dim], 1=valid, 0=invalid
            
        Returns:
            Categorical distribution over actions
        """
        features = self.feature_net(state)
        logits = self.action_head(features)

        if action_mask is not None:
            # Set invalid actions to very negative logits
            logits = logits.masked_fill(action_mask == 0, -1e8)

        return Categorical(logits=logits)

    def get_action(self, state: torch.Tensor, action_mask: Optional[torch.Tensor] = None) -> Tuple[int, float]:
        """Sample action and return log probability."""
        dist = self.forward(state, action_mask)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob.item()


class CriticNetwork(nn.Module):
    """
    Value network for the low-level A2C agent.
    Maps state -> state value V(s).
    """

    def __init__(self, state_dim: int, hidden_sizes: List[int] = [256, 256, 128]):
        super().__init__()

        layers = []
        prev_dim = state_dim
        for h in hidden_sizes:
            layers.extend([nn.Linear(prev_dim, h), nn.ReLU(), nn.LayerNorm(h)])
            prev_dim = h

        self.feature_net = nn.Sequential(*layers)
        self.value_head = nn.Linear(prev_dim, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: [B, state_dim]
        Returns:
            State values [B, 1]
        """
        features = self.feature_net(state)
        return self.value_head(features)


class PPOActorCritic(nn.Module):
    """
    Combined Actor-Critic for PPO (used by high-level agent).
    Shared feature extraction with separate policy and value heads.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_sizes: List[int] = [128, 128]):
        super().__init__()

        layers = []
        prev_dim = state_dim
        for h in hidden_sizes:
            layers.extend([nn.Linear(prev_dim, h), nn.ReLU(), nn.LayerNorm(h)])
            prev_dim = h

        self.shared_net = nn.Sequential(*layers)
        self.policy_head = nn.Linear(prev_dim, action_dim)
        self.value_head = nn.Linear(prev_dim, 1)

        # Embedding output for communication with low-level agent
        self.embedding_head = nn.Linear(prev_dim, 8)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)

    def forward(self, state: torch.Tensor) -> Tuple[Categorical, torch.Tensor, torch.Tensor]:
        """
        Returns:
            policy distribution, state value, policy embedding
        """
        features = self.shared_net(state)
        logits = self.policy_head(features)
        value = self.value_head(features)
        embedding = torch.sigmoid(self.embedding_head(features))

        return Categorical(logits=logits), value, embedding

    def get_action_and_value(self, state: torch.Tensor) -> Tuple[int, float, float, np.ndarray]:
        """Get action, log_prob, value, and embedding."""
        dist, value, embedding = self.forward(state)
        action = dist.sample()
        return (
            action.item(),
            dist.log_prob(action).item(),
            value.item(),
            embedding.detach().cpu().numpy().flatten(),
        )

    def evaluate_actions(self, states: torch.Tensor, actions: torch.Tensor):
        """Evaluate actions for PPO update."""
        dist, values, embeddings = self.forward(states)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, values.squeeze(-1), entropy
