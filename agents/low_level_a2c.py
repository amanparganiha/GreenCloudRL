"""
GreenCloudRL - Low-Level A2C Agent
Advantage Actor-Critic for task-to-VM assignment.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple, Optional, Dict
from copy import deepcopy

from .networks import ActorNetwork, CriticNetwork


class RolloutBuffer:
    """Stores rollout data for N-step A2C updates."""

    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
        self.action_masks = []

    def add(self, state, action, reward, log_prob, value, done, action_mask=None):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)
        self.action_masks.append(action_mask)

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.values.clear()
        self.dones.clear()
        self.action_masks.clear()

    def __len__(self):
        return len(self.states)


class LowLevelA2C:
    """
    Advantage Actor-Critic agent for task scheduling (low-level).
    
    Features:
    - Separate actor and critic networks
    - N-step returns
    - Entropy regularization
    - Invalid action masking
    - Gradient clipping
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        actor_lr: float = 3e-4,
        critic_lr: float = 1e-3,
        gamma: float = 0.99,
        entropy_coeff: float = 0.01,
        value_loss_coeff: float = 0.5,
        max_grad_norm: float = 0.5,
        n_steps: int = 5,
        hidden_sizes: List[int] = [256, 256, 128],
        device: str = "auto",
    ):
        self.gamma = gamma
        self.entropy_coeff = entropy_coeff
        self.value_loss_coeff = value_loss_coeff
        self.max_grad_norm = max_grad_norm
        self.n_steps = n_steps

        # Device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Networks
        self.actor = ActorNetwork(state_dim, action_dim, hidden_sizes).to(self.device)
        self.critic = CriticNetwork(state_dim, hidden_sizes).to(self.device)

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        # Rollout buffer
        self.buffer = RolloutBuffer()

        # Tracking
        self.update_count = 0
        self.total_loss_history = []

    def select_action(self, state: np.ndarray, action_mask: Optional[np.ndarray] = None) -> Tuple[int, float, float]:
        """
        Select action using current policy.
        
        Returns:
            action, log_probability, estimated_value
        """
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        mask_t = None
        if action_mask is not None:
            mask_t = torch.FloatTensor(action_mask).unsqueeze(0).to(self.device)

        with torch.no_grad():
            dist = self.actor(state_t, mask_t)
            value = self.critic(state_t)

        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.item(), log_prob.item(), value.item()

    def store_transition(self, state, action, reward, log_prob, value, done, action_mask=None):
        """Store a transition in the rollout buffer."""
        self.buffer.add(state, action, reward, log_prob, value, done, action_mask)

    def update(self, next_state: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Perform A2C update using collected rollout.
        
        Returns:
            Dictionary of loss metrics
        """
        if len(self.buffer) < self.n_steps:
            return {}

        # Compute returns and advantages
        returns = []
        advantages = []

        # Bootstrap value
        if next_state is not None:
            next_state_t = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                next_value = self.critic(next_state_t).item()
        else:
            next_value = 0.0

        # N-step returns (reverse accumulation)
        R = next_value
        for i in reversed(range(len(self.buffer))):
            if self.buffer.dones[i]:
                R = 0.0
            R = self.buffer.rewards[i] + self.gamma * R
            returns.insert(0, R)
            advantages.insert(0, R - self.buffer.values[i])

        # Convert to tensors
        states = torch.FloatTensor(np.array(self.buffer.states)).to(self.device)
        actions = torch.LongTensor(self.buffer.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.buffer.log_probs).to(self.device)
        returns_t = torch.FloatTensor(returns).to(self.device)
        advantages_t = torch.FloatTensor(advantages).to(self.device)

        # Normalize advantages
        if len(advantages_t) > 1:
            advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        # Action masks
        masks = None
        if self.buffer.action_masks[0] is not None:
            masks = torch.FloatTensor(np.array([
                m if m is not None else np.ones(self.actor.action_head.out_features)
                for m in self.buffer.action_masks
            ])).to(self.device)

        # ── Actor loss ──
        dist = self.actor(states, masks)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        actor_loss = -(log_probs * advantages_t.detach()).mean()
        actor_loss = actor_loss - self.entropy_coeff * entropy

        # ── Critic loss ──
        values = self.critic(states).squeeze(-1)
        critic_loss = self.value_loss_coeff * nn.functional.mse_loss(values, returns_t)

        # ── Update actor ──
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()

        # ── Update critic ──
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()

        # Clear buffer
        self.buffer.clear()
        self.update_count += 1

        metrics = {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "entropy": entropy.item(),
            "avg_advantage": advantages_t.mean().item(),
            "avg_return": returns_t.mean().item(),
        }
        self.total_loss_history.append(actor_loss.item() + critic_loss.item())

        return metrics

    def get_parameters(self) -> dict:
        """Get all parameters (for meta-learning)."""
        return {
            "actor": deepcopy(self.actor.state_dict()),
            "critic": deepcopy(self.critic.state_dict()),
        }

    def set_parameters(self, params: dict):
        """Set parameters (for meta-learning)."""
        self.actor.load_state_dict(params["actor"])
        self.critic.load_state_dict(params["critic"])

    def save(self, path: str):
        """Save model checkpoint."""
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_opt": self.actor_optimizer.state_dict(),
            "critic_opt": self.critic_optimizer.state_dict(),
            "update_count": self.update_count,
        }, path)

    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_opt"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_opt"])
        self.update_count = checkpoint["update_count"]
