"""
GreenCloudRL - High-Level PPO Agent
Proximal Policy Optimization for server power management.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Tuple, Optional
from copy import deepcopy

from .networks import PPOActorCritic


class PPOBuffer:
    """Buffer for PPO rollout collection."""

    def __init__(self):
        self.clear()

    def add(self, state, action, reward, log_prob, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []

    def __len__(self):
        return len(self.states)


class HighLevelPPO:
    """
    PPO agent for high-level server management decisions.
    
    Decides: server on/off, DVFS levels.
    Operates at coarser time scale than the low-level agent.
    Provides policy embedding to communicate strategy to low-level agent.
    """

    def __init__(
        self,
        state_dim: int = 12,
        action_dim: int = 23,   # 1 + 2*N + 3 DVFS levels
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coeff: float = 0.01,
        value_loss_coeff: float = 0.5,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 4,
        minibatch_size: int = 64,
        hidden_sizes: list = [128, 128],
        device: str = "auto",
    ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coeff = entropy_coeff
        self.value_loss_coeff = value_loss_coeff
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.minibatch_size = minibatch_size

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.network = PPOActorCritic(state_dim, action_dim, hidden_sizes).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr, eps=1e-5)

        self.buffer = PPOBuffer()
        self.update_count = 0

    def select_action(self, state: np.ndarray) -> Tuple[int, float, float, np.ndarray]:
        """
        Select high-level action and return embedding for low-level agent.
        
        Returns:
            action, log_prob, value, policy_embedding
        """
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, log_prob, value, embedding = self.network.get_action_and_value(state_t)
        return action, log_prob, value, embedding

    def store_transition(self, state, action, reward, log_prob, value, done):
        self.buffer.add(state, action, reward, log_prob, value, done)

    def compute_gae(self, next_value: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Generalized Advantage Estimation."""
        rewards = self.buffer.rewards
        values = self.buffer.values + [next_value]
        dones = self.buffer.dones

        advantages = []
        gae = 0.0

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = advantages + torch.FloatTensor(self.buffer.values).to(self.device)

        return advantages, returns

    def update(self, next_state: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Perform PPO update."""
        if len(self.buffer) < self.minibatch_size:
            return {}

        # Compute next value
        if next_state is not None:
            state_t = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                _, next_value, _ = self.network(state_t)
                next_value = next_value.item()
        else:
            next_value = 0.0

        advantages, returns = self.compute_gae(next_value)

        # Normalize advantages
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Convert buffer to tensors
        states = torch.FloatTensor(np.array(self.buffer.states)).to(self.device)
        actions = torch.LongTensor(self.buffer.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.buffer.log_probs).to(self.device)

        # PPO update epochs
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0

        for _ in range(self.ppo_epochs):
            # Shuffle data
            indices = np.random.permutation(len(self.buffer))

            for start in range(0, len(indices), self.minibatch_size):
                end = start + self.minibatch_size
                mb_idx = indices[start:end]

                mb_states = states[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_log_probs = old_log_probs[mb_idx]
                mb_advantages = advantages[mb_idx]
                mb_returns = returns[mb_idx]

                # Evaluate actions
                new_log_probs, new_values, entropy = self.network.evaluate_actions(
                    mb_states, mb_actions
                )

                # Policy loss (PPO clipping)
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = nn.functional.mse_loss(new_values, mb_returns)

                # Total loss
                loss = (
                    policy_loss
                    + self.value_loss_coeff * value_loss
                    - self.entropy_coeff * entropy.mean()
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()

        n_updates = max(1, self.ppo_epochs * (len(self.buffer) // self.minibatch_size))
        self.buffer.clear()
        self.update_count += 1

        return {
            "policy_loss": total_policy_loss / n_updates,
            "value_loss": total_value_loss / n_updates,
            "entropy": total_entropy / n_updates,
        }

    def get_embedding(self, state: np.ndarray) -> np.ndarray:
        """Get policy embedding for low-level agent."""
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, _, embedding = self.network(state_t)
        return embedding.cpu().numpy().flatten()

    def save(self, path: str):
        torch.save({
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "update_count": self.update_count,
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint["network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.update_count = checkpoint["update_count"]
