"""
GreenCloudRL - Hierarchical Agent Coordinator
Manages interaction between high-level and low-level agents.
"""

import numpy as np
from typing import Dict, Optional

from .low_level_a2c import LowLevelA2C
from .high_level_ppo import HighLevelPPO


class HierarchicalAgent:
    """
    Coordinates the two-level hierarchical RL system.
    
    High-level agent: Makes server management decisions every N steps.
    Low-level agent: Makes task-to-VM assignments for every task.
    Communication: High-level embedding is fed to low-level as part of state.
    """

    def __init__(self, env, config: dict, device: str = "auto"):
        self.env = env
        self.config = config
        self.decision_interval = config.get("high_level", {}).get("decision_interval", 10)

        # Initialize agents
        self.low_level = LowLevelA2C(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n,
            actor_lr=config.get("low_level", {}).get("actor_lr", 3e-4),
            critic_lr=config.get("low_level", {}).get("critic_lr", 1e-3),
            gamma=config.get("low_level", {}).get("gamma", 0.99),
            entropy_coeff=config.get("low_level", {}).get("entropy_coeff", 0.01),
            value_loss_coeff=config.get("low_level", {}).get("value_loss_coeff", 0.5),
            max_grad_norm=config.get("low_level", {}).get("max_grad_norm", 0.5),
            n_steps=config.get("low_level", {}).get("n_steps", 5),
            hidden_sizes=config.get("low_level", {}).get("hidden_sizes", [256, 256, 128]),
            device=device,
        )

        hl_state_dim = 12  # high-level observation dimension
        hl_action_dim = 1 + 2 * env.num_servers + 3  # no-op + on/off + 3 DVFS
        self.high_level = HighLevelPPO(
            state_dim=hl_state_dim,
            action_dim=hl_action_dim,
            lr=config.get("high_level", {}).get("lr", 3e-4),
            gamma=config.get("high_level", {}).get("gamma", 0.99),
            gae_lambda=config.get("high_level", {}).get("gae_lambda", 0.95),
            clip_epsilon=config.get("high_level", {}).get("clip_epsilon", 0.2),
            entropy_coeff=config.get("high_level", {}).get("entropy_coeff", 0.01),
            ppo_epochs=config.get("high_level", {}).get("ppo_epochs", 4),
            minibatch_size=config.get("high_level", {}).get("minibatch_size", 64),
            hidden_sizes=config.get("high_level", {}).get("hidden_sizes", [128, 128]),
            device=device,
        )

        # Tracking
        self.step_count = 0
        self.high_level_reward_acc = 0.0
        self.last_high_level_state = None
        self.last_high_level_action = None
        self.last_high_level_log_prob = None
        self.last_high_level_value = None

    def select_action(self, obs: np.ndarray, info: dict) -> int:
        """
        Select action using hierarchical policy.
        
        Every decision_interval steps, the high-level agent acts.
        The low-level agent acts on every step.
        """
        self.step_count += 1

        # ── High-level decision ──
        if self.step_count % self.decision_interval == 1 or self.step_count == 1:
            hl_obs = self.env.get_high_level_observation()

            # Store previous high-level transition
            if self.last_high_level_state is not None:
                self.high_level.store_transition(
                    self.last_high_level_state,
                    self.last_high_level_action,
                    self.high_level_reward_acc,
                    self.last_high_level_log_prob,
                    self.last_high_level_value,
                    done=False,
                )
                self.high_level_reward_acc = 0.0

            # Get high-level action and embedding
            hl_action, hl_log_prob, hl_value, embedding = self.high_level.select_action(hl_obs)

            # Apply high-level action to environment
            self.env.apply_high_level_action(hl_action)

            # Send embedding to low-level agent
            self.env.set_high_level_embedding(embedding)

            # Store for next transition
            self.last_high_level_state = hl_obs
            self.last_high_level_action = hl_action
            self.last_high_level_log_prob = hl_log_prob
            self.last_high_level_value = hl_value

        # ── Low-level decision ──
        action_mask = info.get("action_mask", None)
        action, log_prob, value = self.low_level.select_action(obs, action_mask)

        return action

    def store_transition(self, obs, action, reward, log_prob, value, done, info):
        """Store transitions for both agents."""
        action_mask = info.get("action_mask", None)

        # Low-level transition
        _, ll_log_prob, ll_value = self.low_level.select_action(obs, action_mask)
        self.low_level.store_transition(obs, action, reward, ll_log_prob, ll_value, done, action_mask)

        # Accumulate reward for high-level
        self.high_level_reward_acc += reward

    def update(self, next_obs: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Update both agents."""
        metrics = {}

        # Low-level update
        ll_metrics = self.low_level.update(next_obs)
        for k, v in ll_metrics.items():
            metrics[f"low_level/{k}"] = v

        # High-level update (less frequent)
        if len(self.high_level.buffer) >= self.high_level.minibatch_size:
            hl_obs = self.env.get_high_level_observation() if next_obs is not None else None
            hl_metrics = self.high_level.update(hl_obs)
            for k, v in hl_metrics.items():
                metrics[f"high_level/{k}"] = v

        return metrics

    def end_episode(self, final_obs, done):
        """Handle end of episode for both agents."""
        # Store final high-level transition
        if self.last_high_level_state is not None:
            self.high_level.store_transition(
                self.last_high_level_state,
                self.last_high_level_action,
                self.high_level_reward_acc,
                self.last_high_level_log_prob,
                self.last_high_level_value,
                done=True,
            )

        # Reset
        self.step_count = 0
        self.high_level_reward_acc = 0.0
        self.last_high_level_state = None

    def save(self, path: str):
        self.low_level.save(f"{path}_low_level.pt")
        self.high_level.save(f"{path}_high_level.pt")

    def load(self, path: str):
        self.low_level.load(f"{path}_low_level.pt")
        self.high_level.load(f"{path}_high_level.pt")
