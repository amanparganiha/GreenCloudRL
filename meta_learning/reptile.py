"""
GreenCloudRL - Reptile Meta-Learning
Trains the low-level agent to rapidly adapt to unseen workloads.
"""

import torch
import numpy as np
from copy import deepcopy
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


class ReptileMetaLearner:
    """
    Reptile meta-learning for rapid workload adaptation.
    
    Algorithm:
        For each meta-iteration:
            1. Save current parameters theta
            2. For each task (workload distribution):
                a. Initialize agent with theta
                b. Run k inner gradient updates on the task
                c. Record adapted parameters theta'_k
            3. Meta-update: theta = theta + epsilon * (avg(theta'_k) - theta)
    
    This finds an initialization that's close to good solutions for all tasks.
    """

    def __init__(
        self,
        agent,  # LowLevelA2C
        env,    # CloudSchedulingEnv
        meta_lr: float = 1.0,
        meta_lr_decay: float = 0.999,
        meta_lr_min: float = 0.1,
        inner_steps: int = 5,
        tasks_per_batch: int = 4,
        device: str = "auto",
    ):
        self.agent = agent
        self.env = env
        self.meta_lr = meta_lr
        self.meta_lr_decay = meta_lr_decay
        self.meta_lr_min = meta_lr_min
        self.inner_steps = inner_steps
        self.tasks_per_batch = tasks_per_batch

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # History
        self.meta_iteration = 0
        self.adaptation_history = []

    def _run_episode(self, workload) -> float:
        """Run a single episode on a given workload and update agent."""
        self.env.set_workload(workload)
        obs, info = self.env.reset()
        episode_reward = 0.0
        done = False

        while not done:
            action_mask = info.get("action_mask", None)
            action, log_prob, value = self.agent.select_action(obs, action_mask)

            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            self.agent.store_transition(obs, action, reward, log_prob, value, done, action_mask)
            episode_reward += reward
            obs = next_obs

            # Update every n_steps
            if len(self.agent.buffer) >= self.agent.n_steps:
                self.agent.update(next_obs if not done else None)

        # Final update
        if len(self.agent.buffer) > 0:
            self.agent.update(None)

        return episode_reward

    def _average_parameters(self, param_list: List[dict]) -> dict:
        """Average parameters from multiple adapted agents."""
        avg_params = {}
        for key in ["actor", "critic"]:
            avg_state = {}
            for param_name in param_list[0][key]:
                tensors = [p[key][param_name] for p in param_list]
                avg_state[param_name] = torch.stack(tensors).mean(dim=0)
            avg_params[key] = avg_state
        return avg_params

    def _interpolate_parameters(self, original: dict, target: dict, epsilon: float) -> dict:
        """Interpolate between original and target parameters."""
        result = {}
        for key in ["actor", "critic"]:
            result_state = {}
            for param_name in original[key]:
                result_state[param_name] = (
                    original[key][param_name]
                    + epsilon * (target[key][param_name] - original[key][param_name])
                )
            result[key] = result_state
        return result

    def meta_train(
        self,
        meta_tasks: List[list],    # List of workload distributions
        num_meta_iterations: int = 1000,
        log_interval: int = 10,
        eval_tasks: Optional[List[list]] = None,
    ) -> Dict[str, list]:
        """
        Run the Reptile meta-training loop.
        
        Args:
            meta_tasks: List of workload task lists (training distributions)
            num_meta_iterations: Number of outer loop iterations
            log_interval: How often to log progress
            eval_tasks: Optional held-out tasks for evaluation
            
        Returns:
            Training history dict
        """
        history = {
            "meta_loss": [],
            "inner_rewards": [],
            "meta_lr": [],
            "eval_reward": [],
        }

        logger.info(f"Starting meta-training: {num_meta_iterations} iterations, "
                     f"{len(meta_tasks)} tasks, {self.inner_steps} inner steps")

        for meta_iter in tqdm(range(num_meta_iterations), desc="Meta-Training"):
            self.meta_iteration = meta_iter

            # Save original parameters
            original_params = self.agent.get_parameters()

            # Sample tasks for this batch
            task_indices = np.random.choice(
                len(meta_tasks), size=min(self.tasks_per_batch, len(meta_tasks)), replace=False
            )

            adapted_params_list = []
            batch_rewards = []

            for task_idx in task_indices:
                workload = meta_tasks[task_idx]

                # Reset agent to original params
                self.agent.set_parameters(deepcopy(original_params))

                # Inner loop: train on this task for k steps
                task_reward = 0.0
                for inner_step in range(self.inner_steps):
                    ep_reward = self._run_episode(workload)
                    task_reward += ep_reward

                batch_rewards.append(task_reward / self.inner_steps)

                # Record adapted parameters
                adapted_params_list.append(self.agent.get_parameters())

            # ── Reptile meta-update ──
            avg_adapted = self._average_parameters(adapted_params_list)
            new_params = self._interpolate_parameters(
                original_params, avg_adapted, self.meta_lr
            )
            self.agent.set_parameters(new_params)

            # Decay meta learning rate
            self.meta_lr = max(
                self.meta_lr * self.meta_lr_decay,
                self.meta_lr_min,
            )

            # Record history
            avg_reward = np.mean(batch_rewards)
            history["inner_rewards"].append(avg_reward)
            history["meta_lr"].append(self.meta_lr)

            # Logging
            if (meta_iter + 1) % log_interval == 0:
                logger.info(
                    f"Meta-iter {meta_iter + 1}/{num_meta_iterations} | "
                    f"Avg reward: {avg_reward:.2f} | "
                    f"Meta-LR: {self.meta_lr:.4f}"
                )

                # Evaluate on held-out tasks
                if eval_tasks is not None:
                    eval_reward = self.evaluate_adaptation(eval_tasks, num_adapt_episodes=3)
                    history["eval_reward"].append(eval_reward)
                    logger.info(f"  Eval reward: {eval_reward:.2f}")

        return history

    def evaluate_adaptation(
        self,
        eval_tasks: List[list],
        num_adapt_episodes: int = 5,
    ) -> float:
        """
        Evaluate adaptation speed on unseen tasks.
        
        Returns average reward after adaptation.
        """
        original_params = self.agent.get_parameters()
        rewards = []

        for workload in eval_tasks:
            self.agent.set_parameters(deepcopy(original_params))
            task_rewards = []

            for ep in range(num_adapt_episodes):
                ep_reward = self._run_episode(workload)
                task_rewards.append(ep_reward)

            rewards.append(np.mean(task_rewards[-2:]))  # Last 2 episodes

        # Restore original params
        self.agent.set_parameters(original_params)
        return np.mean(rewards)

    def measure_adaptation_curve(
        self,
        workload: list,
        num_episodes: int = 20,
    ) -> List[float]:
        """
        Measure how quickly agent adapts to a new workload.
        Returns reward at each adaptation episode.
        """
        original_params = self.agent.get_parameters()
        self.agent.set_parameters(deepcopy(original_params))

        rewards = []
        for ep in range(num_episodes):
            ep_reward = self._run_episode(workload)
            rewards.append(ep_reward)

        self.agent.set_parameters(original_params)
        return rewards

    def save(self, path: str):
        """Save meta-learned parameters."""
        self.agent.save(path)

    def load(self, path: str):
        """Load meta-learned parameters."""
        self.agent.load(path)
