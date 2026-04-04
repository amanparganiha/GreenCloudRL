"""
GreenCloudRL - Hierarchical Training Script
Train the two-level HRL system on a given workload.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import yaml
import logging
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Optional

from simulator import CloudSchedulingEnv
from agents import HierarchicalAgent, LowLevelA2C
from baselines import get_all_baselines

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def train_single_level(
    env: CloudSchedulingEnv,
    config: dict,
    num_episodes: int = 500,
    device: str = "auto",
) -> LowLevelA2C:
    """
    Stage 1: Train a single-level A2C agent (no hierarchy).
    Used as both a sanity check and an ablation baseline.
    """
    agent = LowLevelA2C(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        actor_lr=config.get("low_level", {}).get("actor_lr", 3e-4),
        critic_lr=config.get("low_level", {}).get("critic_lr", 1e-3),
        gamma=config.get("low_level", {}).get("gamma", 0.99),
        entropy_coeff=config.get("low_level", {}).get("entropy_coeff", 0.01),
        n_steps=config.get("low_level", {}).get("n_steps", 5),
        hidden_sizes=config.get("low_level", {}).get("hidden_sizes", [256, 256, 128]),
        device=device,
    )

    reward_history = []
    best_reward = float("-inf")

    logger.info(f"=== Stage 1: Single-Level A2C Training ({num_episodes} episodes) ===")

    for episode in tqdm(range(num_episodes), desc="Single-Level Training"):
        obs, info = env.reset(seed=episode)
        episode_reward = 0.0
        done = False
        steps = 0

        while not done:
            action_mask = info.get("action_mask", None)
            action, log_prob, value = agent.select_action(obs, action_mask)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.store_transition(obs, action, reward, log_prob, value, done, action_mask)
            episode_reward += reward
            obs = next_obs
            steps += 1

            if len(agent.buffer) >= agent.n_steps:
                agent.update(next_obs if not done else None)

        if len(agent.buffer) > 0:
            agent.update(None)

        reward_history.append(episode_reward)

        if episode_reward > best_reward:
            best_reward = episode_reward

        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(reward_history[-50:])
            logger.info(
                f"Episode {episode+1}/{num_episodes} | "
                f"Avg Reward (50): {avg_reward:.2f} | Best: {best_reward:.2f}"
            )

    return agent


def train_hierarchical(
    env: CloudSchedulingEnv,
    config: dict,
    num_episodes: int = 1000,
    device: str = "auto",
) -> HierarchicalAgent:
    """
    Stage 2: Train the full hierarchical system (high-level PPO + low-level A2C).
    """
    agent = HierarchicalAgent(env, config, device=device)

    reward_history = []
    best_reward = float("-inf")
    save_dir = Path(config.get("paths", {}).get("checkpoint_dir", "results/checkpoints"))
    save_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"=== Stage 2: Hierarchical Training ({num_episodes} episodes) ===")

    for episode in tqdm(range(num_episodes), desc="Hierarchical Training"):
        obs, info = env.reset(seed=episode)
        episode_reward = 0.0
        done = False

        while not done:
            action = agent.select_action(obs, info)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            action_mask = info.get("action_mask", None)
            _, log_prob, value = agent.low_level.select_action(obs, action_mask)
            agent.store_transition(obs, action, reward, log_prob, value, done, info)
            episode_reward += reward
            obs = next_obs

            if len(agent.low_level.buffer) >= agent.low_level.n_steps:
                agent.update(next_obs if not done else None)

        agent.end_episode(obs, done)

        if len(agent.low_level.buffer) > 0:
            agent.update(None)

        reward_history.append(episode_reward)

        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save(str(save_dir / "best_model"))

        if (episode + 1) % config.get("training", {}).get("log_interval", 50) == 0:
            avg_reward = np.mean(reward_history[-50:])
            ep_info = env._get_info()
            logger.info(
                f"Episode {episode+1}/{num_episodes} | "
                f"Avg Reward: {avg_reward:.2f} | Best: {best_reward:.2f} | "
                f"SLA Viol: {ep_info['sla']['violation_rate']:.2%} | "
                f"Energy: {ep_info['energy']['total_energy_kwh']:.4f} kWh"
            )

        if (episode + 1) % config.get("training", {}).get("save_interval", 200) == 0:
            agent.save(str(save_dir / f"checkpoint_ep{episode+1}"))

    return agent


def evaluate_agent(env, agent, num_episodes: int = 10, is_baseline: bool = False) -> Dict:
    """Evaluate an agent over multiple episodes."""
    results = {
        "rewards": [],
        "makespans": [],
        "energy_kwh": [],
        "sla_violation_rates": [],
        "avg_cpu_utils": [],
        "avg_response_times": [],
    }

    for ep in range(num_episodes):
        obs, info = env.reset(seed=1000 + ep)
        episode_reward = 0.0
        done = False

        while not done:
            action_mask = info.get("action_mask", None)
            if is_baseline:
                action = agent.select_action(obs, action_mask)
            elif hasattr(agent, "select_action") and not hasattr(agent, "low_level"):
                action, _, _ = agent.select_action(obs, action_mask)
            else:
                action = agent.select_action(obs, info)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            obs = next_obs

        ep_info = env._get_info()
        results["rewards"].append(episode_reward)
        results["makespans"].append(ep_info.get("makespan", 0))
        results["energy_kwh"].append(ep_info["energy"]["total_energy_kwh"])
        results["sla_violation_rates"].append(ep_info["sla"]["violation_rate"])
        results["avg_cpu_utils"].append(ep_info.get("avg_cpu_util", 0))
        results["avg_response_times"].append(ep_info["sla"]["avg_response_time"])

    # Compute statistics
    summary = {}
    for key, values in results.items():
        summary[f"{key}_mean"] = np.mean(values)
        summary[f"{key}_std"] = np.std(values)

    return summary


def run_baseline_comparison(env, config, num_episodes: int = 10) -> Dict:
    """Run all baselines and return comparison results."""
    baselines = get_all_baselines(
        total_vms=env.total_vms,
        vms_per_server=env.vms_per_server,
    )

    all_results = {}
    for baseline in baselines:
        logger.info(f"Evaluating baseline: {baseline.name}")
        results = evaluate_agent(env, baseline, num_episodes=num_episodes, is_baseline=True)
        all_results[baseline.name] = results
        logger.info(
            f"  {baseline.name}: Reward={results['rewards_mean']:.2f}±{results['rewards_std']:.2f}, "
            f"SLA={results['sla_violation_rates_mean']:.2%}, "
            f"Energy={results['energy_kwh_mean']:.4f} kWh"
        )

    return all_results


if __name__ == "__main__":
    # Load config
    config_path = "configs/default.yaml"
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        config = {}

    # Create environment
    env = CloudSchedulingEnv(
        num_servers=config.get("env", {}).get("num_servers", 10),
        vms_per_server=config.get("env", {}).get("vms_per_server", 5),
        alpha=config.get("reward", {}).get("alpha", 0.4),
        beta=config.get("reward", {}).get("beta", 0.4),
        gamma=config.get("reward", {}).get("gamma", 0.2),
        seed=config.get("env", {}).get("seed", 42),
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Stage 1: Baseline comparison
    logger.info("\n" + "=" * 60)
    logger.info("Running baseline comparison...")
    baseline_results = run_baseline_comparison(env, config, num_episodes=10)

    # Stage 2: Single-level DRL
    logger.info("\n" + "=" * 60)
    single_agent = train_single_level(env, config, num_episodes=300, device=device)
    single_results = evaluate_agent(env, single_agent, num_episodes=10)
    logger.info(
        f"Single-Level DRL: Reward={single_results['rewards_mean']:.2f}, "
        f"SLA={single_results['sla_violation_rates_mean']:.2%}"
    )

    # Stage 3: Hierarchical DRL
    logger.info("\n" + "=" * 60)
    hier_agent = train_hierarchical(env, config, num_episodes=500, device=device)
    hier_results = evaluate_agent(env, hier_agent, num_episodes=10)
    logger.info(
        f"Hierarchical DRL: Reward={hier_results['rewards_mean']:.2f}, "
        f"SLA={hier_results['sla_violation_rates_mean']:.2%}"
    )

    logger.info("\n=== Training Complete ===")
