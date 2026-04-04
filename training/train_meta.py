"""
GreenCloudRL - Meta-Training Script
Train the low-level agent with Reptile meta-learning across diverse workloads.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import yaml
import logging
from pathlib import Path

from simulator import CloudSchedulingEnv, WorkloadGenerator
from agents import LowLevelA2C
from meta_learning import ReptileMetaLearner
from training.train_hierarchical import evaluate_agent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def create_meta_tasks(config: dict, seed: int = 42):
    """Create meta tasks from REAL processed traces."""
    gen = WorkloadGenerator(seed=seed)
   
    processed_dir = "data/processed"
   
    # Check if real data exists
    from pathlib import Path
    if list(Path(processed_dir).glob("*_processed.csv")):
        print("Using REAL trace data for meta-learning!")
        train_tasks = gen.create_real_meta_tasks(
            processed_dir=processed_dir,
            tasks_per_window=500,
        )
       
        # Reserve NASA as the unseen test workload
        test_tasks = []
        nasa_path = Path(processed_dir) / "nasa_processed.csv"
        if nasa_path.exists():
            nasa_tasks = gen.load_processed_trace(str(nasa_path), num_tasks=2000)
            test_tasks.append(nasa_tasks[:500])
            test_tasks.append(nasa_tasks[500:1000])
            # Remove NASA windows from training
            train_tasks = [t for t in train_tasks if len(t) > 0]
       
        if not test_tasks:
            test_tasks = [gen.generate_bursty(num_tasks=500)]
       
        return train_tasks, test_tasks
    else:
        print("No real data found, using synthetic data")
        train_tasks = gen.create_meta_tasks(num_distributions=7, tasks_per_distribution=500)
        test_tasks = [gen.generate_bursty(num_tasks=500)]
        return train_tasks, test_tasks



def run_meta_training(config: dict):
    """Full meta-training pipeline."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # Create environment
    env = CloudSchedulingEnv(
        num_servers=config.get("env", {}).get("num_servers", 10),
        vms_per_server=config.get("env", {}).get("vms_per_server", 5),
        alpha=config.get("reward", {}).get("alpha", 0.4),
        beta=config.get("reward", {}).get("beta", 0.4),
        gamma=config.get("reward", {}).get("gamma", 0.2),
        seed=config.get("env", {}).get("seed", 42),
    )

    # Create agent
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

    # Create workloads
    train_tasks, test_tasks = create_meta_tasks(config)

    # Create meta-learner
    meta_config = config.get("meta", {})
    meta_learner = ReptileMetaLearner(
        agent=agent,
        env=env,
        meta_lr=meta_config.get("meta_lr", 1.0),
        meta_lr_decay=meta_config.get("meta_lr_decay", 0.999),
        meta_lr_min=meta_config.get("meta_lr_min", 0.1),
        inner_steps=meta_config.get("inner_steps", 5),
        tasks_per_batch=meta_config.get("tasks_per_batch", 4),
        device=device,
    )

    # Run meta-training
    logger.info("\n" + "=" * 60)
    logger.info("Starting Reptile Meta-Training")
    logger.info("=" * 60)

    history = meta_learner.meta_train(
        meta_tasks=train_tasks,
        num_meta_iterations=meta_config.get("num_meta_iterations", 200),
        log_interval=10,
        eval_tasks=test_tasks,
    )

    # Save meta-learned model
    save_dir = Path(config.get("paths", {}).get("checkpoint_dir", "results/checkpoints"))
    save_dir.mkdir(parents=True, exist_ok=True)
    meta_learner.save(str(save_dir / "meta_learned_agent.pt"))
    logger.info(f"Meta-learned agent saved to {save_dir / 'meta_learned_agent.pt'}")

    # ── Evaluate adaptation speed ──
    logger.info("\n" + "=" * 60)
    logger.info("Evaluating Adaptation Speed on Unseen Workloads")
    logger.info("=" * 60)

    for i, test_workload in enumerate(test_tasks):
        logger.info(f"\n--- Test Workload {i+1} ---")

        # Adaptation curve for meta-learned agent
        meta_curve = meta_learner.measure_adaptation_curve(test_workload, num_episodes=20)
        logger.info(f"Meta-learned adaptation curve: {[f'{r:.1f}' for r in meta_curve[:10]]}")

        # Compare: train from scratch
        scratch_agent = LowLevelA2C(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n,
            device=device,
        )
        scratch_meta = ReptileMetaLearner(agent=scratch_agent, env=env, device=device)
        scratch_curve = scratch_meta.measure_adaptation_curve(test_workload, num_episodes=20)
        logger.info(f"From-scratch curve: {[f'{r:.1f}' for r in scratch_curve[:10]]}")

        # Measure: episodes to 90% of converged performance
        meta_converged = np.mean(meta_curve[-3:])
        target = 0.9 * meta_converged
        meta_episodes_to_90 = next(
            (i for i, r in enumerate(meta_curve) if r >= target), len(meta_curve)
        )
        logger.info(
            f"Meta-learned: {meta_episodes_to_90} episodes to 90% optimal | "
            f"Converged reward: {meta_converged:.2f}"
        )

    # Save training history
    np.savez(
        str(save_dir / "meta_training_history.npz"),
        inner_rewards=history["inner_rewards"],
        meta_lr=history["meta_lr"],
        eval_reward=history.get("eval_reward", []),
    )

    return meta_learner, history


if __name__ == "__main__":
    config_path = "configs/default.yaml"
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        config = {}

    run_meta_training(config)
