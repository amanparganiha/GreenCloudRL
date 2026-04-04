#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════╗
║                       GreenCloudRL                          ║
║  Hierarchical Meta-RL for Energy-Efficient Cloud Scheduling ║
╚══════════════════════════════════════════════════════════════╝

Main entry point: runs the full training + evaluation pipeline.

Usage:
    python main.py                     # Full pipeline
    python main.py --stage 1           # Only baselines
    python main.py --stage 2           # Single-level DRL
    python main.py --stage 3           # Hierarchical DRL
    python main.py --stage 4           # Meta-learning
    python main.py --stage 5           # Evaluation & plots
    python main.py --stage 6           # Explainability
    python main.py --config configs/default.yaml
"""

import sys
import os
import argparse
import yaml
import torch
import numpy as np
import logging
from pathlib import Path

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from simulator import CloudSchedulingEnv, WorkloadGenerator
from agents import LowLevelA2C, HierarchicalAgent
from meta_learning import ReptileMetaLearner
from baselines import get_all_baselines
from explainability import SHAPExplainer
from training.train_hierarchical import (
    train_single_level, train_hierarchical, evaluate_agent, run_baseline_comparison,
)
from training.train_meta import run_meta_training, create_meta_tasks
from training.evaluate import (
    plot_training_curves, plot_metric_comparison, plot_adaptation_curves,
    plot_ablation_study, plot_energy_breakdown, generate_results_table,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("greencloudrl.log"),
    ],
)
logger = logging.getLogger(__name__)


def print_banner():
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║       ██████╗ ██████╗ ███████╗███████╗███╗   ██╗             ║
    ║      ██╔════╝ ██╔══██╗██╔════╝██╔════╝████╗  ██║             ║
    ║      ██║  ███╗██████╔╝█████╗  █████╗  ██╔██╗ ██║             ║
    ║      ██║   ██║██╔══██╗██╔══╝  ██╔══╝  ██║╚██╗██║             ║
    ║      ╚██████╔╝██║  ██║███████╗███████╗██║ ╚████║             ║
    ║       ╚═════╝ ╚═╝  ╚═╝╚══════╝╚══════╝╚═╝  ╚═══╝             ║
    ║              CloudRL                                         ║
    ║     Hierarchical Meta-RL for Cloud Scheduling                ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def run_full_pipeline(config: dict, stages: list = None):
    """Execute the complete GreenCloudRL pipeline."""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"PyTorch device: {device}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # Paths
    checkpoint_dir = Path(config.get("paths", {}).get("checkpoint_dir", "results/checkpoints"))
    figures_dir = Path(config.get("paths", {}).get("figures_dir", "results/figures"))
    tables_dir = Path("results/tables")
    for d in [checkpoint_dir, figures_dir, tables_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Create environment
    env = CloudSchedulingEnv(
        num_servers=config.get("env", {}).get("num_servers", 10),
        vms_per_server=config.get("env", {}).get("vms_per_server", 5),
        alpha=config.get("reward", {}).get("alpha", 0.4),
        beta=config.get("reward", {}).get("beta", 0.4),
        gamma=config.get("reward", {}).get("gamma", 0.2),
        seed=config.get("env", {}).get("seed", 42),
    )

    if stages is None:
        stages = [1, 2, 3, 4, 5, 6]

    all_results = {}
    reward_histories = {}
    num_eval = config.get("training", {}).get("num_eval_episodes", 10)

    # ══════════════════════════════════════════
    # STAGE 1: Baseline Comparison
    # ══════════════════════════════════════════
    if 1 in stages:
        logger.info("\n" + "═" * 60)
        logger.info("  STAGE 1: Baseline Evaluation")
        logger.info("═" * 60)

        baseline_results = run_baseline_comparison(env, config, num_episodes=num_eval)
        all_results.update(baseline_results)

    # ══════════════════════════════════════════
    # STAGE 2: Single-Level DRL
    # ══════════════════════════════════════════
    if 2 in stages:
        logger.info("\n" + "═" * 60)
        logger.info("  STAGE 2: Single-Level DRL (A2C)")
        logger.info("═" * 60)

        single_agent = train_single_level(
            env, config,
            num_episodes=config.get("training", {}).get("total_episodes", 500) // 2,
            device=device,
        )
        single_results = evaluate_agent(env, single_agent, num_episodes=num_eval)
        all_results["Single-DRL"] = single_results
        single_agent.save(str(checkpoint_dir / "single_level_agent.pt"))

    # ══════════════════════════════════════════
    # STAGE 3: Hierarchical DRL
    # ══════════════════════════════════════════
    if 3 in stages:
        logger.info("\n" + "═" * 60)
        logger.info("  STAGE 3: Hierarchical DRL (A2C + PPO)")
        logger.info("═" * 60)

        hier_agent = train_hierarchical(
            env, config,
            num_episodes=config.get("training", {}).get("total_episodes", 500),
            device=device,
        )
        hier_results = evaluate_agent(env, hier_agent, num_episodes=num_eval)
        all_results["HRL (no meta)"] = hier_results
        hier_agent.save(str(checkpoint_dir / "hierarchical_agent"))

    # ══════════════════════════════════════════
    # STAGE 4: Meta-Learning (Reptile)
    # ══════════════════════════════════════════
    if 4 in stages:
        logger.info("\n" + "═" * 60)
        logger.info("  STAGE 4: Meta-Learning (Reptile + HRL)")
        logger.info("═" * 60)

        meta_learner, meta_history = run_meta_training(config)

        # Evaluate meta-learned agent
        meta_results = evaluate_agent(env, meta_learner.agent, num_episodes=num_eval)
        all_results["GreenCloudRL"] = meta_results

        # Generate adaptation curves
        train_tasks, test_tasks = create_meta_tasks(config)
        if test_tasks:
            meta_curve = meta_learner.measure_adaptation_curve(test_tasks[0], num_episodes=20)

            scratch_agent = LowLevelA2C(
                state_dim=env.observation_space.shape[0],
                action_dim=env.action_space.n,
                device=device,
            )
            scratch_meta = ReptileMetaLearner(agent=scratch_agent, env=env, device=device)
            scratch_curve = scratch_meta.measure_adaptation_curve(test_tasks[0], num_episodes=20)

            adaptation_curves = {
                "GreenCloudRL": meta_curve,
                "From Scratch": scratch_curve,
            }
            plot_adaptation_curves(
                adaptation_curves,
                str(figures_dir / "adaptation_curves.png"),
            )

    # ══════════════════════════════════════════
    # STAGE 5: Generate All Plots & Tables
    # ══════════════════════════════════════════
    if 5 in stages:
        logger.info("\n" + "═" * 60)
        logger.info("  STAGE 5: Generating Results & Figures")
        logger.info("═" * 60)

        if all_results:
            plot_metric_comparison(
                all_results, "rewards", "Episode Reward",
                str(figures_dir / "reward_comparison.png"),
                lower_is_better=False,
            )
            plot_metric_comparison(
                all_results, "energy_kwh", "Energy (kWh)",
                str(figures_dir / "energy_comparison.png"),
            )
            plot_metric_comparison(
                all_results, "sla_violation_rates", "SLA Violation Rate",
                str(figures_dir / "sla_comparison.png"),
            )
            plot_energy_breakdown(all_results, str(figures_dir / "energy_breakdown.png"))
            plot_ablation_study(all_results, str(figures_dir / "ablation_study.png"))
            generate_results_table(all_results, str(tables_dir / "main_results.csv"))
        else:
            logger.warning("No results to plot. Run stages 1-4 first.")

    # ══════════════════════════════════════════
    # STAGE 6: Explainability Analysis
    # ══════════════════════════════════════════
    if 6 in stages:
        logger.info("\n" + "═" * 60)
        logger.info("  STAGE 6: Explainability (SHAP Analysis)")
        logger.info("═" * 60)

        # Load best agent
        agent_path = checkpoint_dir / "meta_learned_agent.pt"
        agent = LowLevelA2C(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n,
            device=device,
        )
        if agent_path.exists():
            agent.load(str(agent_path))
            logger.info("Loaded meta-learned agent for explainability analysis")
        else:
            logger.info("No saved agent found, using untrained agent for demo")

        explainer = SHAPExplainer(agent, env)
        analysis = explainer.run_full_analysis(save_dir=str(figures_dir))

        # Save explanations
        with open(str(tables_dir / "sample_explanations.txt"), "w") as f:
            for i, exp in enumerate(analysis["explanations"]):
                f.write(f"\n{'='*60}\nDecision {i+1}\n{'='*60}\n{exp}\n")

    # ══════════════════════════════════════════
    logger.info("\n" + "═" * 60)
    logger.info("  PIPELINE COMPLETE")
    logger.info("═" * 60)
    logger.info(f"Checkpoints: {checkpoint_dir}/")
    logger.info(f"Figures: {figures_dir}/")
    logger.info(f"Tables: {tables_dir}/")


def main():
    print_banner()

    parser = argparse.ArgumentParser(description="GreenCloudRL - Full Pipeline")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Config file path")
    parser.add_argument("--stage", type=int, nargs="+", default=None,
                        help="Stages to run (1-6). Default: all stages.")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto/cuda/cpu")
    args = parser.parse_args()

    # Load config
    config_path = os.path.join(PROJECT_ROOT, args.config)
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded config from {config_path}")
    else:
        config = {}
        logger.warning(f"Config not found at {config_path}, using defaults")

    run_full_pipeline(config, stages=args.stage)


if __name__ == "__main__":
    main()
