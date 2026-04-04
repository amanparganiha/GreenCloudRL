"""
GreenCloudRL - Evaluation & Visualization Pipeline
Run all experiments and generate publication-quality figures.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Publication-quality plot settings ──
plt.rcParams.update({
    "font.size": 12,
    "font.family": "serif",
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

COLORS = {
    "GreenCloudRL": "#2E86C1",
    "HRL (no meta)": "#27AE60",
    "Single-DRL": "#E67E22",
    "Least-Loaded": "#8E44AD",
    "SJF": "#E74C3C",
    "Round-Robin": "#95A5A6",
    "FCFS": "#BDC3C7",
    "Random": "#D5D8DC",
}


def plot_training_curves(
    reward_histories: Dict[str, List[float]],
    save_path: str,
    title: str = "Training Reward Curves",
    window: int = 20,
):
    """Plot smoothed training reward curves for multiple methods."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for name, rewards in reward_histories.items():
        color = COLORS.get(name, "#333333")
        # Smooth with moving average
        if len(rewards) > window:
            smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
        else:
            smoothed = rewards
        ax.plot(smoothed, label=name, color=color, linewidth=2)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Reward")
    ax.set_title(title, fontweight="bold")
    ax.legend(loc="lower right", frameon=True, fancybox=True)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Saved: {save_path}")


def plot_metric_comparison(
    results: Dict[str, Dict],
    metric: str,
    ylabel: str,
    save_path: str,
    title: str = "",
    lower_is_better: bool = True,
):
    """Bar chart comparing a metric across all methods."""
    names = list(results.keys())
    means = [results[n].get(f"{metric}_mean", 0) for n in names]
    stds = [results[n].get(f"{metric}_std", 0) for n in names]
    colors = [COLORS.get(n, "#333333") for n in names]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(names))
    bars = ax.bar(x, means, yerr=stds, color=colors, capsize=5, alpha=0.85, edgecolor="white", linewidth=1.5)

    # Highlight best
    best_idx = np.argmin(means) if lower_is_better else np.argmax(means)
    bars[best_idx].set_edgecolor("#2C3E50")
    bars[best_idx].set_linewidth(3)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title or f"{ylabel} Comparison", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Saved: {save_path}")


def plot_adaptation_curves(
    curves: Dict[str, List[float]],
    save_path: str,
    title: str = "Adaptation Speed on Unseen Workload",
):
    """Plot adaptation speed comparison."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for name, rewards in curves.items():
        color = COLORS.get(name, "#333333")
        ax.plot(range(1, len(rewards) + 1), rewards, marker="o", label=name,
                color=color, linewidth=2, markersize=5)

    # Mark 90% optimal line
    if "GreenCloudRL" in curves:
        converged = np.mean(curves["GreenCloudRL"][-3:])
        ax.axhline(y=0.9 * converged, color="red", linestyle="--", alpha=0.5, label="90% Optimal")

    ax.set_xlabel("Adaptation Episode")
    ax.set_ylabel("Episode Reward")
    ax.set_title(title, fontweight="bold")
    ax.legend(loc="lower right", frameon=True, fancybox=True)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Saved: {save_path}")


def plot_ablation_study(
    results: Dict[str, Dict],
    save_path: str,
):
    """Create ablation study table as a figure."""
    methods = list(results.keys())
    metrics = ["rewards_mean", "energy_kwh_mean", "sla_violation_rates_mean", "makespans_mean"]
    metric_labels = ["Reward", "Energy (kWh)", "SLA Violation %", "Makespan (s)"]

    fig, axes = plt.subplots(1, len(metrics), figsize=(16, 5))

    for ax, metric, label in zip(axes, metrics, metric_labels):
        values = [results[m].get(metric, 0) for m in methods]
        stds = [results[m].get(metric.replace("_mean", "_std"), 0) for m in methods]
        colors_list = [COLORS.get(m, "#333333") for m in methods]

        bars = ax.barh(range(len(methods)), values, xerr=stds, color=colors_list,
                       capsize=3, alpha=0.85, edgecolor="white")
        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels(methods, fontsize=10)
        ax.set_xlabel(label, fontsize=11)
        ax.grid(axis="x", alpha=0.3)

    fig.suptitle("Ablation Study: Component Contributions", fontweight="bold", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Saved: {save_path}")


def generate_results_table(
    all_results: Dict[str, Dict],
    save_path: str,
):
    """Generate a CSV results table for the paper."""
    rows = []
    for method, res in all_results.items():
        rows.append({
            "Method": method,
            "Reward": f"{res.get('rewards_mean', 0):.2f} ± {res.get('rewards_std', 0):.2f}",
            "Energy (kWh)": f"{res.get('energy_kwh_mean', 0):.4f} ± {res.get('energy_kwh_std', 0):.4f}",
            "SLA Violation (%)": f"{res.get('sla_violation_rates_mean', 0)*100:.1f} ± {res.get('sla_violation_rates_std', 0)*100:.1f}",
            "Makespan (s)": f"{res.get('makespans_mean', 0):.1f} ± {res.get('makespans_std', 0):.1f}",
            "Avg Response (s)": f"{res.get('avg_response_times_mean', 0):.1f} ± {res.get('avg_response_times_std', 0):.1f}",
        })

    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False)
    logger.info(f"Results table saved to {save_path}")
    print("\n" + df.to_string(index=False))
    return df


def plot_energy_breakdown(
    results: Dict[str, Dict],
    save_path: str,
):
    """Plot energy consumption breakdown."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    names = list(results.keys())
    energy = [results[n].get("energy_kwh_mean", 0) for n in names]
    sla = [results[n].get("sla_violation_rates_mean", 0) * 100 for n in names]
    colors = [COLORS.get(n, "#333333") for n in names]

    # Energy bar chart
    ax1.bar(range(len(names)), energy, color=colors, alpha=0.85, edgecolor="white")
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names, rotation=35, ha="right", fontsize=10)
    ax1.set_ylabel("Energy Consumption (kWh)")
    ax1.set_title("Energy Comparison", fontweight="bold")
    ax1.grid(axis="y", alpha=0.3)

    # Energy vs SLA scatter
    for i, name in enumerate(names):
        ax2.scatter(energy[i], sla[i], s=150, c=colors[i], label=name, zorder=5, edgecolor="white", linewidth=1.5)
    ax2.set_xlabel("Energy (kWh)")
    ax2.set_ylabel("SLA Violation Rate (%)")
    ax2.set_title("Energy vs SLA Trade-off", fontweight="bold")
    ax2.legend(fontsize=9, loc="upper right")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Saved: {save_path}")


if __name__ == "__main__":
    # Example usage with dummy data
    figures_dir = Path("results/figures")
    tables_dir = Path("results/tables")
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Dummy results for demonstration
    dummy_results = {
        "GreenCloudRL": {"rewards_mean": -15.2, "rewards_std": 2.1, "energy_kwh_mean": 0.045, "energy_kwh_std": 0.003, "sla_violation_rates_mean": 0.03, "sla_violation_rates_std": 0.01, "makespans_mean": 420, "makespans_std": 30, "avg_response_times_mean": 25, "avg_response_times_std": 5},
        "HRL (no meta)": {"rewards_mean": -18.5, "rewards_std": 3.0, "energy_kwh_mean": 0.052, "energy_kwh_std": 0.004, "sla_violation_rates_mean": 0.05, "sla_violation_rates_std": 0.02, "makespans_mean": 480, "makespans_std": 40, "avg_response_times_mean": 30, "avg_response_times_std": 7},
        "Single-DRL": {"rewards_mean": -22.0, "rewards_std": 4.0, "energy_kwh_mean": 0.058, "energy_kwh_std": 0.005, "sla_violation_rates_mean": 0.08, "sla_violation_rates_std": 0.03, "makespans_mean": 520, "makespans_std": 50, "avg_response_times_mean": 35, "avg_response_times_std": 8},
        "Least-Loaded": {"rewards_mean": -28.0, "rewards_std": 3.5, "energy_kwh_mean": 0.062, "energy_kwh_std": 0.004, "sla_violation_rates_mean": 0.12, "sla_violation_rates_std": 0.04, "makespans_mean": 580, "makespans_std": 45, "avg_response_times_mean": 42, "avg_response_times_std": 10},
        "Round-Robin": {"rewards_mean": -35.0, "rewards_std": 5.0, "energy_kwh_mean": 0.070, "energy_kwh_std": 0.006, "sla_violation_rates_mean": 0.18, "sla_violation_rates_std": 0.05, "makespans_mean": 650, "makespans_std": 60, "avg_response_times_mean": 50, "avg_response_times_std": 12},
        "Random": {"rewards_mean": -45.0, "rewards_std": 8.0, "energy_kwh_mean": 0.080, "energy_kwh_std": 0.008, "sla_violation_rates_mean": 0.30, "sla_violation_rates_std": 0.08, "makespans_mean": 800, "makespans_std": 100, "avg_response_times_mean": 65, "avg_response_times_std": 15},
    }

    # Generate all plots
    plot_metric_comparison(dummy_results, "rewards", "Episode Reward", str(figures_dir / "reward_comparison.png"), lower_is_better=False)
    plot_metric_comparison(dummy_results, "energy_kwh", "Energy (kWh)", str(figures_dir / "energy_comparison.png"))
    plot_metric_comparison(dummy_results, "sla_violation_rates", "SLA Violation Rate", str(figures_dir / "sla_comparison.png"))
    plot_energy_breakdown(dummy_results, str(figures_dir / "energy_breakdown.png"))
    plot_ablation_study(dummy_results, str(figures_dir / "ablation_study.png"))
    generate_results_table(dummy_results, str(tables_dir / "main_results.csv"))

    logger.info("\nAll evaluation plots generated!")
