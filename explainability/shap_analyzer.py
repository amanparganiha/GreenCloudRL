"""
GreenCloudRL - Explainability Module
SHAP-based feature importance and natural language explanations.
"""

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


# Feature names for the observation vector
def get_feature_names(num_servers: int = 10, vms_per_server: int = 5) -> List[str]:
    """Generate human-readable feature names for the observation vector."""
    names = []
    for s in range(num_servers):
        for v in range(vms_per_server):
            names.extend([
                f"S{s}_VM{v}_CPU_util",
                f"S{s}_VM{v}_Mem_util",
                f"S{s}_VM{v}_Queue_len",
                f"S{s}_VM{v}_Idle",
            ])
    names.extend([
        "Task_CPU_req", "Task_Mem_req", "Task_Disk_req", "Task_Net_req",
        "Task_Duration", "Task_Deadline_slack",
        "Task_Type_Compute", "Task_Type_Memory", "Task_Type_IO", "Task_Type_Mixed",
    ])
    names.extend([f"HL_Embed_{i}" for i in range(8)])
    names.extend([
        "Cluster_Active_Servers", "Cluster_Avg_CPU", "Cluster_Avg_Mem",
        "Cluster_Queue_Total", "Cluster_Remaining_Tasks", "Cluster_Time",
    ])
    return names


class SHAPExplainer:
    """
    SHAP-based explainability for scheduling decisions.
    
    Computes feature importance and generates natural language explanations
    for why specific scheduling decisions were made.
    """

    def __init__(
        self,
        agent,
        env,
        num_background: int = 100,
        num_explain: int = 200,
        top_k: int = 5,
    ):
        self.agent = agent
        self.env = env
        self.num_background = num_background
        self.num_explain = num_explain
        self.top_k = top_k
        self.feature_names = get_feature_names(env.num_servers, env.vms_per_server)

    def collect_episodes(self, num_episodes: int = 5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Collect state-action-reward tuples from validation episodes."""
        all_states = []
        all_actions = []
        all_rewards = []

        for _ in range(num_episodes):
            obs, info = self.env.reset()
            done = False
            while not done:
                action_mask = info.get("action_mask", None)
                action, _, _ = self.agent.select_action(obs, action_mask)
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                all_states.append(obs)
                all_actions.append(action)
                all_rewards.append(reward)
                obs = next_obs

        return (
            np.array(all_states),
            np.array(all_actions),
            np.array(all_rewards),
        )

    def compute_shap_values(self, states: np.ndarray) -> np.ndarray:
        """
        Compute approximate SHAP values using perturbation-based method.
        (Avoids heavy dependency on shap library for basic functionality)
        """
        try:
            import shap
            return self._compute_shap_kernel(states)
        except ImportError:
            logger.warning("SHAP library not available, using permutation importance")
            return self._compute_permutation_importance(states)

    def _compute_shap_kernel(self, states: np.ndarray) -> np.ndarray:
        """Compute SHAP values using KernelExplainer."""
        import shap

        background = states[:self.num_background]
        explain = states[:self.num_explain]

        def model_fn(x):
            x_t = torch.FloatTensor(x).to(self.agent.device)
            with torch.no_grad():
                dist = self.agent.actor(x_t)
            return dist.probs.cpu().numpy()

        explainer = shap.KernelExplainer(model_fn, background)
        shap_values = explainer.shap_values(explain, nsamples=100)

        # Average across actions for global importance
        if isinstance(shap_values, list):
            shap_values = np.mean(np.abs(np.stack(shap_values)), axis=0)

        return shap_values

    def _compute_permutation_importance(self, states: np.ndarray) -> np.ndarray:
        """Fallback: Compute feature importance via permutation."""
        n_features = states.shape[1]
        n_samples = min(len(states), self.num_explain)
        sample_states = states[:n_samples]

        # Baseline predictions
        baseline_probs = []
        for s in sample_states:
            s_t = torch.FloatTensor(s).unsqueeze(0).to(self.agent.device)
            with torch.no_grad():
                dist = self.agent.actor(s_t)
            baseline_probs.append(dist.probs.cpu().numpy().flatten())
        baseline_probs = np.array(baseline_probs)

        importance = np.zeros((n_samples, n_features))

        for f in range(n_features):
            permuted = sample_states.copy()
            permuted[:, f] = np.random.permutation(permuted[:, f])

            perturbed_probs = []
            for s in permuted:
                s_t = torch.FloatTensor(s).unsqueeze(0).to(self.agent.device)
                with torch.no_grad():
                    dist = self.agent.actor(s_t)
                perturbed_probs.append(dist.probs.cpu().numpy().flatten())
            perturbed_probs = np.array(perturbed_probs)

            # Importance = change in output
            importance[:, f] = np.mean(np.abs(baseline_probs - perturbed_probs), axis=1)

        return importance

    def generate_explanation(
        self,
        state: np.ndarray,
        action: int,
        shap_values: np.ndarray,
    ) -> str:
        """
        Generate a natural language explanation for a scheduling decision.
        """
        # Flatten SHAP values to 1D (one importance per feature)
        if shap_values.ndim > 1:
            shap_values = np.mean(np.abs(shap_values), axis=-1)
        
        shap_values = np.asarray(shap_values).flatten()

        # Trim feature names to match actual state dim
        names = self.feature_names[:len(shap_values)]

        # Get top-k most important features
        top_indices = np.argsort(np.abs(shap_values))[-self.top_k:][::-1]

        server_idx = action // self.env.vms_per_server
        vm_idx = action % self.env.vms_per_server

        explanation = f"Decision: Assigned task to VM-{vm_idx} on Server-{server_idx}\n\n"
        explanation += "Top factors influencing this decision:\n"

        for rank, idx in enumerate(top_indices, 1):
            if idx < len(names):
                name = names[idx]
                value = float(state[idx]) if idx < len(state) else 0.0
                imp = float(shap_values[idx])
                direction = "increased" if imp > 0 else "decreased"

                explanation += (
                    f"  {rank}. {name} = {value:.3f} "
                    f"(importance: {abs(imp):.4f}, {direction} likelihood)\n"
                )

        return explanation

    def plot_global_importance(
        self,
        shap_values: np.ndarray,
        save_path: Optional[str] = None,
    ):
        """Plot global feature importance summary."""
        # Handle 3D SHAP values (samples x features x actions) -> flatten to 2D
        if shap_values.ndim == 3:
            mean_importance = np.mean(np.abs(shap_values), axis=(0, 2))
        elif shap_values.ndim == 2:
            mean_importance = np.mean(np.abs(shap_values), axis=0)
        else:
            mean_importance = np.abs(shap_values)

        names = self.feature_names[:len(mean_importance)]

        # Top 20 features
        top_k = min(20, len(names))
        top_idx = np.argsort(mean_importance)[-top_k:]

        fig, ax = plt.subplots(figsize=(10, 8))
        y_pos = np.arange(top_k)
        ax.barh(y_pos, mean_importance[top_idx], color="#2E86C1", alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([names[i] for i in top_idx], fontsize=9)
        ax.set_xlabel("Mean |SHAP Value|", fontsize=12)
        ax.set_title("GreenCloudRL - Global Feature Importance", fontsize=14, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved importance plot to {save_path}")
        plt.close()

    def run_full_analysis(self, save_dir: str = "results/figures") -> Dict:
        """Run complete explainability analysis."""
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        logger.info("Collecting episodes for explainability analysis...")
        states, actions, rewards = self.collect_episodes(num_episodes=3)

        logger.info("Computing feature importance...")
        shap_values = self.compute_shap_values(states)

        # Global importance plot
        self.plot_global_importance(shap_values, f"{save_dir}/global_importance.png")

        # Generate sample explanations
        explanations = []
        num_samples = min(5, len(states))
        for i in range(num_samples):
            sv = shap_values[i] if i < len(shap_values) else np.zeros(states.shape[1])
            explanation = self.generate_explanation(states[i], actions[i], sv)
            explanations.append(explanation)
            logger.info(f"\n--- Decision {i+1} ---\n{explanation}")

        return {
            "shap_values": shap_values,
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "explanations": explanations,
        }
