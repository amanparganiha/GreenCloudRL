"""
GreenCloudRL - Baseline Schedulers
Simple scheduling algorithms for comparison against learned policies.
"""

import numpy as np
from typing import Optional


class RandomScheduler:
    """Randomly assigns tasks to available VMs."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.name = "Random"

    def select_action(self, obs: np.ndarray, action_mask: Optional[np.ndarray] = None) -> int:
        if action_mask is not None:
            valid = np.where(action_mask > 0)[0]
            if len(valid) > 0:
                return int(self.rng.choice(valid))
        return int(self.rng.integers(0, 50))  # Default: 10 servers * 5 VMs


class RoundRobinScheduler:
    """Assigns tasks to VMs in round-robin order."""

    def __init__(self, total_vms: int = 50):
        self.total_vms = total_vms
        self.current_vm = 0
        self.name = "Round-Robin"

    def select_action(self, obs: np.ndarray, action_mask: Optional[np.ndarray] = None) -> int:
        if action_mask is not None:
            for _ in range(self.total_vms):
                if action_mask[self.current_vm] > 0:
                    action = self.current_vm
                    self.current_vm = (self.current_vm + 1) % self.total_vms
                    return action
                self.current_vm = (self.current_vm + 1) % self.total_vms

        action = self.current_vm
        self.current_vm = (self.current_vm + 1) % self.total_vms
        return action


class FCFSScheduler:
    """First Come First Served - assigns to first available VM."""

    def __init__(self):
        self.name = "FCFS"

    def select_action(self, obs: np.ndarray, action_mask: Optional[np.ndarray] = None) -> int:
        if action_mask is not None:
            valid = np.where(action_mask > 0)[0]
            if len(valid) > 0:
                return int(valid[0])
        return 0


class LeastLoadedScheduler:
    """Assigns task to the VM with lowest CPU utilization."""

    def __init__(self, vms_per_server: int = 5):
        self.vms_per_server = vms_per_server
        self.name = "Least-Loaded"

    def select_action(self, obs: np.ndarray, action_mask: Optional[np.ndarray] = None) -> int:
        # Extract CPU utilizations from observation (first feature of each VM block)
        # Each VM has 4 features: cpu_util, mem_util, queue_len, idle
        total_vms = len(obs) // 10  # approximate
        cpu_utils = []

        for i in range(0, min(len(obs), 200), 4):
            cpu_utils.append(obs[i])

        if action_mask is not None:
            valid = np.where(action_mask > 0)[0]
            if len(valid) > 0:
                # Find valid VM with lowest CPU utilization
                best_action = valid[0]
                best_util = float("inf")
                for v in valid:
                    if v < len(cpu_utils) and cpu_utils[v] < best_util:
                        best_util = cpu_utils[v]
                        best_action = v
                return int(best_action)

        return int(np.argmin(cpu_utils)) if cpu_utils else 0


class ShortestJobFirstScheduler:
    """Assigns shortest tasks first to VMs with shortest queues."""

    def __init__(self, vms_per_server: int = 5):
        self.vms_per_server = vms_per_server
        self.name = "SJF"

    def select_action(self, obs: np.ndarray, action_mask: Optional[np.ndarray] = None) -> int:
        # Extract queue lengths from observation (3rd feature of each VM block)
        queue_lengths = []
        for i in range(2, min(len(obs), 200), 4):
            queue_lengths.append(obs[i])

        if action_mask is not None:
            valid = np.where(action_mask > 0)[0]
            if len(valid) > 0:
                best_action = valid[0]
                best_queue = float("inf")
                for v in valid:
                    if v < len(queue_lengths) and queue_lengths[v] < best_queue:
                        best_queue = queue_lengths[v]
                        best_action = v
                return int(best_action)

        return int(np.argmin(queue_lengths)) if queue_lengths else 0


def get_all_baselines(total_vms: int = 50, vms_per_server: int = 5, seed: int = 42):
    """Get all baseline schedulers."""
    return [
        RandomScheduler(seed=seed),
        RoundRobinScheduler(total_vms=total_vms),
        FCFSScheduler(),
        LeastLoadedScheduler(vms_per_server=vms_per_server),
        ShortestJobFirstScheduler(vms_per_server=vms_per_server),
    ]
