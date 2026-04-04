"""
GreenCloudRL - Cloud Scheduling Environment
Gymnasium-compatible discrete-event simulation environment for cloud task scheduling.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import simpy
from typing import List, Optional, Tuple, Dict
from copy import deepcopy

from .server import Server, VirtualMachine, ServerStatus
from .task import Task, TaskStatus, TaskType
from .energy_model import EnergyModel
from .sla_tracker import SLATracker
from .workload_generator import WorkloadGenerator


class CloudSchedulingEnv(gym.Env):
    """
    Hierarchical cloud scheduling environment.
    
    Low-level action: Assign task to a specific VM (discrete).
    High-level action: Server power management + DVFS (discrete).
    
    The environment steps through tasks one at a time.
    At each step, the agent selects which VM to assign the current task to.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        num_servers: int = 10,
        vms_per_server: int = 5,
        max_queue_size: int = 500,
        alpha: float = 0.4,
        beta: float = 0.4,
        gamma: float = 0.2,
        sla_penalty_factor: float = 10.0,
        power_idle: float = 120.0,
        power_max: float = 200.0,
        boot_delay: float = 30.0,
        workload: Optional[List[Task]] = None,
        seed: int = 42,
    ):
        super().__init__()

        self.num_servers = num_servers
        self.vms_per_server = vms_per_server
        self.total_vms = num_servers * vms_per_server
        self.max_queue_size = max_queue_size

        # Reward weights
        self.alpha = alpha
        self.beta = beta
        self.gamma_weight = gamma
        self.sla_penalty_factor = sla_penalty_factor

        # Server config
        self.power_idle = power_idle
        self.power_max = power_max
        self.boot_delay = boot_delay

        # Seed
        self.seed_val = seed
        self.rng = np.random.default_rng(seed)

        # Workload
        self._base_workload = workload
        self.workload_gen = WorkloadGenerator(seed=seed)

        # ── Action & Observation Spaces ──
        # Low-level: choose which VM to assign the task to
        self.action_space = spaces.Discrete(self.total_vms)

        # Observation: VM features + task features + high-level context
        vm_features = self.total_vms * 4     # 4 features per VM
        task_features = 10                    # task vector size
        high_level_ctx = 8                    # high-level embedding
        cluster_features = 6                  # cluster-level stats
        self.obs_dim = vm_features + task_features + high_level_ctx + cluster_features
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32
        )

        # High-level action space (for external high-level agent)
        # 0 = no change, 1..N = turn off server i, N+1..2N = turn on server i
        self.high_level_action_space = spaces.Discrete(1 + 2 * num_servers + len([0.6, 0.8, 1.0]))

        # Initialize
        self.servers: List[Server] = []
        self.energy_model = EnergyModel(power_idle=power_idle, power_max=power_max)
        self.sla_tracker = SLATracker()
        self.current_task_idx = 0
        self.current_time = 0.0
        self.tasks: List[Task] = []
        self.completed_tasks: List[Task] = []
        self.high_level_embedding = np.zeros(8, dtype=np.float32)

        # Episode metrics
        self.episode_reward = 0.0
        self.total_makespan = 0.0
        self.step_count = 0

    def set_workload(self, workload: List[Task]):
        """Set workload for meta-learning task switching."""
        self._base_workload = workload

    def set_high_level_embedding(self, embedding: np.ndarray):
        """Set high-level policy embedding for hierarchical communication."""
        self.high_level_embedding = np.array(embedding, dtype=np.float32)[:8]

    def _init_servers(self):
        """Initialize servers and VMs."""
        self.servers = []
        for i in range(self.num_servers):
            server = Server(
                server_id=i,
                cpu_capacity=100.0,
                memory_capacity=64.0,
                power_idle=self.power_idle,
                power_max=self.power_max,
                boot_delay=self.boot_delay,
            )
            server.initialize_vms(self.vms_per_server)
            self.servers.append(server)

    def _get_current_task(self) -> Optional[Task]:
        """Get the next task to schedule."""
        if self.current_task_idx < len(self.tasks):
            return self.tasks[self.current_task_idx]
        return None

    def _get_action_mask(self) -> np.ndarray:
        """
        Get valid action mask.
        1 = VM can accept the task, 0 = cannot.
        """
        mask = np.zeros(self.total_vms, dtype=np.float32)
        task = self._get_current_task()
        if task is None:
            return mask

        for s_idx, server in enumerate(self.servers):
            if not server.is_active:
                continue
            for v_idx, vm in enumerate(server.vms):
                flat_idx = s_idx * self.vms_per_server + v_idx
                if vm.can_accept_task(task.cpu_req, task.memory_req):
                    mask[flat_idx] = 1.0

        # If no VM can accept, allow all active VMs (task will queue)
        if mask.sum() == 0:
            for s_idx, server in enumerate(self.servers):
                if server.is_active:
                    for v_idx in range(len(server.vms)):
                        flat_idx = s_idx * self.vms_per_server + v_idx
                        mask[flat_idx] = 1.0

        return mask

    def _get_observation(self) -> np.ndarray:
        """Build observation vector."""
        obs = []

        # VM features (4 per VM)
        for server in self.servers:
            for vm in server.vms:
                if server.is_active:
                    obs.extend(vm.to_feature_vector())
                else:
                    obs.extend([0.0, 0.0, 0.0, 0.0])

        # Task features
        task = self._get_current_task()
        if task is not None:
            obs.extend(task.to_feature_vector())
        else:
            obs.extend([0.0] * 10)

        # High-level context embedding
        obs.extend(self.high_level_embedding.tolist())

        # Cluster-level stats
        active_servers = sum(1 for s in self.servers if s.is_active)
        avg_cpu = np.mean([s.cpu_utilization for s in self.servers if s.is_active]) if active_servers > 0 else 0
        avg_mem = np.mean([s.memory_utilization for s in self.servers if s.is_active]) if active_servers > 0 else 0
        total_queue = sum(s.total_queue_length for s in self.servers)
        remaining_tasks = max(0, len(self.tasks) - self.current_task_idx)

        obs.extend([
            active_servers / self.num_servers,
            avg_cpu,
            avg_mem,
            min(total_queue / (self.total_vms * 10), 1.0),
            min(remaining_tasks / max(len(self.tasks), 1), 1.0),
            min(self.current_time / 3600.0, 1.0),
        ])

        obs = np.array(obs, dtype=np.float32)

        # Pad or truncate to match observation space
        if len(obs) < self.obs_dim:
            obs = np.concatenate([obs, np.zeros(self.obs_dim - len(obs), dtype=np.float32)])
        elif len(obs) > self.obs_dim:
            obs = obs[:self.obs_dim]

        return np.clip(obs, 0.0, 1.0)

    def _compute_reward(self, task: Task, server_idx: int, vm_idx: int) -> float:
        """
        Compute reward for scheduling a task to a specific VM.
        
        r = -(alpha * completion_time + beta * energy + gamma * sla_penalty)
        """
        vm = self.servers[server_idx].vms[vm_idx]

        # Estimated completion time (wait in queue + execution)
        queue_wait = vm.queue_length * 10.0  # rough estimate
        completion_time = queue_wait + task.duration
        normalized_ct = min(completion_time / 600.0, 1.0)

        # Energy impact
        old_util = self.servers[server_idx].cpu_utilization
        new_util = min(old_util + task.cpu_req / self.servers[server_idx].cpu_capacity, 1.0)
        energy_delta = self.energy_model.compute_power(new_util) * task.duration
        normalized_energy = min(energy_delta / 50000.0, 1.0)

        # SLA penalty
        est_finish_time = self.current_time + completion_time
        sla_violation = max(0, est_finish_time - task.deadline) * self.sla_penalty_factor
        normalized_sla = min(sla_violation / 100.0, 1.0)

        # Load balancing bonus (prefer less loaded VMs)
        load_balance_bonus = (1.0 - vm.cpu_utilization) * 0.1

        reward = -(
            self.alpha * normalized_ct
            + self.beta * normalized_energy
            + self.gamma_weight * normalized_sla
        ) + load_balance_bonus

        return reward

    def _process_task_completion(self, dt: float):
        """Process task completions and resource releases."""
        for server in self.servers:
            if not server.is_active:
                continue
            for vm in server.vms:
                completed = []
                for task in vm.running_tasks:
                    remaining = task.duration - (self.current_time - task.start_time)
                    if remaining <= 0:
                        task.status = TaskStatus.COMPLETED
                        task.completion_time = task.start_time + task.duration
                        vm.release(task.cpu_req, task.memory_req)
                        self.sla_tracker.record_completion(task)
                        self.completed_tasks.append(task)
                        completed.append(task)

                for task in completed:
                    vm.running_tasks.remove(task)

                # Start queued tasks
                while vm.task_queue and vm.cpu_available > 0:
                    queued_task = vm.task_queue[0]
                    if vm.can_accept_task(queued_task.cpu_req, queued_task.memory_req):
                        vm.task_queue.pop(0)
                        queued_task.status = TaskStatus.RUNNING
                        queued_task.start_time = self.current_time
                        queued_task.wait_time = self.current_time - queued_task.arrival_time
                        vm.allocate(queued_task.cpu_req, queued_task.memory_req)
                        vm.running_tasks.append(queued_task)
                    else:
                        break

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, dict]:
        """Reset environment for new episode."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.seed_val = seed

        self._init_servers()
        self.energy_model.reset()
        self.sla_tracker.reset()

        # Generate or load workload
        if self._base_workload is not None:
            self.tasks = deepcopy(self._base_workload)
        else:
            self.tasks = self.workload_gen.generate_synthetic(num_tasks=500)

        self.current_task_idx = 0
        self.current_time = self.tasks[0].arrival_time if self.tasks else 0.0
        self.completed_tasks = []
        self.episode_reward = 0.0
        self.total_makespan = 0.0
        self.step_count = 0

        obs = self._get_observation()
        info = {"action_mask": self._get_action_mask()}

        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one scheduling step.
        
        Args:
            action: Flat VM index (server_idx * vms_per_server + vm_idx)
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        self.step_count += 1

        task = self._get_current_task()
        if task is None:
            obs = self._get_observation()
            return obs, 0.0, True, False, self._get_info()

        # Decode action to server/vm
        server_idx = action // self.vms_per_server
        vm_idx = action % self.vms_per_server

        # Clamp to valid range
        server_idx = min(server_idx, self.num_servers - 1)
        vm_idx = min(vm_idx, self.vms_per_server - 1)

        server = self.servers[server_idx]
        vm = server.vms[vm_idx]

        # Advance time to task arrival
        if task.arrival_time > self.current_time:
            dt = task.arrival_time - self.current_time
            self.energy_model.update(self.servers, dt)
            self._process_task_completion(dt)
            self.current_time = task.arrival_time

        # Assign task
        if server.is_active and vm.can_accept_task(task.cpu_req, task.memory_req):
            task.status = TaskStatus.RUNNING
            task.start_time = self.current_time
            task.assigned_server = server_idx
            task.assigned_vm = vm_idx
            vm.allocate(task.cpu_req, task.memory_req)
            vm.running_tasks.append(task)
        elif server.is_active:
            # Queue the task on this VM
            task.assigned_server = server_idx
            task.assigned_vm = vm_idx
            vm.task_queue.append(task)
        else:
            # Server is off, assign to first available VM
            assigned = False
            for s in self.servers:
                if s.is_active:
                    for v in s.vms:
                        if v.can_accept_task(task.cpu_req, task.memory_req):
                            task.status = TaskStatus.RUNNING
                            task.start_time = self.current_time
                            task.assigned_server = s.server_id
                            task.assigned_vm = v.vm_id
                            v.allocate(task.cpu_req, task.memory_req)
                            v.running_tasks.append(task)
                            assigned = True
                            break
                    if assigned:
                        break
            if not assigned:
                # Queue on first active server's first VM
                for s in self.servers:
                    if s.is_active:
                        s.vms[0].task_queue.append(task)
                        task.assigned_server = s.server_id
                        task.assigned_vm = 0
                        break

        # Compute reward
        reward = self._compute_reward(task, task.assigned_server or 0, task.assigned_vm or 0)
        self.episode_reward += reward

        # Advance to next task
        self.current_task_idx += 1

        # Small time step for energy tracking
        self.energy_model.update(self.servers, 1.0)
        self._process_task_completion(1.0)
        self.current_time += 1.0

        # Check termination
        terminated = self.current_task_idx >= len(self.tasks)
        truncated = self.step_count >= 10000

        if terminated:
            # Process remaining tasks
            for _ in range(500):
                self.energy_model.update(self.servers, 1.0)
                self._process_task_completion(1.0)
                self.current_time += 1.0
                if all(
                    vm.queue_length == 0
                    for s in self.servers
                    for vm in s.vms
                ):
                    break
            self.total_makespan = self.current_time

        obs = self._get_observation()
        info = self._get_info()

        return obs, float(reward), terminated, truncated, info

    def apply_high_level_action(self, action: int):
        """
        Apply high-level server management action.
        
        Actions:
            0: No change
            1..N: Turn off server (action-1)
            N+1..2N: Turn on server (action-N-1)
            2N+1..2N+3: Set DVFS level for all active servers
        """
        N = self.num_servers
        if action == 0:
            return
        elif 1 <= action <= N:
            server_idx = action - 1
            if self.servers[server_idx].is_active:
                self.servers[server_idx].status = ServerStatus.OFF
        elif N + 1 <= action <= 2 * N:
            server_idx = action - N - 1
            if self.servers[server_idx].status == ServerStatus.OFF:
                self.servers[server_idx].status = ServerStatus.BOOTING
                self.servers[server_idx].boot_start_time = self.current_time
        elif action > 2 * N:
            dvfs_idx = min(action - 2 * N - 1, 2)
            dvfs_levels = [0.6, 0.8, 1.0]
            for s in self.servers:
                if s.is_active:
                    s.current_dvfs = dvfs_levels[dvfs_idx]

    def _get_info(self) -> dict:
        """Get current episode information."""
        return {
            "action_mask": self._get_action_mask(),
            "step": self.step_count,
            "current_time": self.current_time,
            "tasks_scheduled": self.current_task_idx,
            "tasks_total": len(self.tasks),
            "tasks_completed": len(self.completed_tasks),
            "episode_reward": self.episode_reward,
            "makespan": self.total_makespan,
            "energy": self.energy_model.get_stats(),
            "sla": self.sla_tracker.get_stats(),
            "avg_cpu_util": np.mean([s.cpu_utilization for s in self.servers if s.is_active]) if any(s.is_active for s in self.servers) else 0,
            "active_servers": sum(1 for s in self.servers if s.is_active),
        }

    def get_high_level_observation(self) -> np.ndarray:
        """Get aggregated observation for the high-level agent."""
        active = sum(1 for s in self.servers if s.is_active)
        total_cpu = np.mean([s.cpu_utilization for s in self.servers if s.is_active]) if active > 0 else 0
        total_mem = np.mean([s.memory_utilization for s in self.servers if s.is_active]) if active > 0 else 0
        total_queue = sum(s.total_queue_length for s in self.servers)
        remaining = max(0, len(self.tasks) - self.current_task_idx)
        total_power = self.energy_model.compute_datacenter_power(self.servers)

        obs = np.array([
            active / self.num_servers,
            total_cpu,
            total_mem,
            min(total_queue / (self.total_vms * 10), 1.0),
            min(remaining / max(len(self.tasks), 1), 1.0),
            total_power / (self.power_max * self.num_servers),
            self.sla_tracker.violation_rate,
            min(self.current_time / 3600.0, 1.0),
            # Time-of-day encoding (for diurnal patterns)
            np.sin(2 * np.pi * self.current_time / 86400),
            np.cos(2 * np.pi * self.current_time / 86400),
            # Arrival rate estimate
            min(self.current_task_idx / max(self.current_time, 1.0) / 20.0, 1.0),
            0.5,  # placeholder for energy price
        ], dtype=np.float32)

        return obs


# ── Init file ──
