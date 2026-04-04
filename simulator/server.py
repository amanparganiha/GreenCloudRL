"""
GreenCloudRL - Server & VM Models
Physical server and virtual machine resource management.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional
import numpy as np


class ServerStatus(Enum):
    OFF = "off"
    BOOTING = "booting"
    ON = "on"
    SHUTTING_DOWN = "shutting_down"


@dataclass
class VirtualMachine:
    """A virtual machine running on a physical server."""

    vm_id: int
    server_id: int
    cpu_capacity: float        # Allocated CPU capacity
    memory_capacity: float     # Allocated memory (GB)

    # Current usage
    cpu_used: float = 0.0
    memory_used: float = 0.0
    running_tasks: list = field(default_factory=list)
    task_queue: list = field(default_factory=list)

    @property
    def cpu_utilization(self) -> float:
        """CPU utilization as fraction [0, 1]."""
        if self.cpu_capacity <= 0:
            return 0.0
        return min(self.cpu_used / self.cpu_capacity, 1.0)

    @property
    def memory_utilization(self) -> float:
        """Memory utilization as fraction [0, 1]."""
        if self.memory_capacity <= 0:
            return 0.0
        return min(self.memory_used / self.memory_capacity, 1.0)

    @property
    def cpu_available(self) -> float:
        return max(0.0, self.cpu_capacity - self.cpu_used)

    @property
    def memory_available(self) -> float:
        return max(0.0, self.memory_capacity - self.memory_used)

    @property
    def queue_length(self) -> int:
        return len(self.task_queue) + len(self.running_tasks)

    @property
    def is_idle(self) -> bool:
        return len(self.running_tasks) == 0 and len(self.task_queue) == 0

    def can_accept_task(self, cpu_req: float, mem_req: float) -> bool:
        """Check if VM has enough resources for a task."""
        return self.cpu_available >= cpu_req and self.memory_available >= mem_req

    def allocate(self, cpu: float, memory: float):
        """Reserve resources for a task."""
        self.cpu_used += cpu
        self.memory_used += memory

    def release(self, cpu: float, memory: float):
        """Free resources when a task completes."""
        self.cpu_used = max(0.0, self.cpu_used - cpu)
        self.memory_used = max(0.0, self.memory_used - memory)

    def to_feature_vector(self) -> list:
        """Feature vector for RL agent observation."""
        return [
            self.cpu_utilization,
            self.memory_utilization,
            min(self.queue_length / 20.0, 1.0),
            1.0 if self.is_idle else 0.0,
        ]


@dataclass
class Server:
    """A physical server hosting multiple VMs."""

    server_id: int
    cpu_capacity: float = 100.0
    memory_capacity: float = 64.0
    disk_capacity: float = 500.0
    bandwidth_capacity: float = 10.0
    power_idle: float = 120.0
    power_max: float = 200.0
    boot_delay: float = 30.0
    dvfs_levels: list = field(default_factory=lambda: [0.6, 0.8, 1.0])

    # State
    status: ServerStatus = ServerStatus.ON
    current_dvfs: float = 1.0
    vms: List[VirtualMachine] = field(default_factory=list)
    boot_start_time: Optional[float] = None

    # Cumulative tracking
    total_energy: float = 0.0
    uptime: float = 0.0

    def initialize_vms(self, num_vms: int):
        """Create VMs with equal resource share."""
        cpu_per_vm = self.cpu_capacity / num_vms
        mem_per_vm = self.memory_capacity / num_vms
        self.vms = [
            VirtualMachine(
                vm_id=i,
                server_id=self.server_id,
                cpu_capacity=cpu_per_vm,
                memory_capacity=mem_per_vm,
            )
            for i in range(num_vms)
        ]

    @property
    def is_active(self) -> bool:
        return self.status == ServerStatus.ON

    @property
    def cpu_utilization(self) -> float:
        """Overall server CPU utilization."""
        if not self.vms or not self.is_active:
            return 0.0
        total_used = sum(vm.cpu_used for vm in self.vms)
        return min(total_used / self.cpu_capacity, 1.0)

    @property
    def memory_utilization(self) -> float:
        """Overall server memory utilization."""
        if not self.vms or not self.is_active:
            return 0.0
        total_used = sum(vm.memory_used for vm in self.vms)
        return min(total_used / self.memory_capacity, 1.0)

    @property
    def num_running_tasks(self) -> int:
        return sum(len(vm.running_tasks) for vm in self.vms)

    @property
    def total_queue_length(self) -> int:
        return sum(vm.queue_length for vm in self.vms)

    def current_power(self) -> float:
        """Power consumption using linear model: P = P_idle + (P_max - P_idle) * u * dvfs."""
        if not self.is_active:
            return 0.0
        u = self.cpu_utilization
        return self.power_idle + (self.power_max - self.power_idle) * u * self.current_dvfs

    def energy_for_interval(self, dt: float) -> float:
        """Energy consumed over time interval dt (Joules = Watts * seconds)."""
        return self.current_power() * dt

    def get_available_vms(self, cpu_req: float, mem_req: float) -> List[int]:
        """Return indices of VMs that can accept a task with given requirements."""
        if not self.is_active:
            return []
        return [
            i for i, vm in enumerate(self.vms)
            if vm.can_accept_task(cpu_req, mem_req)
        ]

    def to_feature_vector(self) -> list:
        """Aggregated server features for high-level agent."""
        return [
            self.cpu_utilization,
            self.memory_utilization,
            1.0 if self.is_active else 0.0,
            self.current_power() / self.power_max,
            self.total_queue_length / max(len(self.vms) * 20, 1),
            self.current_dvfs,
        ]
