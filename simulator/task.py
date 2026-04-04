"""
GreenCloudRL - Task Model
Represents a cloud computing task with resource requirements and deadlines.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class TaskType(Enum):
    COMPUTE = "compute"
    MEMORY = "memory"
    IO = "io"
    MIXED = "mixed"


class TaskStatus(Enum):
    WAITING = "waiting"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"          # SLA violated (missed deadline)


@dataclass
class Task:
    """Represents a single cloud computing task."""

    task_id: int
    arrival_time: float
    cpu_req: float             # CPU requirement (normalized)
    memory_req: float          # Memory requirement (GB)
    disk_req: float = 0.0      # Disk I/O requirement
    net_req: float = 0.0       # Network bandwidth requirement
    duration: float = 10.0     # Expected execution time (seconds)
    deadline: float = 30.0     # Absolute deadline (timestamp)
    task_type: TaskType = TaskType.MIXED
    priority: int = 1          # 1 = normal, higher = more important

    # Runtime tracking
    status: TaskStatus = TaskStatus.WAITING
    start_time: Optional[float] = None
    completion_time: Optional[float] = None
    assigned_server: Optional[int] = None
    assigned_vm: Optional[int] = None
    wait_time: float = 0.0
    energy_consumed: float = 0.0

    @property
    def deadline_slack(self) -> float:
        """Remaining time until deadline from now (requires current time)."""
        if self.start_time is not None:
            return self.deadline - self.start_time
        return self.deadline - self.arrival_time

    @property
    def response_time(self) -> float:
        """Total time from arrival to completion."""
        if self.completion_time is not None:
            return self.completion_time - self.arrival_time
        return float("inf")

    @property
    def is_sla_violated(self) -> bool:
        """Check if task missed its deadline."""
        if self.completion_time is not None:
            return self.completion_time > self.deadline
        return False

    @property
    def sla_violation_time(self) -> float:
        """How much the deadline was exceeded by."""
        if self.completion_time is not None and self.completion_time > self.deadline:
            return self.completion_time - self.deadline
        return 0.0

    def to_feature_vector(self) -> list:
        """Convert task to a normalized feature vector for the agent."""
        type_encoding = {
            TaskType.COMPUTE: [1, 0, 0, 0],
            TaskType.MEMORY: [0, 1, 0, 0],
            TaskType.IO: [0, 0, 1, 0],
            TaskType.MIXED: [0, 0, 0, 1],
        }
        return [
            self.cpu_req / 100.0,
            self.memory_req / 64.0,
            self.disk_req / 500.0,
            self.net_req / 10.0,
            self.duration / 120.0,
            min(self.deadline_slack / 360.0, 1.0),
        ] + type_encoding[self.task_type]
