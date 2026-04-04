"""
GreenCloudRL - SLA Violation Tracker
Monitors and reports Service Level Agreement violations.
"""

from dataclasses import dataclass, field
from typing import List, Dict
import numpy as np


@dataclass
class SLATracker:
    """Tracks SLA violations and generates reports."""

    total_tasks: int = 0
    completed_tasks: int = 0
    violated_tasks: int = 0
    total_violation_time: float = 0.0
    violation_history: List[float] = field(default_factory=list)
    completion_times: List[float] = field(default_factory=list)
    wait_times: List[float] = field(default_factory=list)

    def record_completion(self, task):
        """Record task completion and check SLA."""
        self.total_tasks += 1
        self.completed_tasks += 1
        self.completion_times.append(task.response_time)

        if task.wait_time > 0:
            self.wait_times.append(task.wait_time)

        if task.is_sla_violated:
            self.violated_tasks += 1
            violation = task.sla_violation_time
            self.total_violation_time += violation
            self.violation_history.append(violation)

    @property
    def violation_rate(self) -> float:
        """Percentage of tasks that violated SLA."""
        if self.total_tasks == 0:
            return 0.0
        return self.violated_tasks / self.total_tasks

    @property
    def avg_response_time(self) -> float:
        if not self.completion_times:
            return 0.0
        return np.mean(self.completion_times)

    @property
    def avg_wait_time(self) -> float:
        if not self.wait_times:
            return 0.0
        return np.mean(self.wait_times)

    @property
    def p95_response_time(self) -> float:
        if not self.completion_times:
            return 0.0
        return np.percentile(self.completion_times, 95)

    def get_stats(self) -> Dict:
        return {
            "total_tasks": self.total_tasks,
            "completed_tasks": self.completed_tasks,
            "violated_tasks": self.violated_tasks,
            "violation_rate": self.violation_rate,
            "avg_response_time": self.avg_response_time,
            "avg_wait_time": self.avg_wait_time,
            "p95_response_time": self.p95_response_time,
            "total_violation_time": self.total_violation_time,
        }

    def reset(self):
        self.total_tasks = 0
        self.completed_tasks = 0
        self.violated_tasks = 0
        self.total_violation_time = 0.0
        self.violation_history.clear()
        self.completion_times.clear()
        self.wait_times.clear()
