"""
GreenCloudRL - Workload Generator
Generates task arrivals from real traces or synthetic distributions.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional, Dict
from .task import Task, TaskType


class WorkloadGenerator:
    """
    Generates workloads for the cloud simulator.
    Supports both synthetic generation and real trace replay.
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def generate_synthetic(
        self,
        num_tasks: int = 1000,
        arrival_rate: float = 5.0,
        cpu_range: tuple = (1.0, 30.0),
        memory_range: tuple = (0.5, 8.0),
        duration_range: tuple = (5.0, 120.0),
        deadline_slack: tuple = (1.2, 3.0),
        task_type_probs: Optional[Dict] = None,
    ) -> List[Task]:
        """
        Generate synthetic workload with Poisson arrivals.
        
        Args:
            num_tasks: Number of tasks to generate
            arrival_rate: Average tasks per second
            cpu_range: (min, max) CPU requirement
            memory_range: (min, max) memory requirement in GB
            duration_range: (min, max) task duration in seconds
            deadline_slack: (min, max) multiplier for deadline = duration * slack
            task_type_probs: Probability of each task type
            
        Returns:
            List of Task objects sorted by arrival time
        """
        if task_type_probs is None:
            task_type_probs = {
                TaskType.COMPUTE: 0.35,
                TaskType.MEMORY: 0.25,
                TaskType.IO: 0.20,
                TaskType.MIXED: 0.20,
            }

        # Generate inter-arrival times (exponential distribution)
        inter_arrivals = self.rng.exponential(1.0 / arrival_rate, size=num_tasks)
        arrival_times = np.cumsum(inter_arrivals)

        # Task type distribution
        types = list(task_type_probs.keys())
        probs = list(task_type_probs.values())

        tasks = []
        for i in range(num_tasks):
            task_type = self.rng.choice(types, p=probs)

            # Adjust resource requirements based on task type
            if task_type == TaskType.COMPUTE:
                cpu = self.rng.uniform(cpu_range[0] * 1.5, cpu_range[1])
                mem = self.rng.uniform(memory_range[0], memory_range[1] * 0.5)
            elif task_type == TaskType.MEMORY:
                cpu = self.rng.uniform(cpu_range[0], cpu_range[1] * 0.5)
                mem = self.rng.uniform(memory_range[0] * 1.5, memory_range[1])
            elif task_type == TaskType.IO:
                cpu = self.rng.uniform(cpu_range[0], cpu_range[1] * 0.6)
                mem = self.rng.uniform(memory_range[0], memory_range[1] * 0.6)
            else:  # MIXED
                cpu = self.rng.uniform(*cpu_range)
                mem = self.rng.uniform(*memory_range)

            duration = self.rng.uniform(*duration_range)
            slack = self.rng.uniform(*deadline_slack)
            deadline = arrival_times[i] + duration * slack

            tasks.append(Task(
                task_id=i,
                arrival_time=float(arrival_times[i]),
                cpu_req=float(cpu),
                memory_req=float(mem),
                disk_req=float(self.rng.uniform(0, 50)),
                net_req=float(self.rng.uniform(0, 2)),
                duration=float(duration),
                deadline=float(deadline),
                task_type=task_type,
                priority=int(self.rng.choice([1, 2, 3], p=[0.6, 0.3, 0.1])),
            ))

        return tasks

    def generate_bursty(
        self,
        num_tasks: int = 1000,
        base_rate: float = 3.0,
        burst_rate: float = 20.0,
        burst_prob: float = 0.1,
        burst_duration: float = 10.0,
        **kwargs,
    ) -> List[Task]:
        """
        Generate workload with bursty arrival patterns.
        Useful for stress-testing the scheduler.
        """
        tasks = []
        current_time = 0.0
        task_id = 0
        in_burst = False
        burst_end = 0.0

        while task_id < num_tasks:
            # Check for burst
            if not in_burst and self.rng.random() < burst_prob:
                in_burst = True
                burst_end = current_time + burst_duration

            if in_burst and current_time > burst_end:
                in_burst = False

            rate = burst_rate if in_burst else base_rate
            inter_arrival = self.rng.exponential(1.0 / rate)
            current_time += inter_arrival

            cpu = self.rng.uniform(1.0, 30.0)
            mem = self.rng.uniform(0.5, 8.0)
            duration = self.rng.uniform(5.0, 120.0)
            slack = self.rng.uniform(1.2, 3.0)

            tasks.append(Task(
                task_id=task_id,
                arrival_time=current_time,
                cpu_req=cpu,
                memory_req=mem,
                duration=duration,
                deadline=current_time + duration * slack,
                task_type=self.rng.choice(list(TaskType)),
            ))
            task_id += 1

        return tasks

    def load_processed_trace(self, filepath: str, num_tasks: int = 5000) -> List[Task]:
        """
        Load a PROCESSED trace file (from data/preprocessing.py).
        Supports both .csv and .npz formats.
        
        This is the recommended way to load real data after preprocessing:
            gen = WorkloadGenerator()
            tasks = gen.load_processed_trace("data/processed/google_trace_processed.csv")
        """
        fpath = Path(filepath)
        if not fpath.exists():
            raise FileNotFoundError(
                f"Processed trace not found: {filepath}\n"
                f"Run: python data/preprocessing.py --dataset all"
            )

        if fpath.suffix == ".npz":
            data = np.load(fpath)
            n = min(len(data["arrival_time"]), num_tasks)
            tasks = []
            type_list = list(TaskType)

            for i in range(n):
                tasks.append(Task(
                    task_id=i,
                    arrival_time=float(data["arrival_time"][i]),
                    cpu_req=float(np.clip(data["cpu_request"][i], 1.0, 100.0)),
                    memory_req=float(np.clip(data["memory_request"][i], 0.1, 64.0)),
                    disk_req=float(data["disk_request"][i]) if "disk_request" in data else 0.0,
                    duration=float(np.clip(data["duration"][i], 1.0, 600.0)),
                    deadline=float(data["deadline"][i]),
                    task_type=self.rng.choice(type_list),
                    priority=int(data["priority"][i]) if "priority" in data else 1,
                ))
            return tasks

        elif fpath.suffix == ".csv":
            df = pd.read_csv(fpath, nrows=num_tasks)
            type_map = {"compute": TaskType.COMPUTE, "memory": TaskType.MEMORY,
                        "io": TaskType.IO, "mixed": TaskType.MIXED}
            tasks = []
            for i, row in df.iterrows():
                tasks.append(Task(
                    task_id=int(row.get("task_id", i)),
                    arrival_time=float(row["arrival_time"]),
                    cpu_req=float(np.clip(row["cpu_request"], 1.0, 100.0)),
                    memory_req=float(np.clip(row["memory_request"], 0.1, 64.0)),
                    disk_req=float(row.get("disk_request", 0)),
                    duration=float(np.clip(row["duration"], 1.0, 600.0)),
                    deadline=float(row["deadline"]),
                    task_type=type_map.get(str(row.get("task_type", "mixed")), TaskType.MIXED),
                    priority=int(row.get("our_priority", 1)),
                ))
            return tasks
        else:
            raise ValueError(f"Unsupported file format: {fpath.suffix}. Use .csv or .npz")

    def load_google_trace(self, filepath: str, num_tasks: int = 5000) -> List[Task]:
        """Load Google trace (auto-detects raw vs processed)."""
        return self.load_processed_trace(filepath, num_tasks)

    def create_real_meta_tasks(
        self,
        processed_dir: str = "data/processed",
        tasks_per_window: int = 500,
    ) -> List[List[Task]]:
        """
        Create meta-training tasks from REAL processed traces.
        
        Splits each processed trace into time windows for meta-learning.
        Use this instead of create_meta_tasks() for real data experiments.
        """
        processed = Path(processed_dir)
        meta_tasks = []

        for trace_file in sorted(processed.glob("*_processed.csv")):
            dataset_name = trace_file.stem.replace("_processed", "")
            try:
                all_tasks = self.load_processed_trace(str(trace_file), num_tasks=50000)
            except Exception as e:
                print(f"  Warning: Could not load {trace_file}: {e}")
                continue

            # Split into windows of tasks_per_window
            for start in range(0, len(all_tasks), tasks_per_window):
                window = all_tasks[start:start + tasks_per_window]
                if len(window) >= tasks_per_window // 2:  # At least half-full windows
                    # Re-normalize arrival times within window
                    if window:
                        min_t = window[0].arrival_time
                        for t in window:
                            t.arrival_time -= min_t
                            t.deadline -= min_t
                    meta_tasks.append(window)
                    print(f"  Created window from {dataset_name}: {len(window)} tasks")

        print(f"\nTotal meta-training distributions: {len(meta_tasks)}")
        return meta_tasks

    def create_meta_tasks(
        self,
        num_distributions: int = 7,
        tasks_per_distribution: int = 500,
    ) -> List[List[Task]]:
        """
        Create diverse workload distributions for meta-learning.
        Each distribution represents a different workload pattern.
        """
        meta_tasks = []

        configs = [
            # CPU-heavy workload
            {"cpu_range": (20, 90), "memory_range": (0.5, 4), "arrival_rate": 3.0,
             "duration_range": (10, 60)},
            # Memory-heavy workload
            {"cpu_range": (1, 20), "memory_range": (4, 60), "arrival_rate": 4.0,
             "duration_range": (15, 90)},
            # High-throughput short tasks
            {"cpu_range": (1, 15), "memory_range": (0.5, 4), "arrival_rate": 15.0,
             "duration_range": (2, 20)},
            # Low-throughput long tasks
            {"cpu_range": (10, 50), "memory_range": (2, 16), "arrival_rate": 1.0,
             "duration_range": (60, 300)},
            # Mixed balanced
            {"cpu_range": (5, 40), "memory_range": (1, 8), "arrival_rate": 5.0,
             "duration_range": (10, 120)},
            # Bursty pattern
            {"cpu_range": (1, 30), "memory_range": (0.5, 8), "arrival_rate": 5.0,
             "duration_range": (5, 60)},
            # Deadline-tight
            {"cpu_range": (5, 30), "memory_range": (1, 8), "arrival_rate": 8.0,
             "duration_range": (10, 90), "deadline_slack": (1.05, 1.5)},
        ]

        for i, cfg in enumerate(configs[:num_distributions]):
            if i == 5:  # Bursty
                tasks = self.generate_bursty(
                    num_tasks=tasks_per_distribution,
                    base_rate=cfg["arrival_rate"],
                    burst_rate=cfg["arrival_rate"] * 4,
                )
            else:
                tasks = self.generate_synthetic(
                    num_tasks=tasks_per_distribution, **cfg
                )
            meta_tasks.append(tasks)

        return meta_tasks
