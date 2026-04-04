"""
GreenCloudRL - Data Preprocessing Pipeline
Converts raw trace files into clean Task objects for the simulator.

Supports:
  1. Google Cluster Trace v2 (2011) - CSV format
  2. Alibaba Cluster Trace (2018) - CSV format
  3. HPC2N - SWF (Standard Workload Format)
  4. NASA iPSC - SWF format

Usage:
    python data/preprocessing.py --dataset google --input data/raw --output data/processed
    python data/preprocessing.py --dataset all --input data/raw --output data/processed
"""

import os
import sys
import gzip
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import json

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════
# GOOGLE CLUSTER TRACE v2 (2011)
# ══════════════════════════════════════════════════════════
# CSV columns for task_events (no header in file):
#  0: timestamp (microseconds)
#  1: missing_info
#  2: job_id
#  3: task_index
#  4: machine_id
#  5: event_type (0=SUBMIT, 1=SCHEDULE, 2=EVICT, 3=FAIL, 4=FINISH, 5=KILL, 6=LOST, 7=UPDATE_PENDING, 8=UPDATE_RUNNING)
#  6: user
#  7: scheduling_class (0-3, higher = more latency-sensitive)
#  8: priority (0-11)
#  9: cpu_request (normalized, 0-1 fraction of machine)
# 10: memory_request (normalized, 0-1 fraction of machine)
# 11: disk_space_request (normalized)
# 12: different_machines_restriction

GOOGLE_TASK_EVENT_COLS = [
    "timestamp", "missing_info", "job_id", "task_index", "machine_id",
    "event_type", "user", "scheduling_class", "priority",
    "cpu_request", "memory_request", "disk_request", "diff_machine"
]

# CSV columns for task_usage (no header):
#  0: start_time
#  1: end_time
#  2: job_id
#  3: task_index
#  4: machine_id
#  5: mean_cpu_usage
#  6: canonical_memory_usage
#  7: assigned_memory_usage
#  8: unmapped_page_cache
#  9: total_page_cache
# 10: max_memory_usage
# 11: mean_disk_io_time
# 12: mean_local_disk_space
# 13: max_cpu_usage
# 14: max_disk_io_time
# 15: cpi  (cycles per instruction)
# 16: mai  (memory accesses per instruction)
# 17: sample_portion
# 18: aggregation_type
# 19: sampled_cpu_usage (only in v2.1 / clusterdata-2011-2)

GOOGLE_TASK_USAGE_COLS = [
    "start_time", "end_time", "job_id", "task_index", "machine_id",
    "mean_cpu_usage", "canonical_mem_usage", "assigned_mem_usage",
    "unmapped_page_cache", "total_page_cache", "max_mem_usage",
    "mean_disk_io", "mean_local_disk", "max_cpu_usage", "max_disk_io",
    "cpi", "mai", "sample_portion", "aggregation_type", "sampled_cpu_usage"
]


def preprocess_google_trace(raw_dir: str, output_dir: str, max_tasks: int = 10000):
    """
    Parse Google Cluster Trace v2 (2011) task_events CSV.
    
    The key insight: We extract SUBMIT events to get task arrivals and resource requests,
    then match with FINISH events to get actual task durations.
    """
    logger.info("=" * 60)
    logger.info("Processing Google Cluster Trace v2 (2011)")
    logger.info("=" * 60)

    # ── Find input files ──
    raw_path = Path(raw_dir)
    event_files = sorted(raw_path.glob("google_task_events*.csv*"))
    usage_files = sorted(raw_path.glob("google_task_usage*.csv*"))

    if not event_files:
        # Try alternate naming
        event_files = sorted(raw_path.glob("task_events*.csv*")) + sorted(raw_path.glob("part-*.csv*"))

    if not event_files:
        logger.error(
            "No Google trace files found!\n"
            "Expected: data/raw/google_task_events_part0.csv or google_task_events_part0.csv.gz\n\n"
            "Download with:\n"
            '  curl -o data/raw/google_task_events_part0.csv.gz '
            '"https://commondatastorage.googleapis.com/clusterdata-2011-2/task_events/part-00000-of-00500.csv.gz"'
        )
        return None

    logger.info(f"Found {len(event_files)} event file(s)")

    # ── Read task events ──
    dfs = []
    for f in event_files[:3]:  # Limit to first 3 files
        logger.info(f"  Reading: {f.name}")
        try:
            if str(f).endswith(".gz"):
                df = pd.read_csv(f, header=None, names=GOOGLE_TASK_EVENT_COLS,
                                 compression="gzip", nrows=max_tasks * 5)
            else:
                df = pd.read_csv(f, header=None, names=GOOGLE_TASK_EVENT_COLS,
                                 nrows=max_tasks * 5)
            dfs.append(df)
        except Exception as e:
            logger.warning(f"  Error reading {f.name}: {e}")

    if not dfs:
        logger.error("Could not read any Google trace files")
        return None

    events = pd.concat(dfs, ignore_index=True)
    logger.info(f"  Total events loaded: {len(events)}")

    # ── Filter SUBMIT events (event_type = 0) ──
    submits = events[events["event_type"] == 0].copy()
    logger.info(f"  SUBMIT events: {len(submits)}")

    # ── Filter FINISH events (event_type = 4) for duration calculation ──
    finishes = events[events["event_type"] == 4][["job_id", "task_index", "timestamp"]].copy()
    finishes.rename(columns={"timestamp": "finish_time"}, inplace=True)

    # ── Merge to get durations ──
    merged = submits.merge(finishes, on=["job_id", "task_index"], how="left")

    # ── Clean and convert ──
    # Timestamps are in microseconds, convert to seconds
    merged["arrival_time"] = merged["timestamp"] / 1e6
    merged["finish_time_sec"] = merged["finish_time"] / 1e6

    # Duration = finish - submit (use median if missing)
    merged["duration"] = merged["finish_time_sec"] - merged["arrival_time"]
    median_duration = merged["duration"].dropna().median()
    if pd.isna(median_duration) or median_duration <= 0:
        median_duration = 30.0
    merged["duration"] = merged["duration"].fillna(median_duration)
    merged["duration"] = merged["duration"].clip(lower=1.0, upper=600.0)

    # Normalize arrival times to start from 0
    min_arrival = merged["arrival_time"].min()
    merged["arrival_time"] = merged["arrival_time"] - min_arrival

    # Resource requests (already normalized 0-1 in Google trace)
    # Scale to our simulator's range
    merged["cpu_request"] = merged["cpu_request"].fillna(0.05) * 100.0  # Scale to 0-100
    merged["memory_request"] = merged["memory_request"].fillna(0.02) * 64.0  # Scale to 0-64 GB
    merged["disk_request"] = merged["disk_request"].fillna(0.0) * 500.0

    # Deadline = arrival + duration * slack_factor
    rng = np.random.default_rng(42)
    slack = rng.uniform(1.2, 3.0, size=len(merged))
    merged["deadline"] = merged["arrival_time"] + merged["duration"] * slack

    # Priority mapping (Google: 0-11, ours: 1-3)
    merged["priority"] = merged["priority"].fillna(5).clip(0, 11)
    merged["our_priority"] = np.where(merged["priority"] >= 9, 3,
                              np.where(merged["priority"] >= 5, 2, 1))

    # Scheduling class → task type mapping
    merged["scheduling_class"] = merged["scheduling_class"].fillna(0)
    type_map = {0: "mixed", 1: "compute", 2: "memory", 3: "io"}
    merged["task_type"] = merged["scheduling_class"].map(type_map).fillna("mixed")

    # ── Select and sort ──
    result = merged[[
        "arrival_time", "cpu_request", "memory_request", "disk_request",
        "duration", "deadline", "our_priority", "task_type",
        "job_id", "task_index"
    ]].copy()
    result = result.sort_values("arrival_time").head(max_tasks).reset_index(drop=True)
    result["task_id"] = range(len(result))

    # ── Save ──
    out_path = Path(output_dir) / "google_trace_processed.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False)
    logger.info(f"  Saved {len(result)} tasks to {out_path}")

    # Also save as .npz for fast loading
    npz_path = Path(output_dir) / "google_trace_processed.npz"
    np.savez_compressed(
        npz_path,
        arrival_time=result["arrival_time"].values,
        cpu_request=result["cpu_request"].values,
        memory_request=result["memory_request"].values,
        disk_request=result["disk_request"].values,
        duration=result["duration"].values,
        deadline=result["deadline"].values,
        priority=result["our_priority"].values,
    )
    logger.info(f"  Saved .npz to {npz_path}")

    return result


# ══════════════════════════════════════════════════════════
# ALIBABA CLUSTER TRACE (2018)
# ══════════════════════════════════════════════════════════
# batch_task.csv columns:
#  task_name, instance_num, job_name, task_type, status,
#  start_time, end_time, plan_cpu, plan_mem

def preprocess_alibaba_trace(raw_dir: str, output_dir: str, max_tasks: int = 10000):
    """Parse Alibaba Cluster Trace 2018 batch_task data."""
    logger.info("=" * 60)
    logger.info("Processing Alibaba Cluster Trace (2018)")
    logger.info("=" * 60)

    raw_path = Path(raw_dir)
    files = list(raw_path.glob("batch_task*.csv*"))

    if not files:
        logger.error(
            "No Alibaba trace files found!\n"
            "Expected: data/raw/batch_task.csv\n\n"
            "Download from: https://github.com/alibaba/clusterdata/tree/master/cluster-trace-v2018"
        )
        return None

    logger.info(f"Reading: {files[0].name}")
    try:
        df = pd.read_csv(files[0], nrows=max_tasks * 3)
    except Exception as e:
        logger.error(f"Error: {e}")
        return None

    # Detect columns
    cols = df.columns.tolist()
    logger.info(f"  Columns: {cols}")

    # Map columns (Alibaba uses different naming conventions)
    if "start_time" in cols:
        df["arrival_time"] = pd.to_numeric(df["start_time"], errors="coerce")
    elif len(cols) >= 6:
        df["arrival_time"] = pd.to_numeric(df.iloc[:, 5], errors="coerce")

    if "end_time" in cols:
        df["end_time_val"] = pd.to_numeric(df["end_time"], errors="coerce")
    elif len(cols) >= 7:
        df["end_time_val"] = pd.to_numeric(df.iloc[:, 6], errors="coerce")

    if "plan_cpu" in cols:
        df["cpu_request"] = pd.to_numeric(df["plan_cpu"], errors="coerce")
    elif len(cols) >= 8:
        df["cpu_request"] = pd.to_numeric(df.iloc[:, 7], errors="coerce")

    if "plan_mem" in cols:
        df["memory_request"] = pd.to_numeric(df["plan_mem"], errors="coerce")
    elif len(cols) >= 9:
        df["memory_request"] = pd.to_numeric(df.iloc[:, 8], errors="coerce")

    # Clean
    df = df.dropna(subset=["arrival_time"]).copy()
    df["arrival_time"] = df["arrival_time"] - df["arrival_time"].min()

    df["duration"] = (df.get("end_time_val", df["arrival_time"] + 30) - df["arrival_time"]).clip(1, 600)
    df["duration"] = df["duration"].fillna(30.0)

    df["cpu_request"] = df["cpu_request"].fillna(10).clip(1, 100)
    df["memory_request"] = df["memory_request"].fillna(2).clip(0.1, 64)
    df["disk_request"] = 0.0

    rng = np.random.default_rng(42)
    df["deadline"] = df["arrival_time"] + df["duration"] * rng.uniform(1.2, 3.0, size=len(df))
    df["our_priority"] = 1
    df["task_type"] = "mixed"
    df["task_id"] = range(len(df))

    result = df[[
        "task_id", "arrival_time", "cpu_request", "memory_request", "disk_request",
        "duration", "deadline", "our_priority", "task_type"
    ]].sort_values("arrival_time").head(max_tasks).reset_index(drop=True)

    out_path = Path(output_dir) / "alibaba_trace_processed.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False)

    npz_path = Path(output_dir) / "alibaba_trace_processed.npz"
    np.savez_compressed(
        npz_path,
        arrival_time=result["arrival_time"].values,
        cpu_request=result["cpu_request"].values,
        memory_request=result["memory_request"].values,
        disk_request=result["disk_request"].values,
        duration=result["duration"].values,
        deadline=result["deadline"].values,
        priority=result["our_priority"].values,
    )

    logger.info(f"  Saved {len(result)} tasks to {out_path}")
    return result


# ══════════════════════════════════════════════════════════
# SWF FORMAT (HPC2N + NASA)
# ══════════════════════════════════════════════════════════
# Standard Workload Format columns (space-separated, lines starting with ; are comments):
#  0: Job Number
#  1: Submit Time (seconds from start)
#  2: Wait Time
#  3: Run Time (actual, seconds)
#  4: Number of Allocated Processors
#  5: Average CPU Time Used
#  6: Used Memory (KB)
#  7: Requested Number of Processors
#  8: Requested Time (seconds)
#  9: Requested Memory (KB)
# 10: Status (1=completed, 0=killed/failed, 5=cancelled)
# 11: User ID
# 12: Group ID
# 13: Executable Number
# 14: Queue Number
# 15: Partition Number
# 16: Preceding Job Number
# 17: Think Time

def preprocess_swf_trace(filepath: str, output_dir: str, dataset_name: str, max_tasks: int = 10000):
    """Parse SWF (Standard Workload Format) trace files."""
    logger.info("=" * 60)
    logger.info(f"Processing SWF trace: {dataset_name}")
    logger.info("=" * 60)

    fpath = Path(filepath)
    if not fpath.exists():
        # Try .gz version
        gz_path = fpath.with_suffix(fpath.suffix + ".gz")
        if gz_path.exists():
            fpath = gz_path
        else:
            logger.error(f"File not found: {filepath}")
            logger.error(f"Download from: https://www.cs.huji.ac.il/labs/parallel/workload/")
            return None

    # Read SWF file
    rows = []
    open_fn = gzip.open if str(fpath).endswith(".gz") else open
    mode = "rt" if str(fpath).endswith(".gz") else "r"

    with open_fn(fpath, mode) as f:
        for line in f:
            line = line.strip()
            if line.startswith(";") or not line:
                continue
            parts = line.split()
            if len(parts) >= 10:
                rows.append(parts[:18])

    if not rows:
        logger.error("No valid data rows found in SWF file")
        return None

    cols = [
        "job_id", "submit_time", "wait_time", "run_time", "num_procs_alloc",
        "avg_cpu_time", "used_memory", "num_procs_req", "requested_time",
        "requested_memory", "status", "user_id", "group_id", "executable",
        "queue", "partition", "preceding_job", "think_time"
    ]

    df = pd.DataFrame(rows, columns=cols[:len(rows[0])])
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    logger.info(f"  Raw rows: {len(df)}")

    # Filter: only completed jobs, positive runtime
    df = df[(df["status"] == 1) & (df["run_time"] > 0)].copy()
    df = df[df["submit_time"] >= 0]

    # Convert to our format
    df["arrival_time"] = df["submit_time"] - df["submit_time"].min()

    # CPU request: number of processors * some scaling
    df["cpu_request"] = (df["num_procs_req"].fillna(1).clip(1, 128) / 128.0 * 100.0).clip(1, 100)

    # Memory: convert KB to GB, scale to our range
    df["memory_request"] = (df["requested_memory"].fillna(1024) / (1024 * 1024) * 64).clip(0.1, 64)

    # Duration in seconds
    df["duration"] = df["run_time"].clip(1, 86400)  # Cap at 24 hours

    # Scale durations to reasonable simulator range (1-600 seconds)
    max_dur = df["duration"].quantile(0.95)
    if max_dur > 600:
        df["duration"] = (df["duration"] / max_dur * 600).clip(1, 600)

    # Deadline
    rng = np.random.default_rng(42)
    df["deadline"] = df["arrival_time"] + df["duration"] * rng.uniform(1.2, 3.0, size=len(df))

    # Scale arrival times to reasonable range
    max_arrival = df["arrival_time"].max()
    if max_arrival > 7200:  # If spans more than 2 hours
        df["arrival_time"] = (df["arrival_time"] / max_arrival * 3600)  # Compress to 1 hour
        df["deadline"] = df["arrival_time"] + df["duration"] * rng.uniform(1.2, 3.0, size=len(df))

    df["disk_request"] = 0.0
    df["our_priority"] = 1
    df["task_type"] = "mixed"
    df["task_id"] = range(len(df))

    result = df[[
        "task_id", "arrival_time", "cpu_request", "memory_request", "disk_request",
        "duration", "deadline", "our_priority", "task_type"
    ]].sort_values("arrival_time").head(max_tasks).reset_index(drop=True)
    result["task_id"] = range(len(result))

    out_path = Path(output_dir) / f"{dataset_name}_processed.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False)

    npz_path = Path(output_dir) / f"{dataset_name}_processed.npz"
    np.savez_compressed(
        npz_path,
        arrival_time=result["arrival_time"].values,
        cpu_request=result["cpu_request"].values,
        memory_request=result["memory_request"].values,
        disk_request=result["disk_request"].values,
        duration=result["duration"].values,
        deadline=result["deadline"].values,
        priority=result["our_priority"].values,
    )

    logger.info(f"  Saved {len(result)} tasks to {out_path}")
    return result


# ══════════════════════════════════════════════════════════
# LOAD PROCESSED DATA
# ══════════════════════════════════════════════════════════

def load_processed_trace(filepath: str):
    """
    Load a processed trace file (.csv or .npz) and return Task objects.
    
    This is what the simulator calls to get real workload data.
    """
    # Add parent path for imports
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from simulator.task import Task, TaskType

    fpath = Path(filepath)

    if fpath.suffix == ".npz":
        data = np.load(fpath)
        n = len(data["arrival_time"])
        tasks = []
        type_list = [TaskType.COMPUTE, TaskType.MEMORY, TaskType.IO, TaskType.MIXED]
        rng = np.random.default_rng(42)

        for i in range(n):
            tasks.append(Task(
                task_id=i,
                arrival_time=float(data["arrival_time"][i]),
                cpu_req=float(data["cpu_request"][i]),
                memory_req=float(data["memory_request"][i]),
                disk_req=float(data["disk_request"][i]) if "disk_request" in data else 0.0,
                duration=float(data["duration"][i]),
                deadline=float(data["deadline"][i]),
                task_type=rng.choice(type_list),
                priority=int(data["priority"][i]) if "priority" in data else 1,
            ))
        return tasks

    elif fpath.suffix == ".csv":
        df = pd.read_csv(fpath)
        tasks = []
        type_map = {"compute": TaskType.COMPUTE, "memory": TaskType.MEMORY,
                     "io": TaskType.IO, "mixed": TaskType.MIXED}

        for i, row in df.iterrows():
            tasks.append(Task(
                task_id=int(row.get("task_id", i)),
                arrival_time=float(row["arrival_time"]),
                cpu_req=float(row["cpu_request"]),
                memory_req=float(row["memory_request"]),
                disk_req=float(row.get("disk_request", 0)),
                duration=float(row["duration"]),
                deadline=float(row["deadline"]),
                task_type=type_map.get(str(row.get("task_type", "mixed")), TaskType.MIXED),
                priority=int(row.get("our_priority", 1)),
            ))
        return tasks

    else:
        raise ValueError(f"Unsupported file format: {fpath.suffix}")


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="GreenCloudRL Data Preprocessing")
    parser.add_argument("--dataset", type=str, default="all",
                        choices=["google", "alibaba", "hpc2n", "nasa", "all"],
                        help="Which dataset to process")
    parser.add_argument("--input", type=str, default="data/raw",
                        help="Raw data directory")
    parser.add_argument("--output", type=str, default="data/processed",
                        help="Output directory for processed data")
    parser.add_argument("--max-tasks", type=int, default=10000,
                        help="Maximum tasks to extract per dataset")
    args = parser.parse_args()

    Path(args.output).mkdir(parents=True, exist_ok=True)

    if args.dataset in ("google", "all"):
        preprocess_google_trace(args.input, args.output, args.max_tasks)

    if args.dataset in ("alibaba", "all"):
        preprocess_alibaba_trace(args.input, args.output, args.max_tasks)

    if args.dataset in ("hpc2n", "all"):
        # Try common filenames
        for name in ["HPC2N-2002-2.2-cln.swf", "HPC2N-2002-2.2-cln.swf.gz", "hpc2n.swf", "hpc2n.swf.gz"]:
            fpath = Path(args.input) / name
            if fpath.exists():
                preprocess_swf_trace(str(fpath), args.output, "hpc2n", args.max_tasks)
                break
        else:
            logger.warning("HPC2N file not found in data/raw/. Skipping.")

    if args.dataset in ("nasa", "all"):
        for name in ["NASA-iPSC-1993-3.1-cln.swf", "NASA-iPSC-1993-3.1-cln.swf.gz", "nasa.swf", "nasa.swf.gz"]:
            fpath = Path(args.input) / name
            if fpath.exists():
                preprocess_swf_trace(str(fpath), args.output, "nasa", args.max_tasks)
                break
        else:
            logger.warning("NASA file not found in data/raw/. Skipping.")

    logger.info("\n" + "=" * 60)
    logger.info("Preprocessing complete!")
    logger.info("Processed files:")
    for f in sorted(Path(args.output).glob("*_processed.*")):
        logger.info(f"  {f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
