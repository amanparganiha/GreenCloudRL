# GreenCloudRL 🌱⚡

**Hierarchical Meta-Reinforcement Learning for Energy-Efficient Cloud Task Scheduling**

A novel framework combining **Hierarchical RL** (A2C + PPO), **Meta-Learning** (Reptile), and **SHAP Explainability** to create an adaptive, interpretable cloud scheduler that generalizes to unseen workloads.

---

## Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/YOUR_USERNAME/GreenCloudRL.git
cd GreenCloudRL

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate   # Linux/Mac
# venv\Scripts\activate    # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the full pipeline
python main.py

# Or run specific stages:
python main.py --stage 1          # Baselines only
python main.py --stage 2          # Single-level DRL
python main.py --stage 3          # Hierarchical DRL
python main.py --stage 4          # Meta-learning
python main.py --stage 5          # Generate plots
python main.py --stage 6          # Explainability
python main.py --stage 1 2 3 4 5  # Everything except explainability
```

---

## Project Structure

```
GreenCloudRL/
├── main.py                    # Entry point - runs full pipeline
├── configs/
│   └── default.yaml           # All hyperparameters
├── simulator/
│   ├── cloud_env.py           # Gymnasium-compatible environment
│   ├── server.py              # Server & VM models
│   ├── task.py                # Task model
│   ├── energy_model.py        # Power consumption model
│   ├── workload_generator.py  # Synthetic + trace workloads
│   └── sla_tracker.py         # SLA violation tracking
├── agents/
│   ├── networks.py            # Actor, Critic, PPO networks
│   ├── low_level_a2c.py       # A2C for task assignment
│   ├── high_level_ppo.py      # PPO for server management
│   └── hierarchical_agent.py  # Two-level coordinator
├── meta_learning/
│   └── reptile.py             # Reptile meta-learning
├── explainability/
│   └── shap_analyzer.py       # SHAP analysis + explanations
├── baselines/
│   └── schedulers.py          # Random, RR, FCFS, Least-Loaded, SJF
├── training/
│   ├── train_hierarchical.py  # HRL training loop
│   ├── train_meta.py          # Meta-training loop
│   └── evaluate.py            # Evaluation + plotting
└── results/
    ├── figures/               # Generated plots
    ├── tables/                # CSV results
    └── checkpoints/           # Saved models
```

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   GreenCloudRL                       │
│                                                      │
│  ┌─────────────┐     embedding     ┌──────────────┐ │
│  │ High-Level   │ ──────────────> │  Low-Level    │ │
│  │ PPO Agent    │                  │  A2C Agent    │ │
│  │ (Server Mgmt)│                  │  (Task→VM)    │ │
│  └──────┬───────┘                  └──────┬────────┘ │
│         │ every N steps                    │ per task │
│  ┌──────▼──────────────────────────────────▼───────┐ │
│  │           SimPy Cloud Simulator                  │ │
│  │  [Servers] [VMs] [Tasks] [Energy] [SLA]         │ │
│  └──────────────────────────────────────────────────┘ │
│                                                      │
│  ┌──────────────┐              ┌───────────────────┐ │
│  │   Reptile     │              │  SHAP Explainer   │ │
│  │ Meta-Learning │              │  (Post-training)  │ │
│  └──────────────┘              └───────────────────┘ │
└─────────────────────────────────────────────────────┘
```

---

## Key Features

| Feature | Implementation | Purpose |
|---------|---------------|---------|
| Hierarchical RL | A2C (low) + PPO (high) | Reduces state-action space |
| Meta-Learning | Reptile algorithm | Fast adaptation to new workloads |
| Explainability | SHAP values + NL templates | Interpretable decisions |
| Realistic Sim | SimPy + linear power model | Publication-grade results |
| Action Masking | Invalid VM masking | Prevents impossible assignments |
| Multi-workload | 7+ distributions | Diverse training and testing |

---

## How to Add Real Trace Data

1. Download traces (see `data/` folder README)
2. Place CSV files in `data/raw/`
3. Use the workload generator:

```python
from simulator import WorkloadGenerator
gen = WorkloadGenerator(seed=42)
tasks = gen.load_google_trace("data/raw/google_trace.csv", num_tasks=5000)
```

---

## Configuration

All hyperparameters are in `configs/default.yaml`. Key ones to tune:

```yaml
# Reward weights (must sum to ~1.0)
reward:
  alpha: 0.4   # Makespan importance
  beta: 0.4    # Energy importance
  gamma: 0.2   # SLA importance

# Meta-learning
meta:
  inner_steps: 5         # Gradient updates per task
  tasks_per_batch: 4     # Tasks per meta-step
  meta_lr: 1.0           # Initial meta learning rate
```

---

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{greencloudrl2026,
  title={GreenCloudRL: Hierarchical Meta-Reinforcement Learning 
         for Energy-Efficient Cloud Task Scheduling},
  author={Your Name},
  booktitle={IEEE International Conference on Cloud Engineering (IC2E)},
  year={2026}
}
```

---

## License

MIT License
