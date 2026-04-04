#!/usr/bin/env python3
"""Quick smoke test to verify all components work."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

print("=" * 60)
print("  GreenCloudRL - Smoke Test")
print("=" * 60)

# 1. Test Simulator
print("\n[1/7] Testing Simulator...")
from simulator import CloudSchedulingEnv, WorkloadGenerator
env = CloudSchedulingEnv(num_servers=5, vms_per_server=3, seed=42)
obs, info = env.reset()
print(f"  ✓ Env created. Obs shape: {obs.shape}, Action space: {env.action_space.n}")

# Test step
mask = info.get("action_mask", None)
valid = np.where(mask > 0)[0] if mask is not None else [0]
action = int(valid[0]) if len(valid) > 0 else 0
obs2, reward, term, trunc, info2 = env.step(action)
print(f"  ✓ Step executed. Reward: {reward:.4f}, Done: {term or trunc}")

# 2. Test Workload Generator
print("\n[2/7] Testing Workload Generator...")
gen = WorkloadGenerator(seed=42)
tasks = gen.generate_synthetic(num_tasks=100)
print(f"  ✓ Generated {len(tasks)} synthetic tasks")
meta_tasks = gen.create_meta_tasks(num_distributions=3, tasks_per_distribution=50)
print(f"  ✓ Created {len(meta_tasks)} meta-training distributions")

# 3. Test Neural Networks
print("\n[3/7] Testing Neural Networks...")
from agents.networks import ActorNetwork, CriticNetwork, PPOActorCritic
device = "cpu"
actor = ActorNetwork(obs.shape[0], env.action_space.n, [64, 64]).to(device)
critic = CriticNetwork(obs.shape[0], [64, 64]).to(device)
state_t = torch.FloatTensor(obs).unsqueeze(0)
dist = actor(state_t)
value = critic(state_t)
print(f"  ✓ Actor output: {dist.probs.shape}, Critic output: {value.shape}")

ppo_net = PPOActorCritic(12, 13, [64, 64]).to(device)
hl_obs = torch.randn(1, 12)
dist_hl, val_hl, emb_hl = ppo_net(hl_obs)
print(f"  ✓ PPO output: dist={dist_hl.probs.shape}, value={val_hl.shape}, embed={emb_hl.shape}")

# 4. Test Low-Level A2C
print("\n[4/7] Testing Low-Level A2C Agent...")
from agents import LowLevelA2C
agent = LowLevelA2C(obs.shape[0], env.action_space.n, hidden_sizes=[64, 64], device=device)
action, log_prob, val = agent.select_action(obs, mask)
print(f"  ✓ Action: {action}, LogProb: {log_prob:.4f}, Value: {val:.4f}")

# Mini training loop
obs, info = env.reset()
for step in range(20):
    mask = info.get("action_mask", None)
    action, lp, v = agent.select_action(obs, mask)
    next_obs, reward, term, trunc, info = env.step(action)
    agent.store_transition(obs, action, reward, lp, v, term or trunc, mask)
    obs = next_obs
    if len(agent.buffer) >= 5:
        metrics = agent.update(next_obs)
    if term or trunc:
        break

print(f"  ✓ Training loop ran {step+1} steps successfully")

# 5. Test High-Level PPO
print("\n[5/7] Testing High-Level PPO Agent...")
from agents import HighLevelPPO
hl_agent = HighLevelPPO(state_dim=12, action_dim=13, hidden_sizes=[64, 64], device=device)
hl_obs = env.get_high_level_observation()
hl_action, hl_lp, hl_v, hl_emb = hl_agent.select_action(hl_obs)
print(f"  ✓ HL Action: {hl_action}, Embedding shape: {hl_emb.shape}")

# 6. Test Meta-Learning
print("\n[6/7] Testing Reptile Meta-Learning...")
from meta_learning import ReptileMetaLearner

small_env = CloudSchedulingEnv(num_servers=3, vms_per_server=2, seed=42)
small_agent = LowLevelA2C(
    small_env.observation_space.shape[0], small_env.action_space.n,
    hidden_sizes=[32, 32], device=device,
)
meta = ReptileMetaLearner(
    small_agent, small_env,
    inner_steps=2, tasks_per_batch=2, device=device,
)
small_tasks = gen.create_meta_tasks(num_distributions=3, tasks_per_distribution=30)
history = meta.meta_train(small_tasks, num_meta_iterations=3, log_interval=1)
print(f"  ✓ Meta-training completed: {len(history['inner_rewards'])} iterations")

# 7. Test Baselines
print("\n[7/7] Testing Baselines...")
from baselines import get_all_baselines
baselines = get_all_baselines(total_vms=15, vms_per_server=3)
obs, info = small_env.reset()
for bl in baselines:
    mask = info.get("action_mask", None)
    a = bl.select_action(obs, mask)
    print(f"  ✓ {bl.name}: action={a}")

# Summary
print("\n" + "=" * 60)
print("  ALL TESTS PASSED ✓")
print("=" * 60)
print(f"\nDevice: {device}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"\nProject is ready! Run: python main.py")
