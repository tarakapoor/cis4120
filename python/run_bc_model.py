#!/usr/bin/env python3
"""
Simple BC model runner - loads raw state_dict and runs in environment
"""
import argparse
import json
import torch
import torch.nn as nn
import numpy as np

# Import environment
import hard_stable

# BC Model (simplified, no batch norm for now)
class MLP_no_batch_bias(nn.Module):
    def __init__(self, obs_dim, act_dim, chunk_len=1, hidden_size=32):
        super().__init__()
        output_dim = act_dim * chunk_len
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size, bias=False),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size, bias=False),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size, bias=False),
            nn.GELU(),
            nn.Linear(hidden_size, output_dim, bias=False)
        )
    def forward(self, x): return self.net(x)

def main():
    parser = argparse.ArgumentParser(description='Run BC policy')
    parser.add_argument('--policy', type=str, required=True, help='Path to state_dict .pt file')
    parser.add_argument('--env', type=str, default='hard_stable', help='Environment name')
    parser.add_argument('--num_traj', type=int, default=5, help='Number of trajectories')
    parser.add_argument('--max_steps', type=int, default=300, help='Max steps per trajectory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--capture_activations', action='store_true', help='Capture layer activations')

    args = parser.parse_args()

    # Load state dict
    state_dict = torch.load(args.policy, map_location='cpu')

    # Infer dimensions from state_dict
    first_weight = state_dict['net.0.weight']
    last_weight = state_dict['net.6.weight']

    hidden_size = first_weight.shape[0]
    obs_dim = first_weight.shape[1]
    act_dim = last_weight.shape[0]

    print(f"Loaded model: obs_dim={obs_dim}, hidden={hidden_size}, act_dim={act_dim}")

    # Create model and load weights
    model = MLP_no_batch_bias(obs_dim, act_dim, hidden_size=hidden_size)
    model.load_state_dict(state_dict)
    model.eval()

    # Create environment
    if args.env == 'hard_stable':
        # Load expert perturbation model
        expert_path = './experts/hard_stable_perturb.pt'
        try:
            expert_sd = torch.load(expert_path)
        except:
            # Create dummy if doesn't exist
            env_temp = hard_stable.create_gym_environment(d=4, pair_first=True, pytorch_seed=args.seed)
            expert_sd = env_temp.perturbation_model.state_dict()

        env = hard_stable.create_gym_environment(d=4, pair_first=True, pytorch_seed=args.seed, state_dict=expert_sd)
    else:
        # Try as gymnasium environment
        try:
            import gymnasium as gym
            env = gym.make(args.env, render_mode=None)
            env.reset(seed=args.seed)
            print(f"Created {args.env} environment")
        except:
            raise ValueError(f"Environment {args.env} not supported. Try 'Walker2d-v4', 'HalfCheetah-v4', etc.")

    # Setup activation capture if requested
    activations_data = []
    hooks = []

    if args.capture_activations:
        def make_hook(layer_idx):
            def hook(module, input, output):
                activations_data[-1][f'layer_{layer_idx}'] = output.detach().cpu().numpy().tolist()
            return hook

        # Register hooks for each layer
        for idx, (name, module) in enumerate(model.net.named_children()):
            if isinstance(module, nn.Linear):
                hooks.append(module.register_forward_hook(make_hook(idx)))

    # Run trajectories
    total_rewards = []
    traj_lengths = []

    for ep in range(args.num_traj):
        obs, _ = env.reset(seed=args.seed + ep)
        ep_reward = 0
        steps = 0

        for t in range(args.max_steps):
            if args.capture_activations and ep == 0:  # Only capture first trajectory
                activations_data.append({'timestep': t, 'observation': obs.tolist()})

            with torch.no_grad():
                obs_t = torch.from_numpy(obs).float().unsqueeze(0)
                action = model(obs_t).squeeze(0).numpy()

            if args.capture_activations and ep == 0:
                activations_data[-1]['action'] = action.tolist()

            obs, reward, done, truncated, _ = env.step(action)
            ep_reward += reward
            steps += 1

            if done or truncated:
                break

        total_rewards.append(ep_reward)
        traj_lengths.append(steps)

    # Remove hooks
    for h in hooks:
        h.remove()

    # Compute statistics
    results = {
        'num_trajectories': len(total_rewards),
        'avg_reward': float(np.mean(total_rewards)),
        'max_reward': float(np.max(total_rewards)),
        'min_reward': float(np.min(total_rewards)),
        'avg_length': float(np.mean(traj_lengths)),
        'trajectory_rewards': [float(r) for r in total_rewards],
        'trajectory_lengths': traj_lengths
    }

    if args.capture_activations:
        results['activations'] = activations_data

    print("\nResults:")
    print(f"  Average reward: {results['avg_reward']:.2f}")
    print(f"  Max reward: {results['max_reward']:.2f}")
    print(f"  Min reward: {results['min_reward']:.2f}")
    print(f"  Average trajectory length: {results['avg_length']:.2f}")
    print(f"\nJSON Output:")
    print(json.dumps(results, indent=2))

    return results

if __name__ == '__main__':
    main()
